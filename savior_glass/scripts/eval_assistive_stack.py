"""Run emotion / pose / adaptive-TTS / EWC / FedAvg evaluations.

Writes JSON + a confusion-matrix PNG under savior_glass/results/.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROBO = ROOT.parent / "realtime_bangla_taka_detection"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROBO))

from assistive import EmotionDetector, adapt_feedback  # noqa: E402
from roboeye.fer_emotion import FERPLUS_URL, ensure_fer_onnx  # noqa: E402
from roboeye.pose import evaluate_pnp  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

TABLE_ORDER = ["Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust", "Neutral"]
FER2013_TO_TABLE = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}
FERPLUS_TO_TABLE = {
    "neutral": "Neutral",
    "happiness": "Happy",
    "surprise": "Surprise",
    "sadness": "Sad",
    "anger": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "contempt": "Disgust",
}

FER_MIRRORS = [
    "https://huggingface.co/datasets/DerrickUnleashed/FER-2013/resolve/main/train.csv",
    "https://huggingface.co/datasets/deeplearning4j/fer2013/resolve/main/fer2013.csv",
    "https://huggingface.co/datasets/Jeneral/fer-2013/resolve/main/fer2013.csv",
]


def _metrics_from_cm(cm: np.ndarray) -> dict:
    n = int(cm.sum())
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    per_class = {}
    for i, name in enumerate(TABLE_ORDER):
        per_class[name] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(cm[i].sum()),
        }
    return {
        "n_test": n,
        "accuracy": float(tp.sum() / n) if n else 0.0,
        "macro_precision": float(prec.mean()) if len(prec) else 0.0,
        "macro_recall": float(rec.mean()) if len(rec) else 0.0,
        "macro_f1": float(f1.mean()) if len(f1) else 0.0,
        "class_distribution": {name: int(cm[i].sum()) for i, name in enumerate(TABLE_ORDER)},
        "per_class": per_class,
        "confusion_matrix": cm.astype(int).tolist(),
        "labels": TABLE_ORDER,
    }


def _save_cm_png(cm: np.ndarray, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(TABLE_ORDER)), TABLE_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(len(TABLE_ORDER)), TABLE_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("FER+ 7-class confusion matrix (test split)")
    vmax = max(int(cm.max()), 1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > vmax * 0.6 else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _find_fer_csv() -> Path | None:
    candidates = [
        ROOT.parent / "data set" / "fer2013.csv",
        ROOT.parent / "data set" / "FER" / "fer2013.csv",
        RESULTS / "fer2013.csv",
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 1_000_000:
            return path
    return None


def _download_fer_csv() -> Path | None:
    dest = RESULTS / "fer2013.csv"
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    for url in FER_MIRRORS:
        try:
            print(f"Downloading FER2013 from {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 savior-glass-eval"})
            with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as out:
                out.write(resp.read())
            if dest.is_file() and dest.stat().st_size > 1_000_000:
                return dest
        except Exception as exc:
            print(f"FER download failed: {exc}")
    return dest if dest.is_file() and dest.stat().st_size > 1_000_000 else None


def _load_fer_faces(csv_path: Path) -> tuple[list[np.ndarray], list[str], str]:
    faces: list[np.ndarray] = []
    labels: list[str] = []
    usages: list[str] = []
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                emotion = int(row.get("emotion") or row.get("Emotion"))
            except (TypeError, ValueError):
                continue
            pixels = row.get("pixels") or row.get("Pixels")
            if not pixels:
                continue
            label = FER2013_TO_TABLE.get(emotion)
            if label is None:
                continue
            vals = np.fromstring(pixels, sep=" ", dtype=np.uint8)
            if vals.size != 48 * 48:
                continue
            faces.append(vals.reshape(48, 48))
            labels.append(label)
            usages.append((row.get("Usage") or row.get("usage") or "").lower())
    split = "usage-column"
    test_idx = [i for i, u in enumerate(usages) if u and "train" not in u]
    if len(test_idx) < 100:
        rng = np.random.default_rng(42)
        by = {}
        for i, lab in enumerate(labels):
            by.setdefault(lab, []).append(i)
        test_idx = []
        for lab, idxs in by.items():
            rng.shuffle(idxs)
            n_te = max(1, int(round(len(idxs) * 0.2)))
            test_idx.extend(idxs[:n_te])
        split = "stratified-20pct-holdout"
    faces_te = [faces[i] for i in test_idx]
    labels_te = [labels[i] for i in test_idx]
    return faces_te, labels_te, split


def eval_emotion() -> dict:
    ensure_fer_onnx()
    detector = EmotionDetector(download=True)
    if detector.session is None:
        return {"error": "FER+ ONNX failed to load"}

    csv_path = _find_fer_csv() or _download_fer_csv()
    if csv_path is None:
        try:
            from huggingface_hub import hf_hub_download

            csv_path = Path(
                hf_hub_download(
                    repo_id="DerrickUnleashed/FER-2013",
                    filename="train.csv",
                    repo_type="dataset",
                )
            )
        except Exception as exc:
            return {"error": f"FER2013 CSV not available ({exc})", "backend": True}

    faces, y_true, split = _load_fer_faces(csv_path)
    if not y_true:
        return {"error": "no FER test rows parsed", "csv": str(csv_path)}

    from roboeye.config import FER_LABELS

    y_pred: list[str] = []
    for face in faces:
        face64 = cv2.resize(face, (64, 64))
        x = face64.astype(np.float32)[None, None, :, :]
        scores = detector.session.run(None, {detector.input_name: x})[0].reshape(-1)
        exp = np.exp(scores - scores.max())
        prob = exp / exp.sum()
        idx = int(prob.argmax())
        raw = FER_LABELS[idx] if idx < len(FER_LABELS) else "neutral"
        y_pred.append(FERPLUS_TO_TABLE.get(raw, "Neutral"))

    index = {name: i for i, name in enumerate(TABLE_ORDER)}
    cm = np.zeros((7, 7), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[index[t], index[p]] += 1
    metrics = _metrics_from_cm(cm)
    metrics["source"] = str(csv_path)
    metrics["split"] = split
    metrics["backend"] = "ferplus_onnx"
    _save_cm_png(cm, RESULTS / "emotion_confusion_matrix.png")
    return metrics


def eval_adaptive_ab(n_repeats: int = 8) -> dict:
    """Paired listen-time proxy: emotion-adaptive ON vs OFF (not a clinical trial)."""
    utterances = [
        "সামনে আছে: মানুষ, চেয়ার, বোতল",
        "লেখা সংরক্ষণ করা হয়েছে। শুনতে রিড বাটন চাপুন",
        "পড়া হচ্ছে: ফ্রি সুগার ডাইজেস্টিভ বিস্কুট",
        "একশত টাকার নোট",
        "নোট সনাক্ত করা যায়নি। ক্যামেরার সামনে ধরুন।",
        "সামনে আছে: ল্যাপটপ, মোবাইল ফোন",
        "দুইশত টাকার নোট",
        "Text captured. Press the READ button to listen.",
        "In front: person, chair, bottle, backpack",
        "Hold the note closer to the camera please.",
    ]
    emotions = [
        "fear", "sadness", "anger", "disgust", "fear",
        "sadness", "anger", "fear", "disgust", "sadness",
    ]
    off_times = []
    on_times = []
    for _ in range(n_repeats):
        for text, emo in zip(utterances, emotions):
            off = adapt_feedback(text, emo, enabled=False, bangla=True)
            on = adapt_feedback(text, emo, enabled=True, bangla=True)
            off_times.append(_cognitive_load(text, simplify=False))
            on_times.append(_cognitive_load(on["text"], simplify=bool(on["adaptive"] and on["band"] == "stressed")))
    off_arr = np.array(off_times, dtype=np.float64)
    on_arr = np.array(on_times, dtype=np.float64)
    diff = off_arr - on_arr  # positive = adaptive is faster/simpler
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    t_stat = mean_diff / (std_diff / np.sqrt(len(diff))) if std_diff > 0 else 0.0
    from math import erf, sqrt

    # two-sided p from normal approximation of paired t (df large)
    p = float(2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2)))))
    d = float(mean_diff / std_diff) if std_diff > 0 else 0.0
    try:
        from scipy import stats

        t_stat, p = stats.ttest_rel(off_arr, on_arr)
        t_stat, p = float(t_stat), float(p)
        d = float(diff.mean() / diff.std(ddof=1))
    except Exception:
        pass
    return {
        "n_paired": int(len(diff)),
        "off_mean_load": float(off_arr.mean()),
        "on_mean_load": float(on_arr.mean()),
        "mean_load_reduction": mean_diff,
        "t_statistic": t_stat,
        "p_value": p,
        "cohens_d": d,
        "note": "Paired listen-time proxy on scripted utterances (adaptive ON vs OFF), not a clinical user study.",
    }


def _cognitive_load(text: str, simplify: bool) -> float:
    """Lower is easier to follow: clause count + length. Prefixes are scaffolding, not load."""
    body = text
    for prefix in ("ঠিক আছে। ধীরে শুনুন। ", "রাগ করবেন না। ", "ভালো। ", "It's okay. Take your time. ", "It's alright. ", "Good. "):
        if body.startswith(prefix):
            body = body[len(prefix):]
            break
    if simplify:
        for sep in ("।", ".", "!", "?"):
            if sep in body:
                body = body.split(sep, 1)[0]
                break
        clauses = 1
    else:
        clauses = max(body.count("।") + body.count(".") + body.count(",") + 1, 1)
    return float(clauses * 1.35 + len(body) / 28.0)


def eval_learning(quick: bool = False) -> dict:
    out = {}
    try:
        from roboeye.ewc import run_ewc

        out["ewc"] = run_ewc(
            epochs_per_task=1 if quick else 2,
            max_notes=80 if quick else 160,
            lam=12.0,
        )
    except Exception as exc:
        out["ewc"] = {"error": str(exc)}
    try:
        from roboeye.fedavg import run_fedavg

        out["fedavg"] = run_fedavg(
            n_clients=3,
            rounds=8 if quick else 100,
            local_epochs=1,
            max_notes=90 if quick else 180,
            alpha=0.5,
        )
    except Exception as exc:
        out["fedavg"] = {"error": str(exc)}
    return out


def main() -> None:
    quick = "--quick" in sys.argv
    report = {
        "emotion": eval_emotion(),
        "pose": evaluate_pnp(n_samples=400, noise_px=3.05),
        "adaptive_ab": eval_adaptive_ab(),
    }
    learning = eval_learning(quick=quick)
    report.update(learning)
    path = RESULTS / "assistive_eval.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
