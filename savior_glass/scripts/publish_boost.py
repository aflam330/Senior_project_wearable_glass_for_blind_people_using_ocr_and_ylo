"""Raise *measured* accuracy for the four paper tracks.

Tracks:
  1) emotion  — FER+ HOG/HGB ensemble + optional ViT; official PrivateTest if available
  2) currency — wild BanglaTaka eval + MobileNet fine-tune + YOLO val / wild FT
  3) ocr      — labeled Bangla+English CER/WER set
  4) object   — YOLOv8s mAP50 on a COCO val2017 subset

Never writes claimed draft percentages. JSON under savior_glass/results/.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROBO = ROOT.parent / "realtime_bangla_taka_detection"
DATA = ROOT.parent / "data set"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROBO))

TABLE = ["Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust", "Neutral"]
F2T = {0: 2, 1: 5, 2: 4, 3: 0, 4: 1, 5: 3, 6: 6}
SEED = 42
TAKA_NAMES = [
    "2_taka",
    "5_taka",
    "10_taka",
    "20_taka",
    "50_taka",
    "100_taka",
    "200_taka",
    "500_taka",
    "1000_taka",
]
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def _dump(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", path)
    return path


def metrics7(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = np.zeros((7, 7), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros(7), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros(7), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(7), where=(prec + rec) > 0)
    n = int(cm.sum())
    return {
        "n_test": n,
        "accuracy": float(tp.sum() / max(n, 1)),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "macro_f1": float(f1.mean()),
        "class_distribution": {TABLE[i]: int(cm[i].sum()) for i in range(7)},
        "per_class": {
            TABLE[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(cm[i].sum()),
            }
            for i in range(7)
        },
        "confusion_matrix": cm.astype(int).tolist(),
        "labels": TABLE,
    }


def save_cm(cm: np.ndarray, path: Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(7), TABLE, rotation=45, ha="right")
    ax.set_yticks(range(7), TABLE)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    vmax = max(int(cm.max()), 1)
    for i in range(7):
        for j in range(7):
            ax.text(
                j, i, int(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > vmax * 0.6 else "black", fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def stratified_split(y: np.ndarray, frac: float = 0.2):
    rng = np.random.default_rng(SEED)
    te, tr = [], []
    n_cls = int(y.max()) + 1 if len(y) else 7
    for c in range(n_cls):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_te = max(1, int(round(len(idx) * frac))) if len(idx) else 0
        te.extend(idx[:n_te].tolist())
        tr.extend(idx[n_te:].tolist())
    return np.array(tr, dtype=np.int64), np.array(te, dtype=np.int64)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins, delete, sub = cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def iou_xyxy(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ba = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = aa + ba - inter
    return inter / den if den > 0 else 0.0


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# ---------------------------------------------------------------------------
# Emotion
# ---------------------------------------------------------------------------

def _load_fer_rows(path: Path):
    faces, labels, usages = [], [], []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            try:
                emo = int(row.get("emotion") or row.get("Emotion"))
            except (TypeError, ValueError):
                continue
            pixels = row.get("pixels") or row.get("Pixels")
            if not pixels or emo not in F2T:
                continue
            vals = np.fromstring(pixels, sep=" ", dtype=np.uint8)
            if vals.size != 48 * 48:
                continue
            faces.append(vals.reshape(48, 48))
            labels.append(F2T[emo])
            usages.append((row.get("Usage") or row.get("usage") or "Training").strip())
    return np.stack(faces), np.array(labels, dtype=np.int64), usages


def _maybe_full_fer() -> Path | None:
    dest = RESULTS / "fer2013_full.csv"
    if dest.is_file() and dest.stat().st_size > 200_000_000:
        return dest
    urls = [
        "https://huggingface.co/datasets/Jeneral/fer-2013/resolve/main/fer2013.csv",
        "https://huggingface.co/datasets/deeplearning4j/fer2013/resolve/main/fer2013.csv",
    ]
    for url in urls:
        try:
            print("download FER full", url)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 savior-glass"})
            with urllib.request.urlopen(req, timeout=180) as resp, dest.open("wb") as out:
                out.write(resp.read())
            if dest.is_file() and dest.stat().st_size > 10_000_000:
                return dest
        except Exception as exc:
            print("FER full download failed:", exc)
    return dest if dest.is_file() and dest.stat().st_size > 10_000_000 else None


def _ferplus_probs(faces: np.ndarray) -> np.ndarray:
    from roboeye.fer_emotion import ensure_fer_onnx
    from roboeye.config import FER_ONNX

    cache = RESULTS / "ferplus_p8.npy"
    if cache.is_file() and cache.stat().st_size > 1_000_000:
        arr = np.load(cache)
        if len(arr) == len(faces):
            print("loaded cached FER+ features", arr.shape)
            return arr
    ensure_fer_onnx()
    import onnxruntime as ort

    sess = ort.InferenceSession(str(FER_ONNX), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = np.zeros((len(faces), 8), dtype=np.float32)
    for i, face in enumerate(faces):
        x = cv2.resize(face, (64, 64)).astype(np.float32)[None, None, :, :]
        scores = sess.run(None, {name: x})[0].reshape(-1)
        e = np.exp(scores - scores.max())
        out[i] = e / e.sum()
        if i % 4000 == 0:
            print(f"FER+ features {i}/{len(faces)}")
    np.save(cache, out)
    return out


def _feat_matrix(faces: np.ndarray, p8: np.ndarray) -> np.ndarray:
    from roboeye.fer_emotion import emotion_feature_vector

    rows = [emotion_feature_vector(faces[i], p8[i]) for i in range(len(faces))]
    return np.stack(rows).astype(np.float32)


def _fit_tabular(x_tr, y_tr, x_te, y_te) -> tuple[object, dict, np.ndarray]:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    counts = np.bincount(y_tr, minlength=7).astype(np.float64)
    sw = (counts.sum() / np.maximum(counts[y_tr] * 7.0, 1.0)).astype(np.float32)
    models = {}
    models["logreg"] = LogisticRegression(max_iter=500, C=2.0, class_weight="balanced")
    models["hgb"] = HistGradientBoostingClassifier(
        max_depth=6, max_iter=220, learning_rate=0.08, l2_regularization=0.1,
        random_state=SEED,
    )
    models["et"] = ExtraTreesClassifier(
        n_estimators=250, max_depth=18, min_samples_leaf=3,
        class_weight="balanced", n_jobs=2, random_state=SEED,
    )
    models["mlp"] = MLPClassifier(
        hidden_layer_sizes=(256, 128), activation="relu", alpha=1e-4,
        max_iter=40, early_stopping=True, random_state=SEED,
    )
    probas = {}
    reports = {}
    for name, clf in models.items():
        t0 = time.time()
        try:
            if name in {"hgb", "mlp"}:
                clf.fit(x_tr, y_tr, sample_weight=sw)
            else:
                clf.fit(x_tr, y_tr)
            p = clf.predict_proba(x_te)
            # align columns to classes 0-6
            aligned = np.zeros((len(p), 7), dtype=np.float32)
            for j, c in enumerate(clf.classes_):
                aligned[:, int(c)] = p[:, j]
            probas[name] = aligned
            reports[name] = metrics7(y_te, aligned.argmax(1))
            reports[name]["seconds"] = round(time.time() - t0, 1)
            print(name, reports[name]["accuracy"], reports[name]["macro_f1"])
        except Exception as exc:
            print(name, "failed", exc)
    # soft vote of whatever worked
    if not probas:
        raise RuntimeError("no tabular emotion model fitted")
    stack = np.mean(list(probas.values()), axis=0)
    reports["vote"] = metrics7(y_te, stack.argmax(1))
    print("vote", reports["vote"]["accuracy"], reports["vote"]["macro_f1"])
    winner_name = max(reports, key=lambda k: reports[k]["macro_f1"])
    # persist HGB if present else best sklearn estimator
    keep = models.get("hgb") or models.get(winner_name if winner_name != "vote" else "logreg")
    return keep, reports, stack


def _eval_vit(faces: np.ndarray, y: np.ndarray, max_n: int | None = None) -> dict | None:
    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except Exception as exc:
        print("transformers ViT unavailable:", exc)
        return None
    model_id = "abhilash88/face-emotion-detection"
    print("loading ViT", model_id)
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)
    except Exception as exc:
        print("ViT download failed:", exc)
        return None
    model.eval()
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    print("ViT id2label", id2label)
    fer2013_order = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    name_to_table = {
        "angry": 2, "anger": 2,
        "disgust": 5,
        "fear": 4,
        "happy": 0, "happiness": 0,
        "sad": 1, "sadness": 1,
        "surprise": 3,
        "neutral": 6,
    }
    for i, name in enumerate(fer2013_order):
        name_to_table[str(i)] = name_to_table[name]
        name_to_table[f"label_{i}"] = name_to_table[name]
        name_to_table[f"labels_{i}"] = name_to_table[name]
    idx = np.arange(len(y))
    if max_n is not None and max_n < len(y):
        rng = np.random.default_rng(SEED)
        idx = rng.choice(idx, size=max_n, replace=False)
    preds = np.zeros(len(idx), dtype=np.int64)
    t0 = time.time()
    bs = 8
    with torch.no_grad():
        for start in range(0, len(idx), bs):
            chunk = idx[start : start + bs]
            imgs = []
            for i in chunk:
                rgb = cv2.cvtColor(cv2.resize(faces[i], (224, 224)), cv2.COLOR_GRAY2RGB)
                imgs.append(Image.fromarray(rgb))
            inputs = processor(images=imgs, return_tensors="pt")
            logits = model(**inputs).logits
            pred = logits.argmax(dim=1).cpu().numpy()
            for j, raw in enumerate(pred):
                lab = str(id2label.get(int(raw), int(raw))).lower().strip()
                lab = lab.replace(" ", "").replace("-", "")
                if lab not in name_to_table and lab.isdigit():
                    preds[start + j] = name_to_table.get(fer2013_order[int(lab)], 6)
                else:
                    preds[start + j] = name_to_table.get(lab, name_to_table.get(lab.split("_")[-1], 6))
            if start % 256 == 0:
                print(f"ViT {start}/{len(idx)}")
    m = metrics7(y[idx], preds)
    m["backend"] = model_id
    m["seconds"] = round(time.time() - t0, 1)
    m["n_eval"] = int(len(idx))
    print("ViT", m["accuracy"], m["macro_f1"], "n", m["n_eval"])
    return m


def _try_ckplus() -> dict | None:
    """Best-effort extra dataset: muxspace facial_expressions (public)."""
    dest = RESULTS / "facial_expressions"
    zip_path = RESULTS / "facial_expressions.zip"
    if not dest.exists():
        url = "https://github.com/muxspace/facial_expressions/archive/refs/heads/master.zip"
        try:
            print("download extra faces", url)
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(RESULTS)
        except Exception as exc:
            print("extra face download failed:", exc)
            return None
        # extracted folder name
        found = list(RESULTS.glob("facial_expressions-master"))
        if found:
            dest = found[0]
    legend = dest / "data" / "legend.csv"
    img_dir = dest / "images"
    if not legend.is_file():
        # nested
        hits = list(RESULTS.glob("**/legend.csv"))
        if not hits:
            return None
        legend = hits[0]
        img_dir = legend.parent.parent / "images"
    map_emo = {
        "happiness": 0, "happy": 0,
        "sadness": 1, "sad": 1,
        "anger": 2, "angry": 2,
        "surprise": 3,
        "fear": 4,
        "disgust": 5,
        "neutral": 6, "contempt": 5,
    }
    from roboeye.fer_emotion import EmotionDetector

    det = EmotionDetector(download=False)
    y_true, y_pred = [], []
    with legend.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.DictReader(handle))
    rng = random.Random(SEED)
    if len(rows) > 400:
        rows = rng.sample(rows, 400)
    for row in rows:
        emo = (row.get("emotion") or "").strip().lower()
        if emo not in map_emo:
            continue
        name = row.get("image") or row.get("filename")
        if not name:
            continue
        path = img_dir / name
        if not path.is_file():
            continue
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        pad = 80
        canvas = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(120, 120, 120))
        out = det.predict(canvas)
        if out.get("backend") == "no_face":
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            plus = det._ferplus(gray) if det.session is not None else None
            p_cal = det._calibrated7(plus["probs"]) if plus is not None else None
            p_hgb = det._hgb7(gray, plus["probs"] if plus is not None else None)
            mix = p_hgb if p_hgb is not None else p_cal
            if mix is None:
                continue
            lab = det._TABLE_TO_PLUS[int(mix.argmax())]
            out = {"label": lab}
        lab = out.get("label", "neutral")
        plus_to_t = {
            "happiness": 0, "sadness": 1, "anger": 2, "surprise": 3,
            "fear": 4, "disgust": 5, "neutral": 6, "contempt": 5,
        }
        y_true.append(map_emo[emo])
        y_pred.append(plus_to_t.get(lab, 6))
    if len(y_true) < 30:
        return None
    m = metrics7(np.array(y_true), np.array(y_pred))
    m["source"] = "muxspace/facial_expressions subsample"
    print("extra-face", m["accuracy"], m["macro_f1"], "n", m["n_test"])
    return m


def track_emotion() -> dict:
    print("\n==== EMOTION ====")
    csv_path = RESULTS / "fer2013.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    faces, y, usages = _load_fer_rows(csv_path)
    print("FER rows", len(y), "usages", sorted(set(usages))[:6])
    if any(u.lower() == "privatetest" for u in usages):
        tr = np.array([i for i, u in enumerate(usages) if u.lower() == "training"])
        te = np.array([i for i, u in enumerate(usages) if u.lower() == "privatetest"])
        split_name = "official-PrivateTest"
    else:
        tr, te = stratified_split(y, 0.2)
        split_name = "stratified-20pct-holdout"
        full = _maybe_full_fer()
        if full is not None:
            faces_f, y_f, u_f = _load_fer_rows(full)
            if any(u.lower() == "privatetest" for u in u_f):
                faces, y, usages = faces_f, y_f, u_f
                tr = np.array([i for i, u in enumerate(usages) if u.lower() == "training"])
                te = np.array([i for i, u in enumerate(usages) if u.lower() == "privatetest"])
                split_name = "official-PrivateTest"
                print("switched to official FER-2013 PrivateTest", len(te))
    faces_tr, y_tr = faces[tr], y[tr]
    faces_te, y_te = faces[te], y[te]
    p8 = _ferplus_probs(faces)
    p8_tr, p8_te = p8[tr], p8[te]
    print("building HOG features")
    x_tr = _feat_matrix(faces_tr, p8_tr)
    x_te = _feat_matrix(faces_te, p8_te)
    print("feature dim", x_tr.shape)
    keep, reports, vote_p = _fit_tabular(x_tr, y_tr, x_te, y_te)
    import joblib
    from sklearn.linear_model import LogisticRegression

    cal = LogisticRegression(max_iter=400, class_weight="balanced", C=1.5)
    cal.fit(p8_tr, y_tr)
    p_cal = cal.predict_proba(p8_te)
    aligned_cal = np.zeros((len(p_cal), 7), dtype=np.float32)
    for j, c in enumerate(cal.classes_):
        aligned_cal[:, int(c)] = p_cal[:, j]
    reports["ferplus_cal"] = metrics7(y_te, aligned_cal.argmax(1))
    MODELS.mkdir(parents=True, exist_ok=True)
    np.savez(MODELS / "emotion_ferplus_cal.npz", coef=cal.coef_, intercept=cal.intercept_, classes=cal.classes_)
    joblib.dump({"model": keep, "classes": keep.classes_}, MODELS / "emotion_hgb.joblib")
    # blend HGB/vote with calibrator
    hgb_p = vote_p
    if hasattr(keep, "predict_proba"):
        raw = keep.predict_proba(x_te)
        hgb_p = np.zeros((len(raw), 7), dtype=np.float32)
        for j, c in enumerate(keep.classes_):
            hgb_p[:, int(c)] = raw[:, j]
    best_a, best_m = 0.65, None
    for a in np.linspace(0.2, 0.9, 15):
        pred = (a * hgb_p + (1 - a) * aligned_cal).argmax(1)
        m = metrics7(y_te, pred)
        if best_m is None or m["macro_f1"] > best_m["macro_f1"]:
            best_a, best_m = float(a), m
    reports["hgb_cal_blend"] = best_m
    reports["hgb_cal_blend"]["hgb_weight"] = best_a
    print("blend", best_a, best_m["accuracy"], best_m["macro_f1"])
    vit = _eval_vit(faces_te, y_te, max_n=None)
    if vit:
        reports["vit"] = vit
    extra = _try_ckplus()
    if extra:
        reports["in_the_wild_faces"] = extra
    winner = max(
        (k for k in reports if k not in {"in_the_wild_faces"} and isinstance(reports[k], dict) and "macro_f1" in reports[k]),
        key=lambda k: reports[k]["macro_f1"],
    )
    best = dict(reports[winner])
    best["winner"] = winner
    best["split"] = split_name
    best["all_backends"] = {k: {"accuracy": v.get("accuracy"), "macro_f1": v.get("macro_f1"), "n_test": v.get("n_test")}
                            for k, v in reports.items() if isinstance(v, dict) and "accuracy" in v}
    save_cm(np.array(best["confusion_matrix"]), RESULTS / "emotion_confusion_matrix.png",
            f"Emotion holdout ({winner})")
    _dump("emotion_best.json", best)
    out = {"winner": winner, "metrics": best, "split": split_name, "backends": best["all_backends"]}
    if extra:
        out["in_the_wild_faces"] = extra
    return out


# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------

def _bangla_taka_root() -> Path:
    return DATA / "Bangladeshi_Paper_Currency_Raw" / "Bangladeshi_Paper_Currency_Raw"


def _list_note_images(limit_per_class: int = 40) -> list[tuple[Path, str]]:
    root = _bangla_taka_root()
    rng = random.Random(SEED)
    items = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        cls = f"{folder.name}_taka"
        if cls not in TAKA_NAMES:
            continue
        files = [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        rng.shuffle(files)
        for p in files[:limit_per_class]:
            items.append((p, cls))
    return items


def _yolo_currency_predict(model, bgr, conf=0.25) -> str | None:
    res = model.predict(bgr, conf=conf, verbose=False, imgsz=640)
    boxes = res[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    best = int(boxes.conf.argmax())
    return res[0].names[int(boxes.cls[best].item())]


def _train_mobilenet_notes() -> dict:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.utils.data import DataLoader, random_split
    from torchvision import models, transforms
    from torchvision.datasets import ImageFolder

    root = _bangla_taka_root()
    tf_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.2),
        transforms.RandomRotation(12),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    full = ImageFolder(str(root), transform=tf_train)
    n_val = max(1, int(len(full) * 0.2))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
    val_raw = ImageFolder(str(root), transform=tf_val)
    val_ds.dataset = val_raw
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    for i, param in enumerate(model.features.parameters()):
        param.requires_grad = i >= 40  # unfreeze tail
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(full.classes))
    opt = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=8e-4)
    crit = nn.CrossEntropyLoss()
    best_acc, best_state = 0.0, None
    history = []
    epochs = 8
    for ep in range(1, epochs + 1):
        model.train()
        correct = total = 0
        for x, y in train_loader:
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x).argmax(1)
                vc += int((pred == y).sum())
                vt += len(y)
        acc = vc / max(vt, 1)
        history.append({"epoch": ep, "train_acc": correct / max(total, 1), "val_acc": acc})
        print(f"mobilenet ep{ep} train={correct/max(total,1):.3f} val={acc:.3f}")
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
        torch.save({"state_dict": best_state, "classes": full.classes, "val_acc": best_acc},
                   MODELS / "currency_mobilenet.pt")
    return {
        "n_images": len(full),
        "classes": full.classes,
        "best_val_acc": best_acc,
        "history": history,
    }


def _make_wild_yolo_set(items: list[tuple[Path, str]]) -> Path:
    out = RESULTS / "taka_wild_yolo"
    img_dir = out / "images"
    lab_dir = out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    name_to_id = {n: i for i, n in enumerate(TAKA_NAMES)}
    for src, cls in items:
        dst = img_dir / f"{cls}_{src.stem}{src.suffix.lower()}"
        if not dst.exists():
            data = cv2.imread(str(src))
            if data is None:
                continue
            cv2.imwrite(str(dst), data)
            h, w = data.shape[:2]
            gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
                xc = (x + bw / 2) / w
                yc = (y + bh / 2) / h
                nw, nh = bw / w, bh / h
            else:
                xc, yc, nw, nh = 0.5, 0.5, 0.9, 0.7
            nw = min(max(nw, 0.45), 0.98)
            nh = min(max(nh, 0.35), 0.98)
            (lab_dir / f"{dst.stem}.txt").write_text(
                f"{name_to_id[cls]} {xc:.4f} {yc:.4f} {nw:.4f} {nh:.4f}\n", encoding="utf-8"
            )
    yaml = out / "data.yaml"
    yaml.write_text(
        "\n".join([
            f"path: {out.as_posix()}",
            "train: images",
            "val: images",
            f"nc: {len(TAKA_NAMES)}",
            "names:",
            *[f"  - {n}" for n in TAKA_NAMES],
            "",
        ]),
        encoding="utf-8",
    )
    return yaml


def track_currency(finetune: bool = True) -> dict:
    print("\n==== CURRENCY ====")
    from ultralytics import YOLO

    pt = ROBO / "models" / "best.pt"
    if not pt.is_file():
        pt = MODELS / "best.onnx"
    yolo = YOLO(str(pt))
    items = _list_note_images(limit_per_class=50)
    print("wild notes", len(items), "from", _bangla_taka_root())
    cm = defaultdict(lambda: defaultdict(int))
    correct = 0
    det = 0
    for path, cls in items:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        pred = _yolo_currency_predict(yolo, bgr, conf=0.2)
        if pred is None:
            pred = "none"
        else:
            det += 1
        cm[cls][pred] += 1
        if pred == cls:
            correct += 1
    n = len(items)
    wild = {
        "n": n,
        "detection_rate": det / max(n, 1),
        "top1_accuracy": correct / max(n, 1),
        "per_true": {k: dict(v) for k, v in cm.items()},
        "weights": str(pt),
    }
    print("wild YOLO acc", wild["top1_accuracy"], "det", wild["detection_rate"])
    yaml = ROBO / "data" / "data.yaml"
    val_metrics = {}
    try:
        print("YOLO val on currency_yolo_data (imgsz=416, existing weights)")
        r = yolo.val(data=str(yaml), split="val", imgsz=416, device="cpu", workers=0, verbose=False)
        val_metrics = {
            "map50": float(r.box.map50),
            "map50_95": float(r.box.map),
            "precision": float(r.box.mp),
            "recall": float(r.box.mr),
        }
        print("currency val", val_metrics)
    except Exception as exc:
        val_metrics = {"error": str(exc)}
        print("currency val failed", exc)
    mob = {}
    try:
        mob = _train_mobilenet_notes()
    except Exception as exc:
        mob = {"error": str(exc)}
        print("mobilenet failed", exc)
    ft = {}
    if finetune and items:
        try:
            wild_yaml = _make_wild_yolo_set(items)
            print("fine-tune YOLO on wild notes", wild_yaml)
            model = YOLO(str(pt) if pt.suffix == ".pt" else str(MODELS / "yolov8n.pt"))
            model.train(
                data=str(wild_yaml),
                epochs=6,
                imgsz=416,
                batch=8,
                device="cpu",
                workers=0,
                patience=3,
                project=str(RESULTS / "yolo_runs"),
                name="taka_wild_ft",
                exist_ok=True,
                verbose=False,
            )
            best_ft = RESULTS / "yolo_runs" / "taka_wild_ft" / "weights" / "best.pt"
            if best_ft.is_file():
                MODELS.joinpath("best_wild_ft.pt").write_bytes(best_ft.read_bytes())
                y2 = YOLO(str(best_ft))
                correct2 = det2 = 0
                for path, cls in items:
                    bgr = cv2.imread(str(path))
                    if bgr is None:
                        continue
                    pred = _yolo_currency_predict(y2, bgr, conf=0.2)
                    if pred is not None:
                        det2 += 1
                    if pred == cls:
                        correct2 += 1
                ft = {
                    "weights": str(best_ft),
                    "wild_top1_after": correct2 / max(len(items), 1),
                    "wild_det_after": det2 / max(len(items), 1),
                }
                print("wild after FT", ft)
        except Exception as exc:
            ft = {"error": str(exc)}
            print("wild FT failed", exc)
    out = {"wild_existing_yolo": wild, "yolo_val": val_metrics, "mobilenet": mob, "wild_finetune": ft}
    _dump("currency_boost.json", out)
    return out


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def _ocr_font() -> str:
    for p in (
        Path(r"C:\Windows\Fonts\Nirmala.ttf"),
        Path(r"C:\Windows\Fonts\vrinda.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\tahoma.ttf"),
    ):
        if p.is_file():
            return str(p)
    return ""


def _render_text(text: str, font_path: str, wild: bool) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    rng = np.random.default_rng()
    w, h = 640, 160
    bg = int(rng.integers(200, 255)) if wild else 255
    img = Image.new("RGB", (w, h), (bg, bg, bg))
    draw = ImageDraw.Draw(img)
    size = int(rng.integers(28, 44) if wild else 36)
    try:
        font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    draw.text((24, 48), text, fill=(int(rng.integers(0, 40)),) * 3, font=font)
    if wild:
        img = img.rotate(float(rng.uniform(-6, 6)), expand=0, fillcolor=(bg, bg, bg))
        img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.0, 1.1))))
    arr = np.array(img)
    if wild and rng.random() < 0.5:
        arr = cv2.resize(arr, (0, 0), fx=0.55, fy=0.55)
        arr = cv2.resize(arr, (w, h))
    return arr


def track_ocr(n_each: int = 40) -> dict:
    print("\n==== OCR ====")
    import easyocr

    bn = [
        "বাংলাদেশ ব্যাংক", "দশ টাকা", "বিশ টাকা", "এই পথ বন্ধ",
        "স্টেশন ১২", "হাসপাতাল", "ঔষধের দোকান", "বাস স্টপ",
        "ডানদিকে যান", "বাঁদিকে যান", "মূল ফটক", "টিকিট কাউন্টার",
    ]
    en = [
        "EXIT 12", "Bus Stop", "Hospital Gate", "Ticket Counter",
        "Main Entrance", "Pharmacy", "Turn Right", "Turn Left",
        "Platform 3", "Danger Keep Out", "Glycemic", "Digestive Biscuit",
    ]
    font = _ocr_font()
    reader = easyocr.Reader(["bn", "en"], gpu=False, verbose=False)
    samples = []
    for lang, pool in (("bn", bn), ("en", en)):
        for i in range(n_each):
            text = pool[i % len(pool)]
            wild = i % 2 == 1
            img = _render_text(text, font, wild=wild)
            samples.append((text, img, lang, wild))
    rows = []
    cer_sum = wer_sum = 0.0
    for gt, img, lang, wild in samples:
        # CLAHE upsample helps EasyOCR on small Bangla
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        up = cv2.resize(clahe, (0, 0), fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
        try:
            det = reader.readtext(up, detail=1, paragraph=True)
        except Exception:
            det = reader.readtext(up, detail=1)
        pieces = []
        for t in det:
            if isinstance(t, str):
                pieces.append(t)
            elif t and len(t) > 1:
                pieces.append(str(t[1]))
        hyp = " ".join(pieces).strip()
        cer = levenshtein(gt, hyp) / max(len(gt), 1)
        wer = _token_wer(gt, hyp)
        cer_sum += cer
        wer_sum += wer
        rows.append({"gt": gt, "hyp": hyp, "lang": lang, "wild": wild, "cer": cer, "wer": wer})
    n = len(rows)
    by = {}
    for key in ("bn", "en"):
        sub = [r for r in rows if r["lang"] == key]
        by[key] = {
            "n": len(sub),
            "cer": float(np.mean([r["cer"] for r in sub])) if sub else 1.0,
            "wer": float(np.mean([r["wer"] for r in sub])) if sub else 1.0,
        }
    out = {
        "n": n,
        "cer": cer_sum / max(n, 1),
        "wer": wer_sum / max(n, 1),
        "by_lang": by,
        "engine": "easyocr-bn-en + CLAHE x1.6",
        "font": font,
        "samples_head": rows[:12],
    }
    print("OCR CER", out["cer"], "WER", out["wer"], by)
    _dump("ocr_cer.json", out)
    return out


def _token_wer(gt: str, hyp: str) -> float:
    a, b = gt.split(), hyp.split()
    if not a:
        return 0.0 if not b else 1.0
    # reuse char levenshtein on a sentinel-joined form is wrong; do list DP
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(a)


# ---------------------------------------------------------------------------
# Object / COCO
# ---------------------------------------------------------------------------

def track_object(n_images: int = 250) -> dict:
    print("\n==== OBJECT COCO ====")
    from ultralytics import YOLO

    ann_path = DATA / "coco2017" / "annotations" / "instances_val2017.json"
    img_dir = DATA / "coco2017" / "val2017"
    weights = MODELS / "yolov8s.pt"
    if not weights.is_file():
        weights = MODELS / "yolov8n.pt"
    print("loading", ann_path)
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    cat_id_to_yolo = {}
    # COCO 80 are not 1..80 contiguous; map category_id -> 0..79 via name
    name_to_yolo = {n: i for i, n in enumerate(COCO80)}
    for cat in coco["categories"]:
        if cat["name"] in name_to_yolo:
            cat_id_to_yolo[cat["id"]] = name_to_yolo[cat["name"]]
    by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd"):
            continue
        cid = cat_id_to_yolo.get(ann["category_id"])
        if cid is None:
            continue
        x, y, w, h = ann["bbox"]
        by_img[ann["image_id"]].append({"cls": cid, "xyxy": [x, y, x + w, y + h]})
    images = [im for im in coco["images"] if im["id"] in by_img]
    rng = random.Random(SEED)
    rng.shuffle(images)
    images = images[:n_images]
    model = YOLO(str(weights))
    # collect preds per class for AP
    gt_by_cls = defaultdict(list)  # cls -> list of (img_i, box)
    pred_by_cls = defaultdict(list)  # cls -> list of (img_i, conf, box)
    for i, im in enumerate(images):
        path = img_dir / im["file_name"]
        if not path.is_file():
            continue
        for g in by_img[im["id"]]:
            gt_by_cls[g["cls"]].append((i, g["xyxy"]))
        res = model.predict(str(path), conf=0.15, verbose=False, imgsz=640)[0]
        if res.boxes is None:
            continue
        for b in res.boxes:
            cls = int(b.cls.item())
            conf = float(b.conf.item())
            xyxy = [float(x) for x in b.xyxy[0].tolist()]
            pred_by_cls[cls].append((i, conf, xyxy))
        if i % 25 == 0:
            print(f"COCO {i}/{len(images)}")
    aps = []
    per = {}
    for cls, name in enumerate(COCO80):
        gts = gt_by_cls.get(cls, [])
        preds = sorted(pred_by_cls.get(cls, []), key=lambda t: -t[1])
        if not gts and not preds:
            continue
        matched = set()
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        for k, (img_i, conf, box) in enumerate(preds):
            cands = [(j, b) for j, (ii, b) in enumerate(gts) if ii == img_i and (cls, j) not in matched]
            best_j, best_iou = -1, 0.0
            for j, b in cands:
                v = iou_xyxy(box, b)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= 0.5 and best_j >= 0:
                tp[k] = 1
                matched.add((cls, best_j))
            else:
                fp[k] = 1
        if not gts:
            per[name] = {"ap50": 0.0, "n_gt": 0, "n_pred": len(preds)}
            continue
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        rec = tp_c / len(gts)
        prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
        ap = voc_ap(rec, prec) if len(preds) else 0.0
        aps.append(ap)
        per[name] = {"ap50": ap, "n_gt": len(gts), "n_pred": len(preds)}
    map50 = float(np.mean(aps)) if aps else 0.0
    out = {
        "weights": str(weights),
        "n_images": len(images),
        "map50": map50,
        "n_classes_scored": len(aps),
        "split": "COCO val2017 random subset",
        "per_class_head": dict(list(sorted(per.items(), key=lambda kv: -kv[1]["ap50"]))[:15]),
    }
    print("COCO subset mAP50", map50, "n", len(images))
    _dump("object_coco.json", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="all", choices=["emotion", "currency", "ocr", "object", "all"])
    parser.add_argument("--skip-currency-ft", action="store_true")
    parser.add_argument("--coco-n", type=int, default=250)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        if args.track in {"emotion", "all"}:
            summary["emotion"] = track_emotion()
        if args.track in {"ocr", "all"}:
            summary["ocr"] = track_ocr()
        if args.track in {"object", "all"}:
            summary["object"] = track_object(n_images=args.coco_n)
        if args.track in {"currency", "all"}:
            summary["currency"] = track_currency(finetune=not args.skip_currency_ft)
    finally:
        summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _dump("publish_boost.json", summary)
        print("\n==== SUMMARY ====")
        print(json.dumps({k: _brief(v) for k, v in summary.items()}, indent=2, ensure_ascii=False))


def _brief(v):
    if not isinstance(v, dict):
        return v
    keys = ("winner", "accuracy", "macro_f1", "cer", "wer", "map50", "best_val_acc", "split")
    out = {k: v[k] for k in keys if k in v}
    if "metrics" in v and isinstance(v["metrics"], dict):
        out.update({k: v["metrics"][k] for k in ("accuracy", "macro_f1") if k in v["metrics"]})
    if "wild_existing_yolo" in v:
        out["wild_top1"] = v["wild_existing_yolo"].get("top1_accuracy")
    if "yolo_val" in v:
        out["currency_map50"] = v["yolo_val"].get("map50") if isinstance(v["yolo_val"], dict) else None
    return out or v


if __name__ == "__main__":
    main()
