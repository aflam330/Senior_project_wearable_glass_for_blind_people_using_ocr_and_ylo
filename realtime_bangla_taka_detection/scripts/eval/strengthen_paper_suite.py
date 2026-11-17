"""Paper-strengthening suite (no real counterfeit-UV required).

Produces honest stress metrics for BD authenticity, gated coverage,
calibration (ECE), per-denom breakdown, offline glass-style latency,
and a fibre-free UV simulation summary pointer.

Usage:
  python scripts/eval/strengthen_paper_suite.py
  python scripts/eval/strengthen_paper_suite.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.bd_counterfeit_auth import BDCounterfeitAuth, IMG, _TF
from roboeye.config import BD_COUNTERFEIT_AUTH_WEIGHTS, DATA_SET, DEVICE, MODELS_DIR

MANIFEST = DATA_SET / "BD_Auth_Combined" / "manifest.csv"
OUT_DIR = ROOT / "results" / "strengthen"
OUT_JSON = OUT_DIR / "paper_strengthen_suite.json"
OUT_MD = ROOT.parent / "paper_evidence" / "STRENGTHEN_2026-11-18.md"
UV_STRESS = (
    ROOT.parent / "savior_glass" / "results" / "uv_check" / "uv_stress_sim.json"
)


def load_rows(split: str = "test") -> list[dict]:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    return [r for r in rows if r["split"] == split]


def corrupt(img: Image.Image, kind: str, rng: np.random.Generator) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    if kind == "clean":
        return img
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(1.2, 2.8))))
    if kind == "jpeg":
        q = int(rng.integers(25, 55))
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return img
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return Image.fromarray(cv2.cvtColor(dec, cv2.COLOR_BGR2RGB))
    if kind == "dark":
        return ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.35, 0.55)))
    if kind == "bright":
        return ImageEnhance.Brightness(img).enhance(float(rng.uniform(1.4, 1.8)))
    if kind == "noise":
        noise = rng.normal(0, float(rng.uniform(12, 28)), arr.shape).astype(np.float32)
        out = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(out)
    if kind == "crop":
        w, h = img.size
        s = float(rng.uniform(0.72, 0.88))
        nw, nh = int(w * s), int(h * s)
        x0 = int(rng.integers(0, max(w - nw, 1)))
        y0 = int(rng.integers(0, max(h - nh, 1)))
        return img.crop((x0, y0, x0 + nw, y0 + nh)).resize((w, h), Image.Resampling.BILINEAR)
    if kind == "rotate":
        ang = float(rng.choice([-35, -25, 25, 35]))
        return img.rotate(ang, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=(0, 0, 0))
    return img


@torch.inference_mode()
def score_batch(model, pil_list: list[Image.Image], device) -> tuple[np.ndarray, np.ndarray]:
    xs = torch.stack([_TF(im) for im in pil_list]).to(device)
    prob = F.softmax(model(xs), dim=1).cpu().numpy()
    pred = prob.argmax(1)
    conf = prob.max(1)
    return pred, conf


def ece_score(confs: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    if len(confs) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (confs >= bins[i]) & (confs < bins[i + 1] if i < n_bins - 1 else confs <= bins[i + 1])
        if not m.any():
            continue
        ece += (m.mean()) * abs(correct[m].mean() - confs[m].mean())
    return float(ece)


def gated(ys, ps, confs, thr: float) -> dict:
    decided = confs >= thr
    n_dec = int(decided.sum())
    n_abs = int((~decided).sum())
    acc = float((ys[decided] == ps[decided]).mean()) if n_dec else 0.0
    return {
        "min_conf": thr,
        "coverage": float(n_dec / max(len(ys), 1)),
        "accuracy_when_decided": acc,
        "abstain": n_abs,
        "decided": n_dec,
    }


def eval_corruption(model, rows: list[dict], kind: str, device, limit: int | None) -> dict:
    rng = np.random.default_rng(42 + abs(hash(kind)) % 10_000)
    use = rows if limit is None else rows[:limit]
    ys, ps, confs, notes, denoms = [], [], [], [], []
    batch_imgs, batch_meta = [], []
    bs = 32

    def flush():
        nonlocal batch_imgs, batch_meta
        if not batch_imgs:
            return
        pred, conf = score_batch(model, batch_imgs, device)
        for (y, note, denom), p, c in zip(batch_meta, pred, conf):
            ys.append(int(y))
            ps.append(int(p))
            confs.append(float(c))
            notes.append(note)
            denoms.append(denom)
        batch_imgs, batch_meta = [], []

    for r in use:
        try:
            im = Image.open(r["path"]).convert("RGB")
        except Exception:
            continue
        im = corrupt(im, kind, rng)
        batch_imgs.append(im)
        batch_meta.append((r["label"], r["note_id"], r["denom"]))
        if len(batch_imgs) >= bs:
            flush()
    flush()

    ys_a = np.array(ys)
    ps_a = np.array(ps)
    cf_a = np.array(confs)
    ok = ys_a == ps_a
    # note majority
    by = defaultdict(list)
    for n, y, p in zip(notes, ys_a, ps_a):
        by[n].append((int(y), int(p)))
    n_ok = sum(1 for pairs in by.values() if Counter(p for _, p in pairs).most_common(1)[0][0] == pairs[0][0])
    per_denom = {}
    for d in sorted(set(denoms)):
        m = np.array([x == d for x in denoms])
        per_denom[d] = {
            "n": int(m.sum()),
            "acc": float((ys_a[m] == ps_a[m]).mean()) if m.any() else 0.0,
        }
    fake = ys_a == 0
    real = ys_a == 1
    return {
        "corruption": kind,
        "n_images": int(len(ys_a)),
        "n_notes": int(len(by)),
        "image_acc": float(ok.mean()) if len(ok) else 0.0,
        "note_acc": float(n_ok / max(len(by), 1)),
        "fake_recall": float((ps_a[fake] == 0).mean()) if fake.any() else 0.0,
        "real_recall": float((ps_a[real] == 1).mean()) if real.any() else 0.0,
        "mean_conf": float(cf_a.mean()) if len(cf_a) else 0.0,
        "ece": ece_score(cf_a, ok.astype(np.float64)),
        "gated": {str(t): gated(ys_a, ps_a, cf_a, t) for t in (0.70, 0.90, 0.95)},
        "per_denom": per_denom,
    }


def latency_bench(clf: BDCounterfeitAuth, rows: list[dict], n: int = 64) -> dict:
    use = rows[:n]
    # warmup
    for r in use[:4]:
        clf.predict_path(r["path"])
    t0 = time.perf_counter()
    for r in use:
        clf.predict_path(r["path"])
    dt = time.perf_counter() - t0
    return {
        "n": len(use),
        "total_s": round(dt, 3),
        "ms_per_image": round(1000.0 * dt / max(len(use), 1), 2),
        "device": str(DEVICE),
    }


def load_uv_stress() -> dict | None:
    if UV_STRESS.is_file():
        return json.loads(UV_STRESS.read_text(encoding="utf-8"))
    return None


def write_markdown(payload: dict) -> None:
    stress = payload["bd_auth_stress"]
    lines = [
        "# Paper strengthen suite (2026-11-18)",
        "",
        "Code-only strengthening. **No real counterfeit UV photographs** (known gap).",
        "",
        "## BD authenticity under corruptions",
        "",
        "| Corruption | Image acc | Note acc | Fake recall | Real recall | ECE | Acc@0.90 (coverage) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, m in stress.items():
        g = m["gated"]["0.9"]
        lines.append(
            f"| {kind} | {m['image_acc']:.3f} | {m['note_acc']:.3f} | "
            f"{m['fake_recall']:.3f} | {m['real_recall']:.3f} | {m['ece']:.3f} | "
            f"{g['accuracy_when_decided']:.3f} ({g['coverage']:.2f}) |"
        )
    lat = payload["glass_style_latency"]
    lines += [
        "",
        "## Offline glass-style BD auth latency",
        "",
        f"- Device: `{lat['device']}`",
        f"- {lat['ms_per_image']} ms / image ({lat['n']} images)",
        "",
        "## UV fibre-free simulation",
        "",
    ]
    uv = payload.get("uv_stress_sim")
    if uv:
        lines.append(f"- Loaded existing simulation: `{UV_STRESS}`")
        if isinstance(uv, dict):
            # keep short; keys vary by older runs
            lines.append(f"- Summary keys: {', '.join(list(uv.keys())[:12])}")
    else:
        lines.append("- Not found yet; run `savior_glass/scripts/uv_stress_sim.py`.")
    lines += [
        "",
        "## Framing for papers",
        "",
        "1. Report **clean + stressed** BD auth numbers (do not claim 100% field accuracy).",
        "2. Prefer **gated** accuracy at min_conf=0.90 with coverage.",
        "3. UV: genuine fibres measured; counterfeit UV unavailable — fibre-free simulation only.",
        "4. Glass speaks likely-genuine / check-by-hand only.",
        "",
        f"Machine JSON: `{OUT_JSON}`",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Fewer images / corruptions")
    ap.add_argument("--weights", type=Path, default=BD_COUNTERFEIT_AUTH_WEIGHTS)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows("test")
    if not rows:
        raise SystemExit(f"No test rows in {MANIFEST}")

    clf = BDCounterfeitAuth(weights=args.weights, min_conf=0.90)
    if not clf.loaded:
        raise SystemExit(f"Missing weights: {args.weights}")
    model = clf.model

    kinds = ["clean", "blur", "jpeg", "dark", "noise", "crop", "rotate"]
    if args.quick:
        kinds = ["clean", "blur", "jpeg", "dark", "noise"]
    limit = 120 if args.quick else None

    stress = {}
    for kind in kinds:
        print(f"stress: {kind} ...", flush=True)
        stress[kind] = eval_corruption(model, rows, kind, DEVICE, limit)
        print(
            f"  img={stress[kind]['image_acc']:.3f} note={stress[kind]['note_acc']:.3f} "
            f"ece={stress[kind]['ece']:.3f}",
            flush=True,
        )

    print("latency ...", flush=True)
    latency = latency_bench(clf, rows, n=48 if args.quick else 96)

    # External real-note false-fake check (PaperCurrencyRaw samples if present)
    external = {"n": 0, "called_fake": 0, "false_fake_rate": None}
    raw = DATA_SET / "Bangladeshi_Paper_Currency_Raw" / "Bangladeshi_Paper_Currency_Raw"
    if raw.is_dir():
        paths = []
        for d in ("200", "500", "1000"):
            folder = raw / d
            if folder.is_dir():
                paths.extend(sorted(folder.glob("*.jpg"))[:40])
                paths.extend(sorted(folder.glob("*.png"))[:40])
        if args.quick:
            paths = paths[:60]
        bad = 0
        for p in paths:
            r = clf.predict_path(p)
            if r["verdict"] == "COUNTERFEIT":
                bad += 1
        external = {
            "n": len(paths),
            "called_fake": bad,
            "false_fake_rate": (bad / len(paths)) if paths else None,
            "source": str(raw),
        }
        print(f"external real false-fake: {bad}/{len(paths)}", flush=True)

    payload = {
        "weights": str(args.weights),
        "device": str(DEVICE),
        "test_n": len(rows),
        "bd_auth_stress": stress,
        "glass_style_latency": latency,
        "external_real_false_fake": external,
        "uv_stress_sim": load_uv_stress(),
        "caveats": [
            "No real counterfeit UV photographs (project gap).",
            "Stress corruptions are synthetic; field cameras may differ.",
            "Clean-split near-perfect scores may reflect source-domain ease.",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
