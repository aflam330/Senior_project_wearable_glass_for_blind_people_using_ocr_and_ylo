"""Does PRMVT rely on the serial number? Test-time masking on JaalTaka TEST notes (no retraining).

For each test note, views 1-4 are registered to the whole-note template (as in synth_whole_notes.py).
The two serial-number boxes of the template (lower-left and upper-right, fractions of the note) are
mapped back into each view with the inverse homography and filled with the view's median colour.
PRMVT (prefix_ft seed 42) is scored on the first k = 1..4 views, masked and unmasked, and the
counterfeit catch rate is split by whether the note's serial was seen among TRAIN counterfeits
(results/jaal_whole/serial_split.json).
Output: results/jaal_whole/serial_mask_test.json
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
import synth_whole_notes as sw  # noqa: E402

OUT = ROOT / "results" / "jaal_whole"
# serial boxes on the template, (x0, y0, x1, y1) as fractions of the note
SERIAL_BOXES = [(0.06, 0.78, 0.44, 0.97), (0.60, 0.12, 0.92, 0.34)]


def masked_views(item):
    nid, paths, label, split = item
    views = []
    for p in paths[:4]:
        img = cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_2)
        s = 700 / img.shape[1]
        col = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        views.append((col, cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)))
    regs = {d: [sw._reg(g, d) for _, g in views] for d in sw.TEMPLATES}
    denom = max(regs, key=lambda d: sum(r[0] for r in regs[d]))
    th, tw = sw._T[denom][1]
    out, n_masked = [], 0
    for (col, _), (inl, H) in zip(views, regs[denom]):
        m = col.copy()
        if H is not None and inl >= sw.MIN_INLIERS:
            Hi = np.linalg.inv(H)
            fill = np.median(col.reshape(-1, 3), 0).astype(np.uint8).tolist()
            for x0, y0, x1, y1 in SERIAL_BOXES:
                box = np.float32([[x0 * tw, y0 * th], [x1 * tw, y0 * th], [x1 * tw, y1 * th], [x0 * tw, y1 * th]]).reshape(-1, 1, 2)
                poly = cv2.perspectiveTransform(box, Hi).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(m, [poly], fill)
            n_masked += 1
        out.append((cv2.imencode(".png", col)[1].tobytes(), cv2.imencode(".png", m)[1].tobytes()))
    return nid, label, n_masked, out


def main() -> None:
    from roboeye.authenticity import default_transform
    from roboeye.camva.notes import load_splits
    from roboeye.qduig.engine import load_qduig
    splits, records = load_splits()
    items = [(n, records[n]["view_paths"], int(records[n]["label"]), "test") for n in splits["test"]]
    with ProcessPoolExecutor(max_workers=4, initializer=sw._init) as ex:
        regs = list(ex.map(masked_views, items, chunksize=4))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_qduig(ROOT / "results/qduig/prefix_ft/seed42/checkpoint.pt").to(dev).eval()
    tf = default_transform(train=False)
    seen = set(json.loads((OUT / "serial_split.json").read_text(encoding="utf-8"))["unseen_note_ids"])
    dec = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    res = {"n_notes": len(regs), "views_masked_of_4": [r[2] for r in regs].count(4), "by_k": {}}
    for k in range(1, 5):
        rows = []
        for nid, label, _, vs in regs:
            ps = []
            for which in (0, 1):
                x = torch.stack([tf(Image.fromarray(cv2.cvtColor(dec(v[which]), cv2.COLOR_BGR2RGB))) for v in vs[:k]]).unsqueeze(0).to(dev)
                with torch.inference_mode():
                    ps.append(float(net(x, torch.ones(1, k, dtype=torch.long, device=dev))["prob"].reshape(-1)[0]))
            rows.append((nid, label, ps[0], ps[1]))
        y = np.array([r[1] for r in rows])
        entry = {}
        for name, col in (("unmasked", 2), ("serial_masked", 3)):
            p = np.array([r[col] for r in rows])
            cf_unseen = [r for r in rows if r[1] == 0 and r[0] in seen]
            cf_seen = [r for r in rows if r[1] == 0 and r[0] not in seen]
            entry[name] = {"accuracy": float(((p >= 0.5) == (y == 1)).mean()),
                           "genuine_correct": int(((p >= 0.5) & (y == 1)).sum()), "genuine_n": int((y == 1).sum()),
                           "cf_caught_serial_seen": sum(r[col] < 0.5 for r in cf_seen), "cf_serial_seen_n": len(cf_seen),
                           "cf_caught_serial_unseen": sum(r[col] < 0.5 for r in cf_unseen), "cf_serial_unseen_n": len(cf_unseen)}
        res["by_k"][str(k)] = entry
        print(k, entry, flush=True)
    (OUT / "serial_mask_test.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
