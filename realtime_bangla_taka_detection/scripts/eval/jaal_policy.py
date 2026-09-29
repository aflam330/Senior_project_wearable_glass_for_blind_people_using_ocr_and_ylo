"""Policy E for the glass: "likely genuine" or "check by hand", never "counterfeit".

Rules fixed in results/jaal_whole/PREREGISTERED_OPERATING_POINTS.md before whole-note scores were read:
  primary tau   = highest p(genuine) of any JaalTaka VAL counterfeit note
  secondary tau = 99th percentile of JaalTaka VAL counterfeit scores
  checker       = lowest share of VAL genuine notes sent to "check by hand" at the primary tau
Then the whole-note scores in results/jaal_whole/scores.json are read once.

Output: results/jaal_whole/policy.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from roboeye.authenticity import default_transform  # noqa: E402
from roboeye.camva.notes import load_splits  # noqa: E402
from roboeye.qduig.engine import load_qduig  # noqa: E402
import eval_backbone_probes as probes  # noqa: E402
from jaal_whole_note import DEV, PRMVT, prmvt_prob, resnet_feats, wilson  # noqa: E402

OUT = ROOT / "results" / "jaal_whole"
CHECKERS = [f"S{k}" for k in range(1, 5)] + [f"R{k}" for k in range(1, 5)]


def val_scores() -> dict:
    cache = OUT / "val_scores.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    splits, records = load_splits()
    net = load_qduig(PRMVT).to(DEV).eval()
    tf = default_transform(train=False)
    rmodel, rtf = probes.backbone("resnet50")
    fc = np.load(ROOT / "results" / "sota" / "feats_resnet50.npz", allow_pickle=True)
    pos = {n: i for i, n in enumerate(fc["ids"])}
    xtr = fc["x"][[pos[n] for n in splits["train"]]].reshape(-1, fc["x"].shape[2])
    ytr = np.repeat([int(records[n]["label"]) for n in splits["train"]], 6)
    lr = LogisticRegression(C=100.0, max_iter=3000).fit(xtr, ytr)
    out = {"label": [], **{c: [] for c in CHECKERS}}
    for nid in splits["val"]:
        views = [cv2.imread(p) for p in records[nid]["view_paths"][:4]]
        pr = lr.predict_proba(resnet_feats(rmodel, rtf, views))[:, 1]
        out["label"].append(int(records[nid]["label"]))
        for k in range(1, 5):
            out[f"S{k}"].append(prmvt_prob(net, tf, views[:k]))
            out[f"R{k}"].append(float(pr[:k].mean()))
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def split_scores(split: str) -> dict:
    """PRMVT checkers S1-S4 on real JaalTaka views of one split (cached)."""
    cache = OUT / f"{split}_scores.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    splits, records = load_splits()
    net = load_qduig(PRMVT).to(DEV).eval()
    tf = default_transform(train=False)
    out = {"label": [], **{f"S{k}": [] for k in range(1, 5)}}
    for nid in splits[split]:
        views = [cv2.imread(p) for p in records[nid]["view_paths"][:4]]
        out["label"].append(int(records[nid]["label"]))
        for k in range(1, 5):
            out[f"S{k}"].append(prmvt_prob(net, tf, views[:k]))
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def main() -> None:
    v = val_scores()
    y = np.asarray(v["label"])
    ops = {}
    for c in CHECKERS:
        s = np.asarray(v[c])
        cf, gen = s[y == 0], s[y == 1]
        for name, tau in (("primary", float(cf.max())), ("secondary", float(np.quantile(cf, 0.99)))):
            # "likely genuine" means p > tau (strict), so the VAL counterfeit that sets the primary tau is not passed
            ops[(c, name)] = {"tau": tau, "val_counterfeit_passed": int((cf > tau).sum()), "val_counterfeit_n": int(len(cf)),
                              "val_genuine_passed": int((gen > tau).sum()), "val_genuine_n": int(len(gen))}
    chosen = min(CHECKERS, key=lambda c: (-ops[(c, "primary")]["val_genuine_passed"], int(c[1:])))

    rows = json.loads((OUT / "scores.json").read_text(encoding="utf-8"))
    test = {}
    for (c, name), op in ops.items():
        res = {}
        for setname, truth in (("cf", "counterfeit"), ("cf", "genuine"), ("bm", "genuine"), ("bt", "genuine")):
            det = [r for r in rows if r["set"] == setname and r["truth"] == truth and r.get("yolo") and not r["augmented"]]
            passed = [r for r in det if r[c] > op["tau"]]
            e = {"n": len(det), "said_likely_genuine": len(passed), "wilson95": wilson(len(passed), len(det))}
            if truth == "counterfeit":
                g = defaultdict(lambda: [0, 0])
                for r in det:
                    g[r["group"]][0] += 1
                    g[r["group"]][1] += int(r[c] > op["tau"])
                e["groups_n_images_n_passed"] = dict(g)
                e["groups_with_any_pass"] = sum(1 for a in g.values() if a[1] > 0)
                e["groups"] = len(g)
            else:
                e["groups"] = len({r["group"] for r in det})
            res[f"{setname}/{truth}"] = e
        test[f"{c}/{name}"] = {**op, "whole_note": res}
    # In-domain safety check: the chosen checker and tau on real JaalTaka TEST views (read once)
    t = split_scores("test")
    ty, ts = np.asarray(t["label"]), np.asarray(t[chosen])
    tau_c = ops[(chosen, "primary")]["tau"]
    jt = {"checker": chosen, "tau": tau_c,
          "counterfeit_passed": int((ts[ty == 0] > tau_c).sum()), "counterfeit_n": int((ty == 0).sum()),
          "genuine_passed": int((ts[ty == 1] > tau_c).sum()), "genuine_n": int((ty == 1).sum())}
    jt["counterfeit_passed_wilson95"] = wilson(jt["counterfeit_passed"], jt["counterfeit_n"])
    print("JaalTaka TEST (real views):", jt)
    blob = {"rules": "results/jaal_whole/PREREGISTERED_OPERATING_POINTS.md", "chosen_checker_on_val": chosen,
            "jaaltaka_test_real_views": jt, "operating_points": test}
    (OUT / "policy.json").write_text(json.dumps(blob, indent=1), encoding="utf-8")
    print("chosen on VAL:", chosen)
    for key, t in test.items():
        w = t["whole_note"]
        print(f"{key:14s} tau={t['tau']:.4f} valCF {t['val_counterfeit_passed']}/{t['val_counterfeit_n']} valGEN {t['val_genuine_passed']}/{t['val_genuine_n']} | "
              + " ".join(f"{k}:{e['said_likely_genuine']}/{e['n']}" for k, e in w.items()))


if __name__ == "__main__":
    main()
