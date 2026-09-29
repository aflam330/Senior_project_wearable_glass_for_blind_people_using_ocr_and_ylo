"""End-to-end check of the glass's safe counterfeit policy, through the app's own code.

Runs savior_glass CurrencyMode.detect_live (config.JAAL_SAFE_POLICY_ENABLED = True) on whole-note
photos and records the spoken sentence. Checks that "জাল" is never spoken and counts
"সম্ভবত আসল" (likely genuine) per set. No image here trained or tuned any model or threshold.

Sets: counterfeit set (all images), Bangla Money 500/1000 (all), BanglaTaka raw 500/1000
(300 sampled, seed 42).
Output: results/jaal_whole/app_check.json
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "savior_glass"))
sys.path.insert(0, str(ROOT / "realtime_bangla_taka_detection"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from modes.currency_mode import CurrencyMode  # noqa: E402
from jaal_whole_note import BM, BT, CF, IMG_EXT, wilson  # noqa: E402

OUT = ROOT / "realtime_bangla_taka_detection" / "results" / "jaal_whole" / "app_check.json"


def items() -> list[tuple[str, str, Path, bool]]:
    out = []
    for folder in sorted(p for p in CF.iterdir() if p.is_dir()):
        truth = "counterfeit" if "Counterfeit" in folder.name else "genuine"
        out += [("cf", truth, p, p.name.startswith("augmented")) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG_EXT]
    for d in ("500", "1000"):
        out += [("bm", "genuine", p, False) for p in sorted((BM / d).iterdir()) if p.suffix.lower() in IMG_EXT]
    bt = [p for d in ("500", "1000") for p in sorted((BT / d).iterdir()) if p.suffix.lower() in IMG_EXT]
    random.Random(42).shuffle(bt)
    out += [("bt", "genuine", p, False) for p in bt[:300]]
    return out


def main() -> None:
    assert config.JAAL_SAFE_POLICY_ENABLED and not config.JAAL_VERDICT_ENABLED
    mode = CurrencyMode()
    mode._load_yolo()
    mode._load_auth()
    assert mode.safe_policy and mode._auth_kind == "qduig"
    rows = []
    for i, (s, truth, p, aug) in enumerate(items()):
        hits = mode.detect_live(cv2.imread(str(p)))
        h = hits[0] if hits else {}
        rows.append({"set": s, "truth": truth, "augmented": aug, "path": str(p.relative_to(ROOT)),
                     "yolo": h.get("name"), "auth": h.get("auth"), "p": h.get("genuine_prob"), "text": h.get("text")})
        if (i + 1) % 200 == 0:
            print(i + 1, flush=True)
    said_jaal = [r for r in rows if r["text"] and "জাল" in r["text"]]
    summ = {"spoken_jaal": len(said_jaal), "tau": config.JAAL_SAFE_TAU, "sets": {}}
    for s, truth in (("cf", "counterfeit"), ("cf", "genuine"), ("bm", "genuine"), ("bt", "genuine")):
        rs = [r for r in rows if r["set"] == s and r["truth"] == truth and not r["augmented"]]
        c = Counter(r["auth"] for r in rs)
        n_checked = c["likely_genuine"] + c["check_by_hand"]
        summ["sets"][f"{s}/{truth}"] = {"n_images": len(rs), "outcomes": dict(c), "checked": n_checked,
                                        "likely_genuine": c["likely_genuine"],
                                        "likely_genuine_wilson95": wilson(c["likely_genuine"], n_checked)}
    aug = [r for r in rows if r["augmented"]]
    summ["cf_counterfeit_augmented_copies"] = dict(Counter(r["auth"] for r in aug))
    OUT.write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
