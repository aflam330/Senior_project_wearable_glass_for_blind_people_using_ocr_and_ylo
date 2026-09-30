"""Self-test for print_capture.py: every refusal rule, on synthetic images in a temp folder.

Run: python scripts/dataset/test_print_capture.py   (or pytest on this file)
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from print_capture import Registry, RuleError, main  # noqa: E402


def _img(path: Path, seed: int, dark: bool = False) -> Path:
    rng = np.random.default_rng(seed)
    im = rng.integers(0, 40 if dark else 255, (240, 480, 3), dtype=np.uint8)
    cv2.imwrite(str(path), im)
    return path


def _meta(note, prt, label="genuine", currency="BDT", denom="500", lighting="backlight"):
    return {"note_id": note, "print_id": prt, "currency": currency, "denomination": denom, "label": label,
            "camera": "glass", "lighting": lighting}


def _refused(fn) -> bool:
    try:
        fn()
    except RuleError:
        return True
    return False


def test_rules() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        prints = [{"print_id": f"P{i:03d}", "currency": c, "denomination": "500", "label": lab}
                  for i, (c, lab) in enumerate([(c, lab) for c in ("BDT", "INR") for lab in LABELS_X for _ in range(10)])]
        with open(d / "prints.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(prints[0]))
            w.writeheader()
            w.writerows(prints)
        root = d / "ds"
        assert main(["plan", "--root", str(root), "--prints", str(d / "prints.csv")]) == 0
        reg = Registry(root)
        split = {p: reg.split_of(p) for p in reg.splits["prints"]}
        assert set(split.values()) == {"train", "val", "test"}
        # every stratum reaches test
        for c in ("BDT", "INR"):
            for lab in LABELS_X:
                assert any(split[p["print_id"]] == "test" for p in prints if p["currency"] == c and p["label"] == lab)
        before = dict(split)
        Registry(root).plan(prints + [{"print_id": "NEW", "currency": "BDT", "denomination": "500", "label": "genuine"}], seed=7)
        after = Registry(root)
        assert all(after.split_of(p) == s for p, s in before.items()), "replanning moved an existing print"
        assert after.split_of("NEW") in ("train", "val", "test")

        reg = Registry(root)
        r = reg.add(_img(d / "a.jpg", 1), _meta("N1", "P000"))
        assert r["split"] == split["P000"] and (root / r["file"]).exists()
        reg.add(_img(d / "b.jpg", 2), _meta("N1", "P000", lighting="room"))
        assert _refused(lambda: reg.add(d / "a.jpg", _meta("N9", "P000"))), "duplicate file accepted"
        assert _refused(lambda: reg.add(_img(d / "c.jpg", 3), _meta("N1", "P001"))), "note moved to a second print"
        assert _refused(lambda: reg.add(_img(d / "e.jpg", 5), _meta("N2", "UNPLANNED"))), "print without a split accepted"
        assert _refused(lambda: reg.add(_img(d / "f.jpg", 6), _meta("N3", "P000", label="counterfeit"))), "label contradicts plan"
        assert _refused(lambda: reg.add(_img(d / "g.jpg", 7), _meta("N4", "P000", currency="INR"))), "currency contradicts plan"
        assert _refused(lambda: reg.add(_img(d / "i.jpg", 9), _meta("bad:id", "P001"))), "unsafe id accepted"
        q = reg.add(_img(d / "h.jpg", 8, dark=True), _meta("N5", "P001"))
        assert "dark" in q["quality"]["flags"]
        errors, counts = Registry(root).check()
        assert not errors, errors
        # tamper: move a stored row to another split -> check must fail
        m = root / "manifest.jsonl"
        lines = m.read_text(encoding="utf-8").splitlines()
        other = next(s for s in ("train", "val", "test") if s != split["P000"])
        lines.append(lines[0].replace(f'"split": "{split["P000"]}"', f'"split": "{other}"').replace('"sha256": "', '"sha256": "x'))
        m.write_text("\n".join(lines) + "\n", encoding="utf-8")
        errors, _ = Registry(root).check()
        assert any("in splits" in e for e in errors), errors
        assert main(["check", "--root", str(root)]) == 1


LABELS_X = ("genuine", "counterfeit")

if __name__ == "__main__":
    test_rules()
    print("print_capture: all rules hold")
