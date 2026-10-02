"""Task 7 decision on val: the final pipeline with each fine-tuned recognizer (seeds 42-44), end to end, 3 image seeds."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_bench import bench as B, final_pipeline as F  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "ocr_bench"


def main() -> None:
    c = F.load_choices()
    res = {}
    for seed in (42, 43, 44):
        F.use_recognizer(Path(__file__).resolve().parent / "finetuned" / f"recognizer_seed{seed}.pth")
        r = B.score(F.make_reader(c), "val")
        B.save(r, OUT / f"val_t7_ft_seed{seed}.json", {"set": "val", "task": 7, "seed": seed, "choices": c})
        res[seed] = r["overall"]["cer"]
        print(B.line(f"val final_ft_seed{seed}", r), flush=True)
    (OUT / "task7_val.json").write_text(json.dumps({"val_cer_by_seed": res}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
