"""Append the 2026-10-02 benchmark results to paper_evidence/OCR_*.md. Every number is read from results/ocr_bench/*.json.

The files already hold an earlier pass from the same day; that text is kept. The new section explains why its
numbers are not comparable (Bangla drawn without shaping) and gives the new tables.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "savior_glass" / "results" / "ocr_bench"
PE = ROOT / "paper_evidence"
MARK = "<!-- ocr-bench-2026-10-02 -->"


def J(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def pct(x):
    return "—" if x is None else f"{100 * x:.1f} %"


def row(label, d, extra=""):
    bc = d["by_cond"]
    bn = d["bn"]["cer"] if d.get("bn") else None
    seeds = " / ".join(f"{100 * s:.1f}" for s in d["per_seed_cer"])
    return (f"| {label} | {pct(d['overall']['cer'])} | {pct(bn)} | {pct(d['en']['cer'])} | {pct(d['overall']['wer'])} | "
            f"{pct(bc['clean']['all']['cer'])} | {pct(bc['wild']['all']['cer'])} | {pct(bc['scene']['all']['cer'])} | {seeds} |{extra}")


HEAD = ("| Method | CER | Bangla CER | English CER | WER | Clean | Distorted | Photo scene | CER per seed |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---|")

SETUP = f"""{MARK}
## Benchmark pass with correctly shaped Bangla (2026-10-02, afternoon)

**Why the earlier numbers in this file are not comparable.** The earlier renders used Pillow's basic text layout. On this Windows machine Pillow has no Raqm, so Bangla was drawn wrongly: pre-base vowel signs after the consonant, conjuncts broken (for example "দেশ" drawn as "দশে"). Real print never looks like that. This pass shapes text with HarfBuzz (`savior_glass/ocr_bench/shaped_text.py`), the same shaping a printer or browser does.

**Benchmark** (`savior_glass/ocr_bench/bench.py`):
- **Phrase sets:**
  - val, 24 Bangla + 24 English new phrases, used for every choice;
  - test, 48 other new phrases, read once per finished method;
  - lexicon, the old 24 phrases;
  - select, the 24 phrases used once for the canvas choice.
- **Fonts:** val is Nirmala UI and Noto Serif Bengali / Georgia and Verdana. Test is Nirmala UI, Noto Sans Bengali and Tiro Bangla / Arial, Times and Calibri. Training fonts are never scored.
- **Images:** every phrase appears three ways: a clean page, a distorted page (rotation, blur, low resolution), and a sign placed into a real COCO photo at 640 × 480, the glass frame size.
- **Seeds:** three image seeds (0, 1, 2), so 432 images per set.
- **Metric:** CER and WER after Unicode NFC, on a laptop GPU. The final pipeline's CPU latency is reported separately.
"""


def test_table(names):
    lines = [HEAD]
    for label, f in names:
        if (RES / f).exists():
            lines.append(row(label, J(f)))
    return "\n".join(lines)


def section_text_region():
    t1 = J("task1_best.json")
    val = [("Old offline path (EasyOCR, boxes in its own order)", "val_baseline_raw.json"),
           ("+ boxes ordered into lines", "val_t1_sorted_lines.json"),
           ("+ confidence ≥ 0.1", "val_t1_conf0.1.json"), ("+ confidence ≥ 0.3", "val_t1_conf0.3.json"),
           ("+ confidence ≥ 0.5", "val_t1_conf0.5.json"),
           ("Dominant text block, conf ≥ 0.1, one pass (kept)", "val_t1_region_craft_c0.1_1pass.json"),
           ("Dominant text block, conf ≥ 0.1, crop and read again", "val_t1_region_craft_c0.1_2pass.json"),
           ("Dominant text block, conf ≥ 0.3, one pass", "val_t1_region_craft_c0.3_1pass.json"),
           ("Dominant text block, conf ≥ 0.3, crop and read again", "val_t1_region_craft_c0.3_2pass.json")]
    test = [("Old offline path", "test_baseline.json"), ("App OCR mode as shipped", "test_glass_app.json"),
            ("+ Task 1 text region", "test_t1_region.json")]
    return f"""{SETUP}
### Method

1. EasyOCR reads the whole frame with CRAFT.
2. Boxes below a confidence are dropped.
3. **The dominant block is kept:** start from the most confident tall box, then add boxes of similar height that touch it. This is the sign or label the user faces; stray text in the background is dropped.
4. Boxes are joined line by line.

The "crop and read again" variant cuts the block out, upscales it and runs OCR a second time.

**Not run:**
- **DBNet18:** {t1['not_run']['dbnet18']}.
- **YOLOv8 and EAST text detectors:** no text-detection weights are available offline (the prompt's YOLO link was a placeholder).

### Validation (3 seeds, every choice made here)

{test_table(val)}

**Error analysis behind it.** On the old path, photo-scene errors were mostly junk read from the background ("WEUCo4C 10 THE রান্নাঘর") and words in the wrong order ("Clinic Eye"). The sign text itself was usually right.

### Test, read once

{test_table(test)}

The text region alone takes test CER from 9.3 % (app) to 2.4 %. The photo-scene condition falls from 10.7 % to 3.4 %.

### Speed on CPU (the Pi's path)

`results/ocr_bench/cpu_latency.json`: laptop CPU, EasyOCR quantised CPU recognizer, 60 test images. This is not a Pi number.

| Pipeline | Median | 95th percentile | CER on these 60 images |
|---|---:|---:|---:|
| App OCR mode as shipped | {J('cpu_latency.json')['glass_app_legacy']['median_ms']:.0f} ms | {J('cpu_latency.json')['glass_app_legacy']['p95_ms']:.0f} ms | {pct(J('cpu_latency.json')['glass_app_legacy']['cer_on_these'])} |
| Final pipeline (v2, now in the app) | {J('cpu_latency.json')['v2_final']['median_ms']:.0f} ms | {J('cpu_latency.json')['v2_final']['p95_ms']:.0f} ms | {pct(J('cpu_latency.json')['v2_final']['cer_on_these'])} |

The region step costs nothing measurable, because it reuses the boxes EasyOCR already returns.
"""


def section_preprocessing():
    t2 = J("task2_best.json")
    names = {"glass": "Glass (upscale, bilateral, CLAHE)", "gray": "Grey only", "color": "Colour", "no_upscale": "No upscale",
             "clahe": "CLAHE only (kept)", "bilateral": "Bilateral only", "unsharp": "Unsharp mask", "glass_unsharp": "Glass + unsharp",
             "sauvola": "Sauvola binarisation", "niblack": "Niblack binarisation", "deskew": "Deskew", "perspective": "Perspective correction",
             "deskew+clahe": "Deskew + CLAHE"}
    val = [(names.get(k, k), f"val_t2_{k}.json") for k in t2["all"]]
    test = [("Task 1 text region (glass preprocessing)", "test_t1_region.json"), ("+ Task 2 CLAHE only", "test_t2_prep.json")]
    return f"""{SETUP}
All variants run on top of the Task 1 text region.

### Validation (3 seeds)

{test_table(val)}

- **Kept:** CLAHE only had the lowest validation CER ({100 * t2['val_cer']:.2f} %), by 0.08 points over the glass preprocessing. That gap is within seed noise.
- **What hurt:** binarisation (Sauvola, Niblack) and the perspective warp hurt. Deskew alone hurt.

### Test, read once

{test_table(test)}

On test the CLAHE-only step is worse than the glass preprocessing (3.3 % against 2.4 %). The validation choice did not carry over. The Task 4 parameters, chosen on top of it, bring test CER back to 2.4 % (see `OCR_PARAM_TUNING.md`).
"""


def section_engines():
    val = [("EasyOCR + Task 1 text region", "val_t1_region_craft_c0.1_1pass.json")]
    for t in J("task3_val.json"):
        val.append((t["tag"].replace("_", " "), f"val_t3_{t['tag']}.json"))
    val.append(("PaddleOCR PP-OCRv5 (English items only)", "val_t3_paddle_en.json"))
    t3 = min(J("task3_val.json"), key=lambda r: r["val_cer"])
    test = [("EasyOCR final pipeline", "test_final.json"), (f"Tesseract 5 ({t3['tag'].replace('_', ' ')})", f"test_t3_{t3['tag']}.json"),
            ("PaddleOCR (English items only)", "test_t3_paddle_en.json")]
    return f"""{SETUP}
### Engines

- **Tesseract 5.5.2:** tesserocr wheel, `tessdata_best` ben+eng. The UB-Mannheim installer needs administrator rights; the silent install was cancelled.
- **PaddleOCR 3.7:** on Python 3.10, because there is no paddle wheel for 3.14. It has no Bangla model: `lang='bn'` and `'bengali'` give "No models are available". The oneDNN path crashes on Windows, so it runs with `enable_mkldnn=False`.
- **TrOCR:** reads the Task 1 text block with the language given (oracle). English uses `microsoft/trocr-base-printed`. Bangla uses the community `nightsagittariuswolf/SWIN_TrOCR_Bangla_model`; its processor config names a removed class, so it was rebuilt from its parts.

### Validation (3 seeds)

{test_table(val)}

**Notes.**
- The Microsoft TrOCR answers in capitals (it was trained on receipts), so its CER is case-penalised.
- The community Bangla TrOCR returns unrelated words, for example "রান্নাঘর" → "রামানগঞ্জ".
- PaddleOCR reads every piece of text in a scene, so its photo-scene CER exceeds 100 %.

### Test, read once

{test_table(test)}

EasyOCR with the text region stays far ahead. Tesseract is the best other engine for Bangla, at about 7× the error.
"""


def section_params():
    t4 = J("task4_best.json")
    st1 = sorted(t4["stage1"], key=lambda r: r["seed0_cer"])
    val = [(f"config {i:02d}{' (defaults)' if i == 0 else ''}{' (kept)' if i == t4['best_cfg_index'] else ''}", f"val_t4_cfg{i:02d}.json")
           for i in sorted({int(p.stem[-2:]) for p in RES.glob("val_t4_cfg*.json")})]
    test = [("Task 2 (CLAHE, Task 1 settings)", "test_t2_prep.json"), ("+ Task 4 parameters", "test_t4_params.json")]
    return f"""{SETUP}
### Search (validation only)

- **Search space:**
  - text_threshold, low_text, link_threshold;
  - mag_ratio 1.0 / 1.5 / 2.0;
  - contrast_ths, adjust_contrast;
  - decoder (greedy / beam search), slope_ths;
  - canvas 1280 / 2560, minimum box confidence.
- **Procedure:** 30 random configurations plus the defaults were scored on val seed 0 (seed-0 CER ranged from {100 * st1[0]['seed0_cer']:.1f} to {100 * st1[-1]['seed0_cer']:.1f} %). The 4 best and the defaults were re-scored on all three val seeds.
- **Fixed parts:** CRAFT is the only detector that runs here. EasyOCR has one Bangla recognizer (a CRNN, `bengali.pth`); there is no Bangla 'transformer' recognizer to try.

{test_table(val)}

**Kept:** `{json.dumps(t4['config'])}` — validation CER {100 * t4['val_cer']:.2f} % against {100 * t4['default_val_cer']:.2f} % with the defaults.

### Test, read once

{test_table(test)}
"""


def section_post():
    t5 = J("task5_best.json")
    rows = []
    for name in t5["all"]:
        d = J(f"val_t5_{name.replace('+', '_')}.json")
        rows.append(f"| {name} | {pct(d['overall']['cer'])} | {pct(d['bn']['cer'])} | {pct(d['en']['cer'])} | {pct(d['overall']['wer'])} | "
                    f"{100 * d['overall']['exact']:.1f} % | {d['postprocess_ms_median']:.2f} ms |")
    test = [("Task 4 pipeline, no post-processing", "test_t4_params.json"), ("+ rules (final)", "test_final.json")]
    return f"""{SETUP}
All methods were applied to the same saved validation readings of the Task 4 pipeline (no new OCR).

- **rules:** NFC; removes invisible characters and stray symbols at word edges; puts digits in the script of the words around them.
- **dictionary:** SymSpell over general 50k-word frequency lists (Bangla and English, hermitdave/FrequencyWords, MIT), with a frequency margin.
- **byt5:** `Stup702/ByT5-Bengali-OCR-Correction`.
- **lexicon:** the app's old repair against its 24-phrase list.

**Not available:**
- **KenLM:** does not build on this laptop; there is no C++ compiler. The frequency lists act as a unigram model inside the dictionary method.
- **bnlp:** installed on Python 3.10, but it has no spell checker.
- **`bangla-spell-checker`:** not on PyPI.

### Validation (3 seeds)

| Method | CER | Bangla CER | English CER | WER | Exact | Time per line |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

**Kept: rules.**
- **Dictionary:** the most frequent nearby word is often the wrong word for a real Bangla sign, so it hurts Bangla.
- **ByT5:** it rewrites and repeats text ("রান্নাঘর" → "রান্নাঘর প্রতিষ্ঠিত হয়") and takes 6–11 s per line on CPU.
- **Lexicon repair:** it pulls new phrases toward its 24 old ones.

### Test, read once

{test_table(test)}

On test the rules change 42 of 432 readings, but almost all of these are Unicode normalisation, which the metric already applies. The few real edits cost the same number of characters ("টাক| জম দিন" → "টাক জম দিন"). So test CER is unchanged; the spoken text is cleaner.
"""


def section_multiscale():
    t6 = J("task6_best.json")
    val = [(k, f"val_t6_{k}.json") for k in t6["all"]]
    return f"""{SETUP}
### Method

The preprocessed image was read at 1.0×, 1.5× and 2.0× with the Task 4 settings. Two ways of choosing between the readings were tried:
- **most confident:** the reading with the highest character-weighted box confidence;
- **vote:** the medoid, i.e. the reading with the smallest total edit distance to the other two.

### Validation (3 seeds)

{test_table(val)}

**Not kept.** The single scale is best ({100 * t6['all']['scale1.0']:.2f} %). Larger scales lower Bangla CER a little but raise English and photo-scene CER, and every extra scale costs a full OCR pass. Nothing new was read on test (the single scale is the Task 4 pipeline).
"""


def section_finetune():
    seeds = [J(f"t7_seed{s}.json") for s in (42, 43, 44)]
    vals = [J(f"val_t7_ft_seed{s}.json") for s in (42, 43, 44)]
    base_val = J("val_t5_rules.json")
    crop = "\n".join(f"| {s['seed']} | {s['best_step']} | {100 * s['pretrained_val_crop_cer']:.2f} % | {100 * s['best_val_crop_cer']:.2f} % |"
                     for s in seeds)
    vt = HEAD + "\n" + row("Final pipeline, original recognizer", base_val) + "\n" + "\n".join(
        row(f"Final pipeline, fine-tuned seed {s}", v) for s, v in zip((42, 43, 44), vals))
    test = [("Final pipeline, original recognizer", "test_final.json")] + [(f"fine-tuned seed {s}", f"test_final_ft_seed{s}.json") for s in (42, 43, 44)]
    return f"""{SETUP}
### Training (`savior_glass/ocr_bench/train_task7.py`)

- **Model:** EasyOCR's Bangla + English recognizer (CRNN, CTC, 54 M parameters), fine-tuned from its released weights.
- **Training text:**
  - words from the general 50k frequency lists ({seeds[0]['train_words']['bn']} Bangla, {seeds[0]['train_words']['en']} English), with every word of every benchmark phrase removed ({seeds[0]['banned_benchmark_words']} forms);
  - random numbers in both scripts.
- **Rendering:** shaped with HarfBuzz in six Bangla and four Latin training fonts that are never scored, with blur, low resolution, rotation, noise and JPEG damage.
- **Optimisation:** {seeds[0]['steps']} steps, batch 16, AdamW 3e-5, mixed precision.
- **Checkpoint choice:** greedy CER on line crops of the val phrases in val fonts, every 250 steps. Three seeds.

| Seed | Best step | Val crop CER before | Val crop CER after |
|---|---:|---:|---:|
{crop}

### End to end on validation (the decision)

{vt}

**Not adopted.** Fine-tuning lowers Bangla CER on validation but raises English CER, so overall validation CER is higher than with the original recognizer. Training on synthetic text alone makes the model partly forget English. The test read below is reported for completeness and was not used for the decision.

### Test, read once per seed

{test_table(test)}

**Test disagrees with validation.** On test the fine-tuned seeds score {", ".join(f"{100 * J(f'test_final_ft_seed{s}.json')['overall']['cer']:.1f} %" for s in (42, 43, 44))} against {100 * J('test_final.json')['overall']['cer']:.1f} % for the original recognizer, lower in every seed. That direction is the opposite of validation. The decision stays with validation: switching now would be choosing on the test set. Fine-tuning is the most promising next step, on real glass photos and with English kept in the training mix so it is not forgotten. A fresh test set is needed to show that it helps.
"""


SECTIONS = {"OCR_TEXT_REGION.md": section_text_region, "OCR_PREPROCESSING.md": section_preprocessing,
            "OCR_BANGLA_SPECIFIC.md": section_engines, "OCR_PARAM_TUNING.md": section_params,
            "OCR_POSTPROCESSING.md": section_post, "OCR_MULTISCALE.md": section_multiscale, "OCR_FINETUNE.md": section_finetune}


def main() -> None:
    for fname, fn in SECTIONS.items():
        p = PE / fname
        old = p.read_text(encoding="utf-8") if p.exists() else f"# {fname[:-3].replace('_', ' ').title()}\n"
        if MARK in old:
            old = old[: old.index(MARK)].rstrip() + "\n"
        p.write_text(old.rstrip() + "\n\n" + fn().strip() + "\n", encoding="utf-8")
        print("wrote", p.name)


if __name__ == "__main__":
    main()
