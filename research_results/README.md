# Research evidence database

One place for the project's **experiment-level results**, in a form that can be turned into tables, graphs and paper text without retyping numbers.

It is not a log. A record is one finished experiment: a model or pipeline evaluated on a dataset under stated conditions, with its metrics, the file the numbers came from, and a verification status. Frame-by-frame output stays in the normal result files and field logs.

```text
research_results/
├── experiments.json        the master file (source of truth)
├── schema.json             JSON Schema of the master file
├── results_db.py           load / validate / append / collect / review / export
├── export_results.py       CSV files and ready-to-paste tables
├── import_existing.py      reads the result files already in the repository
├── hooks.py                what evaluation scripts call to record a result
├── inbox/                  records written by scripts, waiting to be merged
├── evidence/               small evidence copies for results whose raw files are not committed (field runs)
├── exports/                generated: experiments.csv, metrics.csv, comparisons.csv, claims.csv, comparison_tables.md
└── test_results_db.py, test_hooks.py
```

## 1. Why it exists

Results were spread over about 200 JSON files and 190 reports. Each paper table meant finding the right file and copying numbers by hand. Here every result is one record with the same fields, so a comparison across models, datasets, settings or hardware is a filter and a group-by.

## 2. The JSON structure

`experiments.json` has three lists: `experiments`, `comparison_groups` and `publication_claims`. An experiment record uses only the blocks that apply to it.

| Block | Meaning |
|---|---|
| `experiment_id` | `EXP_0001`, … Assigned by the tool, never reused |
| `experiment_name`, `task`, `mode`, `algorithm` | What was tested |
| `timestamp`, `recorded_at` | When the result was produced; when the record was written |
| `model` | Name, architecture, weights, training settings |
| `dataset` | Name, split, samples, classes, real or synthetic |
| `configuration` | Thresholds and settings that define this run |
| `metrics` | Quality numbers: accuracy, F1, mAP, CER, WER, AUC, … (flexible dictionary) |
| `performance` | Speed and resources: latency, FPS, memory, temperature |
| `raw_measurements` | Small raw values behind the metrics (counts, confusion matrix, intervals), or paths to large files |
| `human_feedback` | Participant results, kept apart from model metrics |
| `hardware`, `software_environment` | Device, CPU, GPU, OS, library versions |
| `test_conditions` | Lighting, number of views, image condition, … |
| `run` | `seed`, and `rerun_of` when it repeats an earlier record |
| `runs` | Values of repeated runs inside one record (for example CER per image seed) |
| `experiment_group` | Runs that repeat the same experiment (the three seeds of one method) |
| `comparison_group`, `comparison.variant` | Which comparison the record belongs to, and its name inside it |
| `comparison.decision` | `kept`, `not_kept`, `baseline`, where a choice was made |
| `ablation` | Component and configuration, for ablation chains |
| `versioning.code_commit` | Git commit of the code or of the evidence file |
| `evidence` | File, key path inside it (`selector`), and SHA-256 |
| `verification` | How the record was verified |
| `status` | See section 9 |
| `notes` | Short experiment-level observations |

Metric names are free, so OCR records carry `cer` / `wer`, detector records carry `map50`, device records carry `latency_ms_median`. Nothing is computed that was not measured.

## 3. How experiments are recorded

There are three ways in, and all end in `experiments.json`:

1. **Import of existing result files:** `python research_results/import_existing.py`. It reads the numbers from the result files; nothing is typed.
2. **Automatically, from evaluation scripts** (section 5).
3. **By hand** (section 4).

**Rules the tool enforces:**
- Records are append-only.
- Ids are never reused.
- The same evidence (file + selector + file hash) is recorded once.
- If a result file changes and is imported again, a **new** record is added and linked to the old one with `run.rerun_of`; the old record stays as it was.

## 4. Adding an experiment by hand

Write a small JSON file and add it:

```json
{
  "experiment_name": "Taka detector on glass-camera photos",
  "task": "currency_detection",
  "status": "unverified",
  "model": {"name": "YOLOv8s Taka detector"},
  "dataset": {"name": "Glass photos, October", "split": "test", "samples": 120, "real_or_synthetic": "real"},
  "metrics": {"accuracy": 0.0},
  "hardware": {"device": "Raspberry Pi 5"},
  "test_conditions": {"lighting": "indoor", "distance": "40 cm"},
  "evidence": [{"type": "evaluation_output", "file": "savior_glass/results/glass_photos_eval.json"}],
  "notes": ["Replace 0.0 with the measured value. Do not add a record before the experiment has run."]
}
```

```bash
python research_results/results_db.py add my_record.json
```

- **Planned work:** use `"status": "planned"` and no metrics.
- **A run that crashed:** use `"status": "failed"` and say why in `notes`.
- **`"verified"`:** the evidence file must exist and `verification.method` must say how it was checked.

## 5. Automatic collection

These scripts write one record per finished experiment into `inbox/` (they never touch the master file):

| Script | What is recorded |
|---|---|
| `savior_glass/ocr_bench/bench.py` (`save`, used by every OCR benchmark runner) | One record per method × phrase set: CER, WER, exact match, CER per language and condition, CER per image seed |
| `savior_glass/scripts/benchmark_pi5.py` (full runs) | One record per module: median / 95th-percentile latency, FPS, memory, temperature; one for the sustained run |
| `savior_glass/scripts/field_report.py <run> --record` | One record per real-time test run: latency per mode, temperature, memory, and accuracy if `labels.csv` was filled in |
| `savior_glass/scripts/measure_mode_switch.py <out.json>` | Mode-switch timings |
| `realtime_bangla_taka_detection/scripts/train/train_watermark_localizer.py` | One record per seed |

Any other script can record a result with one call:

```python
from research_results import hooks
hooks.call("generic", "ResNet-50 fine-tune, seed 42", "counterfeit_detection", out_path,
           metrics={"accuracy_k6": acc}, run={"seed": 42})
```

Then merge, validate and export:

```bash
python research_results/results_db.py collect
```

- **Why an inbox:** the Pi and the laptop can both record results and push them through git without ever editing the same file.
- **Merged files:** they move to `inbox/merged/`.
- **Speed:** recording happens once, after an evaluation ends; it adds nothing to inference time.

## 6. CSV export

```bash
python research_results/export_results.py
```

| File | Shape | Use |
|---|---|---|
| `exports/experiments.csv` | One row per experiment, one column per metric | Excel, Google Sheets, SPSS |
| `exports/metrics.csv` | One row per number, with model, dataset, split, hardware, seed, group | pandas, R, MATLAB, Origin: filter and plot |
| `exports/comparisons.csv` | Per comparison group and variant: n runs, mean, sd, min, max | Tables, bar charts with error bars |
| `exports/claims.csv` | Claims and their supporting experiments | Checking a paper draft |
| `exports/comparison_tables.md` | One table per comparison group and split | Paste into a report |

The CSV files are generated. Edit the JSON (through the tool), never the CSV.

## 7. Graph-ready data

```python
import pandas as pd
m = pd.read_csv("research_results/exports/metrics.csv")

# accuracy by method on unseen prints, mean and sd over seeds
up = m[(m.comparison_group_id == "UNSEEN_PRINTS") & (m.metric == "accuracy_k6")]
up.groupby("variant")["value"].agg(["mean", "std", "count"]).plot.bar(y="mean", yerr="std")

# latency by module on the Pi
pi = m[(m.comparison_group_id == "PI5_LATENCY") & (m.metric == "latency_ms_median")]
pi.plot.barh(x="variant", y="value")

# OCR ablation on the held-out test set
ocr = m[(m.comparison_group_id == "OCR_BENCHMARK") & (m.metric == "cer") & m.dataset_split.str.startswith("held-out")]
```

Comparison groups currently in the file: `TAKA_DETECTOR`, `JAALTAKA_VIEW_COUNT`, `JAALTAKA_FROZEN_PROBES`, `UNSEEN_PRINTS`, `WATERMARK_CHECK`, `COUNTERFEIT_SAFETY`, `WHOLE_NOTE_TRANSFER`, `OCR_BENCHMARK`, `OCR_CANVAS_SIZE`, `PI5_LATENCY`, `DEVICE_BEHAVIOUR`.

## 8. Keeping history

- **No delete or edit command.** The tool has neither.
- **Re-runs:** a re-run is a new record. If its evidence file was overwritten by the re-run, the earlier record is still in the file with its numbers and its old file hash; `python research_results/results_db.py check-evidence` lists such records.
- **Failed, negative and planned experiments stay in the file.**
  - `failed`: it ran and crashed.
  - `not_kept` in `comparison.decision`: it ran and lost.
  - `planned`: it has not run.
- **Git:** `experiments.json` is committed; git history is the second line of defence.

## 9. Verification status

| Status | Meaning | How a record gets it |
|---|---|---|
| `verified` | Supported by an evaluation output that exists | Import of a saved result file, or `review` |
| `partially_verified` | Real numbers, not yet looked at by a person | Every automatically collected record starts here |
| `unverified` | A value without checkable evidence | Hand-entered |
| `failed` | The experiment ran and failed | Hand-entered or imported |
| `planned` | Not run; carries no metrics | Hand-entered |

`verified` is never a default. To promote a record after checking it:

```bash
python research_results/results_db.py review EXP_0195 EXP_0196 --by "your name" --note "checked against the Pi JSON"
```

- **What changes:** only the status and the verification note. The change is logged in the record's `status_history`.
- **When it is refused:** if the evidence file is missing or no longer matches its recorded hash.

## 10. Publication claims

A claim is a sentence you intend to print, linked to the experiments behind it. Claims are only added by you:

```json
{"claim": "On unseen counterfeit prints the watermark hybrid reaches 95.0 % at six views over three seeds.",
 "supporting_experiments": ["EXP_0028", "EXP_0035", "EXP_0042"], "status": "verified"}
```

```bash
python research_results/results_db.py add-claim my_claim.json
```

- **Validation:** a claim cannot be `verified` unless every supporting experiment exists and is itself `verified`.
- **Find the ids** in `exports/experiments.csv`.
- **The older `paper_evidence/CLAIM_REGISTRY.json`** (single numbers checked against result files) stays as it is.

## Checks

```bash
python research_results/results_db.py validate      # structure, ids, statuses, numbers, evidence files, claims
python research_results/results_db.py summary
python research_results/test_results_db.py          # safety rules, on a temporary copy
python research_results/test_hooks.py               # automatic collection, on a temporary copy
```
