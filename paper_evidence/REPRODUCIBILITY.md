# Reproducibility

Commands are run from `realtime_bangla_taka_detection` with Python 3.10. The working GPU stack on this machine is `py -3.10` and `torch==1.12.1+cu113`. `NOVEL_WORKERS` must be 0 on this Windows host. The test split is not used to choose checkpoints or to fit temperature or HER.

## Dataset

JaalTaka lives outside the repository at `data set/JaalTaka`. It is not in git. The note-disjoint split is `results/camva/splits/split_metadata.json`, seed 42, 974 / 208 / 208 notes.

## Environment

```
py -3.10 -m pip install numpy==1.26.4 pillow opencv-python pyyaml matplotlib
```

Torch is the CUDA 11.3 wheel for Python 3.10. A newer wheel does not see the GPU on this driver.

## Prefix-robust model (already trained)

```
py -3.10 scripts/train_qduig.py --config configs/proposed_prefix.yaml --seed 42 --output-dir results/qduig/prefix/seed42
py -3.10 scripts/train_qduig.py --config configs/proposed_prefix_ft.yaml --seed 42 --output-dir results/qduig/prefix_ft/seed42 --resume results/qduig/prefix/seed42/checkpoint.pt
py -3.10 scripts/eval_prefix_views.py --checkpoint results/qduig/prefix_ft/seed42/checkpoint.pt --config configs/proposed_prefix_ft.yaml --split test --seed 42 --output-dir results/qduig/prefix_ft/seed42/test_views
```

Seeds 43 and 44:

```
py -3.10 scripts/run_phase1_multiseed.py
```

## Other algorithms

```
set NOVEL_WORKERS=0
set NOVEL_COMPILE=0
py -3.10 scripts/train/train_novel.py --algo ndal --seed 42 --config configs/v2/ndal.yaml --output-dir results/novel_v2/ndal/seed42
py -3.10 scripts/eval/eval_novel.py --algo ndal --seed 42 --config configs/v2/ndal.yaml --checkpoint results/novel_v2/ndal/seed42/checkpoint.pt --split test --output-dir results/novel_v2/ndal/seed42/test
```

UGF, SFAQ, IGCR, and OGPD v1 use the yaml files in `configs/` and `results/novel/`. The v2 algorithms use `configs/v2/`.

## Ablations

```
py -3.10 scripts/train/run_ablation_prefix.py
```

Full and the CNN+ViT baseline are not retrained. `-cost` and `-calibration` are evaluations of the saved prefix-robust checkpoint.

## Calibration, figures, tables

```
py -3.10 scripts/eval/run_calibration_suite.py
py -3.10 scripts/eval/make_publication_figures.py
py -3.10 scripts/eval/write_paper_tables.py
py -3.10 scripts/eval/extend_theory.py
```

## Not reproducible here

Pi 5 measurement, a phone export, and a user study. Those are NOT_MEASURED.
