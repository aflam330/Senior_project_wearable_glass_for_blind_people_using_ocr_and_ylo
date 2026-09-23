# Reproducibility

```
cd realtime_bangla_taka_detection
python run_research_pipeline.py --config configs/final_research.yaml
```

Every run directory under `results/qduig/` writes config.yaml, environment.txt, git_commit.txt, seed.txt, dataset_manifest.json, train_log.csv, metrics, predictions.
Existing CAMVA artifacts under `results/camva/` are not overwritten.
