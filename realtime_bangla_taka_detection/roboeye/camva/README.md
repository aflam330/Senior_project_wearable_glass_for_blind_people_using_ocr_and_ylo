# CAMVA module (do not overwrite authenticity.py)

New package: `roboeye/camva/`

```
python scripts/make_camva_splits.py
python scripts/train_camva.py --epochs 6 --seeds 1
python scripts/evaluate_camva.py
python scripts/calibrate_camva.py
python scripts/evaluate_ablation.py
python scripts/evaluate_adaptive.py
python scripts/evaluate_camva_robustness.py
python scripts/write_camva_report.py
python scripts/validate_claims.py
```

Old `models/authenticity_cnn_vit.pt` is never written by these scripts.
Artifacts: `results/camva/`
If CAMVA loses to the retrained baseline, `evaluate_camva.py` still prints ABSOLUTE IMPROVEMENT (negative) and paired p-values. Do not round that into a SOTA claim.
