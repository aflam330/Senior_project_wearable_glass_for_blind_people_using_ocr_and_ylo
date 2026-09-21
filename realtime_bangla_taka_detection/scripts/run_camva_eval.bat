@echo off
REM Run after train_camva.py finishes. Does not retrain.
cd /d "%~dp0\.."
python -u scripts\evaluate_camva.py
python -u scripts\calibrate_camva.py
python -u scripts\evaluate_ablation.py
python -u scripts\evaluate_adaptive.py
python -u scripts\evaluate_camva_robustness.py
python -u scripts\write_camva_report.py
echo Done. Read results\camva\README.md
