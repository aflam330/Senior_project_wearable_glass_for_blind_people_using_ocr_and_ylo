# Open-source release

The repository remote is `https://github.com/aflam330/Senior_project_wearable_glass_for_blind_people_using_ocr_and_ylo`.

Added for release:

- `LICENSE` (MIT)
- `CITATION.cff`
- `CONTRIBUTING.md`
- `setup.py`

Model cards are in `paper_evidence/model_cards/`. They describe checkpoints that already have test files. They are not Hugging Face uploads. No model was pushed to the Hub in this pass.

A Docker file that would run the Pi 5 benchmark is not an image that was built here. The Pi script is `savior_glass/scripts/deploy_pi5.sh` plus `benchmark_pi5.py`.

`scripts/validate_claims.py` remains the claim check. It was not re-run as a substitute for the new result files; those files cite their JSON sources directly.
