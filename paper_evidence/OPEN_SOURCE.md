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

---

## Update 2026-09-29

**Prepared, not published.** Nothing was pushed to GitHub or uploaded to Hugging Face. Publishing is left to the authors.

| Item | State |
|---|---|
| `LICENSE`, `CITATION.cff`, `CONTRIBUTING.md`, `setup.py`, root `README.md` | present |
| `requirements.txt` for both code bases | present. The PyTorch pins in different documents still disagree; see `CORRECTIONS.md` and `REPRODUCIBILITY.md` |
| Model cards with Hugging Face front-matter | `model_cards/TAKA_DETECTOR.md` (new) and `model_cards/PRMVT.md` (rewritten with the out-of-scope warning); MTPT, VCIE and OCCLUSION cards unchanged |
| Pi 5 container | `savior_glass/docker/Dockerfile`: **not built** (no Docker host or Pi available) |
| Benchmark package | see `BENCHMARK_RELEASE.md` |

**Blockers before a public release:**
1. The JaalTaka dataset's source and license are undocumented. Without them the split and checkpoints trained on it cannot be redistributed with confidence.
2. The COCO, BanglaTaka, NSTU-BDTAKA, Bangla Money and RAF-DB licenses must be checked for redistributing derived composites and weights.

**Resolved later on 2026-09-29: JaalTaka provenance.** JaalTaka is published on Mendeley Data:
- Piyas, Tisha, Rahman, Islam and Preenon, V2, 11 September 2025
- DOI 10.17632/2m7wk5cy4c.2, **CC BY 4.0**
- Data article in Data in Brief (S235234092501090X)
- genuine notes from three Bangladeshi banks, counterfeit notes via the Rapid Action Battalion, smartphone cameras

CC BY 4.0 allows redistribution with attribution, so blocker 1 above is resolved. Cite the dataset in any release. The thesis `ref.bib` entry now has the authors and DOI.
