# Auto feature discovery

| item | class | reason |
| --- | --- | --- |
| Pi 5 latency script already in `benchmark_pi5.py` | PROTOCOL_READY | No Pi attached. |
| Entropy rejection already measured | ADD NOW | Written up in `REJECTION_PRODUCTION.md`. The ensemble rule's occluded wrong-verdict share is 0.004807692307692308 (`occlusion_decision.json`). |
| Data-dependent PAC-Bayes | ADD NOW | Bound 0.1127 is in `THEORY_FULL.md`. |
| VCIE 1-view checkpoint and MTPT 6+3 | ADD NOW | Test files exist and are cited in `PAPER_FINAL.md`. |
| Multi-note detector composites | ADD NOW | Measured in `BBOX_EXTENDED.md`. Every pasted note at IoU ≥ 0.5. |
| Live camera boxes | NOT_NEEDED for a fabricated score | No controlled live capture in this session. |
| ModelNet, MOSI, ADNI, COCO multi-view | NOT_NEEDED as a download in this pass | Not present locally. Marked NOT_MEASURED in `VCDS_UNIVERSAL.md`. |
| NSTU / USD / EUR / INR authenticity | NOT_NEEDED as a fake transfer number | NSTU labels are not genuine/counterfeit. |
| User study, TFLite phone latency | PROTOCOL_READY | No participants and no phone. |
| Deeper VCIE or extra MTPT weighting | NOT_NEEDED | Both targets were already met by the measured runs. |
| Comparison folder present | COUNTED | `comparison_dataset_manifest.json`. Denomination and coin labels, plus ModelNet point clouds. |
| ModelNet turntable probe | MEASURED | `subset10_seed42_e40.json`. |
