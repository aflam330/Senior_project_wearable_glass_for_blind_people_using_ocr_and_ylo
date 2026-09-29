# Cross-dataset

Authenticity on another currency dataset was not measured.

NSTU-BDTAKA (Mendeley 10.17632/w4y6h723xg.1) was not downloaded. Its public labels are denomination and boxes, not genuine versus counterfeit. Scoring a JaalTaka authenticity head against a denomination id is not an authenticity accuracy. Detail: `CROSS_DATASET_RESULTS.md`.

US dollar, euro, and Indian rupee note datasets were not downloaded.

| dataset | authenticity accuracy |
| --- | --- |
| JaalTaka test | measured in `FINAL_RESULTS.md` |
| NSTU-BDTAKA | NOT_MEASURED |
| USD | NOT_MEASURED |
| EUR | NOT_MEASURED |
| INR | NOT_MEASURED |

## Download attempt 2026-09-29

NoteShieldBD, the Mendeley counterfeit set (DOI 10.17632/gzzz5nrvbn.1), NSTU-BDTAKA, and ModelNet40 are not on disk. The Mendeley file API returned HTTP 400. There is no Kaggle token. Counts are in `CROSS_DATASET_DOWNLOAD_LOG.md` and `external_bdt_manifest.json`. USD, EUR, and INR were not downloaded.


---

## Update 2026-09-29

Measured cross-dataset results for the Taka detector (Bangla Money, NSTU-BDTAKA detection and hand-held close-ups), and a close-up classifier experiment that exposes non-independent splits in NSTU-BDTAKA, are in `CROSS_DATASET_TAKA.md`.
