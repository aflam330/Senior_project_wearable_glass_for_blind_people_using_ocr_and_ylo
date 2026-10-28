# SARA paper package

This package holds everything needed to write a journal or conference paper about **SARA (Smart Assistive Reading Aid)**, the offline Bangla assistive smart glass on a Raspberry Pi 5, built as a CSE senior project at Independent University, Bangladesh (2026).

## Start here

1. `01_PROJECT_FULL_INFORMATION.md`: the whole project in one document. It covers the problem, hardware, software, datasets, every method, the results, what failed, the corrections, limitations, and what may and may not be claimed.
2. `02_RESULTS_MASTER_TABLE.md` / `.csv`: every measured number, with its test set, conditions, thesis table, evidence note and status (current / superseded / not a test / informal / not measured).
3. `03_WRITING_GUIDE.md`: how the two drafts are organised, what to fill in before submission, and a checklist.
4. `latex/`: two compiled paper drafts.
   - `ieee_conference/` uses IEEEtran (conference mode), with `main.tex`, `references.bib`, `figures/` and the compiled `main.pdf`.
   - `q1_journal/` uses elsarticle (Elsevier, `preprint,12pt`), with the same files.
   - Both folders carry the class and bibliography-style files, so they compile with any TeX installation: `pdflatex main`, `bibtex main`, `pdflatex main` twice.

## Folders

| Folder | Contents |
|---|---|
| `figures/thesis/` | Every figure and photo used in the thesis (diagrams, charts, prototype and hardware photos, live tests, UV photos, JaalTaka views, literature device photos) |
| `figures/thesis_extra/` | Thesis figures 56 (voice commands in noise) and 58 (hand-held close-ups); figures 52–55 and 57 are in `figures/publication/` |
| `figures/publication/` | Publication-style charts made from the result files (PNG and PDF), figures 1–51 |
| `figures/prototype_and_test_photos/` | Photos of prototype version 2, worn tests and live note tests |
| `figures/illustrations_not_evidence/` | The two cartoon / photo-style infographics used in the slides. They are illustrations, not evidence, and must be marked as such if used |
| `figures/FIGURE_INDEX.md` / `.csv` | What each figure shows, its thesis caption, and warnings about outdated figures |
| `results_data/glass_app/` | Result files of the glass application: OCR, objects, emotion, voice, speech, UV, Pi benchmarks, multi-note, haptics |
| `results_data/taka_and_counterfeit/` | Result files of the Taka detector and the counterfeit and watermark research: splits, per-seed evaluations, serial audit, DMWR, localizer, safety, calibration, robustness, hand-held fine-tune |
| `results_data/research_results/` | Earlier research-pipeline results |
| `results_data/pi5_benchmark_report/` | The Raspberry Pi 5 benchmark report of 13 October 2026, with raw files and plots |
| `evidence_notes/` | 240+ dated experiment notes (Markdown), the claim registry (`CLAIM_REGISTRY.json`, `claim_registry.csv`), corrections (`CORRECTIONS.md`), the claim value audit, LaTeX tables, model cards, and the user-study protocol (templates only, no participant data) |
| `documents/` | Thesis PDF (124 pages), features PDF, results/benchmarks PDF, the presentation (PPTX and PDF) with its change log and QA report, the literature spreadsheet, and the thesis LaTeX source with `ref.bib` |

## Important

- **Which numbers to use:** the thesis is the reference. Where an older evidence note gives a different number, the thesis and `CORRECTIONS.md` say which one is current. `02_RESULTS_MASTER_TABLE.md` marks superseded values.
- **Why some notes disagree:** the evidence notes are dated working notes and some describe experiments that were later corrected. They are kept so that the path to each result can be followed. Do not cite a note without checking its status.
- **What was left out:** model weights, datasets, audio, pickled intermediate files, the API key and any personal or participant data. Weights and code are in the two GitHub repositories listed in `01_PROJECT_FULL_INFORMATION.md`.
- **Before submitting either draft:**
  - add the authors' e-mail addresses and ORCIDs
  - check the venue's page limit and template version
  - confirm the claims flagged in `documents/SARA_QA_Report.md`
