# Literature search (2026-09-30)

**Method and scope.** Web searches run from this session, 7 queries over two rounds. Only claims visible in the search results are recorded. **Scopus, IEEE Xplore, ACM DL and Web of Science were not searched directly** (no access from here). This is a targeted search, not a systematic review. The authors' own review (`docs/RELATED_WORK_BIBLIOGRAPHY.md`) should be merged in before submission.

## 1. Bangladeshi Taka counterfeit detection

| Work | What it reports | Relation to this project |
|---|---|---|
| JaalTaka (Data in Brief, 2025) | 1,390 notes (802 genuine, 588 counterfeit), six region images per note, counterfeits from the Rapid Action Battalion; the "first publicly available" Bangladeshi counterfeit dataset | Our benchmark. We show its counterfeits share a few printed serials, so random splits are not print-disjoint ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12774690/), [Mendeley](https://data.mendeley.com/datasets/2m7wk5cy4c/2)) |
| CNN counterfeit detection for BDT | Modified AlexNet + SVM; one model 85.4 % → 90.03 % after retraining | Single-split accuracies; no unseen-print evaluation ([academia.edu](https://www.academia.edu/116590712/Enhanced_Counterfeit_Detection_of_Bangladesh_Currency_through_Convolutional_Neural_Networks_A_Deep_Learning_Approach), [ResearchGate](https://www.researchgate.net/publication/364609677_A_Deep_Learning_Approach_for_Detecting_Bangladeshi_Counterfeit_Currency)) |
| Dual-stream MobileNet + EfficientNet for BDT recognition (arXiv 2602.07015) | Real-time Bangladeshi currency *recognition* | Denomination, not counterfeit ([arXiv](https://arxiv.org/pdf/2602.07015)) |

## 2. Watermark and transmitted-light counterfeit detection

| Work | What it reports | Relation |
|---|---|---|
| Multinational banknote type and fitness classification with visible reflection plus IR-transmission images | Aligned visible-light and IR-transmission images fed to a CNN | Uses transmitted light with dedicated sensors; ours uses a phone photo held against light ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6412798/)) |
| Fake-banknote detection for visually impaired people from smartphone visible-light images | Smartphone images; aimed at blind users | Closest assistive framing; no watermark-window model ([ResearchGate](https://www.researchgate.net/publication/340305632_Deep_Learning-Based_Fake-Banknote_Detection_for_the_Visually_Impaired_People_Using_Visible-Light_Images_Captured_by_Smartphone_Cameras)) |
| Fake banknote recognition / multinational fake detection (MDPI) | CNN counterfeit classifiers | Single-split evaluation ([Applied Sciences](https://doi.org/10.3390/app11031281), [Mathematics](https://www.mdpi.com/2227-7390/10/9/1616)) |

**Gap.** No work was found that isolates the watermark window as a separate, registered input and measures its value on counterfeit prints unseen in training.

## 3. Serial numbers

- Serial recognition is standard in counters, sorters and ATMs, and is used to trace notes and detect counterfeits by database lookup.
- Counterfeiters often duplicate existing serials ([PMG](https://www.pmgnotes.com/news/article/8686/Counterfeit-Detection-Duplicated-Serial-Number/), [Ma & Yan 2021](https://cerv.aut.ac.nz/wp-content/uploads/2021/12/Ma-Yan2021_Article_BanknoteSerialNumberRecognitio.pdf), [PMC 9699018](https://ncbi.nlm.nih.gov/pmc/articles/PMC9699018)).
- **Our result agrees:** a serial list catches known bundles (1 of 4 prints on real photos, 0 / 450 genuine false alarms) but never a new print (Proposition 12).

## 4. Leakage and group-disjoint evaluation

- Improper splitting with near-duplicates inflates test accuracy by 5–30 % in reported studies; group-disjoint splitting is recommended ([Scientific Data](https://www.nature.com/articles/s41597-022-01618-6), [arXiv 2401.13796](https://arxiv.org/pdf/2401.13796)).
- **Our print-sharing finding is a new instance of this in counterfeit detection.** The note-disjoint vs print-disjoint gap here is about 8–9 points for one network.

## 5. Multi-view learning with missing views

- View dropout is a known way to gain robustness to missing views, and it can hurt full-view accuracy ([arXiv 2501.01132](https://arxiv.org/pdf/2501.01132), [arXiv 2303.17117](https://arxiv.org/html/2303.17117v4)).
- **Our contribution is not the technique.** It is the analysis (non-identifiability), the fusion-dependence finding (pooling vs concatenation), and the counterfeit application.

## 6. Assistive currency recognition

- Euro-banknote sunglasses on a Raspberry Pi (84 % detection, 97.5 % value).
- YOLOv8–v10 for Egyptian notes.
- BankNote-Net.
- Sri Lankan notes.
- Consumer readers (iBill, EyeNote).

Sources: [PMC 5298757](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5298757/), [Scientific Reports](https://www.nature.com/articles/s41598-025-20646-x), [arXiv 2204.03738](https://arxiv.org/pdf/2204.03738), [arXiv 2502.14267](https://arxiv.org/pdf/2502.14267), [NICOA](https://www.nicoa.org/currency-reader-free-to-people-who-are-visually-impaired/).

**Gap.** None found that pairs a counterfeit check with a stated safety policy (never asserting "counterfeit") and a finite-sample guarantee.
