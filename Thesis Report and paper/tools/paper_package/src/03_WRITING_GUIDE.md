# Writing guide for the two drafts

## The two drafts

| | IEEE conference (`latex/ieee_conference/`) | Q1 journal (`latex/q1_journal/`) |
|---|---|---|
| Template | IEEEtran, `conference` option, two columns | elsarticle (Elsevier), `preprint,12pt`, single column, numbered references |
| Length | About 6 pages | About 24 pages |
| Focus | The system and its three strongest findings: prefix training (view-count collapse), the serial shortcut with the print-disjoint split, and the safe verdict. DMWR, OCR and the Pi are given briefly | Everything in the conference paper, plus the full methods (compositing, hand-held fine-tuning, DMWR, UV/ENFR, OCR pipeline, voice), every result table, negative results, the audit and corrections, threats to validity, and the declarations Elsevier journals ask for |
| Title | SARA: An Offline Bangla Assistive Smart Glass with Never-Accusing Banknote Screening on a Raspberry Pi 5 | Seeing Money, Text and Surroundings Offline: A Bangla Assistive Smart Glass with Prefix-Robust Multi-View Counterfeit Screening, Physics-Based Watermark Residuals and a Never-Accusing Verdict |

The elsarticle class is the template of many Elsevier Q1 journals in this area. Examples: *Engineering Applications of Artificial Intelligence*, *Expert Systems with Applications*, *Pattern Recognition Letters* (a letter), *Computers in Biology and Medicine* (if framed clinically). For an IEEE journal (*IEEE Access*, *IEEE Transactions on Human-Machine Systems*) the journal text can move to IEEEtran `journal` mode with little change. For Springer journals use `svjour3` / `sn-jnl`.

Which venue accepts the paper, and how likely that is, cannot be known in advance. No probability is given here.

## Fill in before submission

1. **Authors:** e-mail addresses, ORCIDs and the corresponding author (placeholders are marked `% TODO`).
2. **Venue rules:**
   - the venue's page limit, anonymisation (double-blind venues: remove names, the repository links and the self-identifying acknowledgement)
   - keywords list (the journal asks for 4–7)
   - the highlights file (Elsevier: 3–5 bullets of at most 85 characters; a draft is in the journal `main.tex` as a comment)
3. **Data and code availability:** confirm that both GitHub repositories are public at submission. Consider archiving them on Zenodo for a DOI.
4. **Ethics statement for the participant trials:** if any user result is added later, it needs the IRB / ethics approval number. No user result is reported in these drafts.
5. **Funding:** none stated; fill in if any.
6. **Claims to confirm:** check the items flagged in `documents/SARA_QA_Report.md`, especially OrCam's current Bangla support and the device-comparison rows.

## Rules the drafts follow (keep them when editing)

- **Sources:** every number comes from `02_RESULTS_MASTER_TABLE.md`, never from an older evidence note without checking its status.
- **Conditions:** synthetic, independent-photo, hand-held and informal results are kept apart. Each number says which kind it is.
- **No state-of-the-art claim on JaalTaka:** frozen probes reach 98–99%. The hybrid and the fine-tuned ResNet-50 are a tie.
- **Counterfeits:** the glass is a screening aid. It never says "counterfeit", and no claim of real-world counterfeit detection is made.
- **UV:** the UV check is a prototype; its fibre-free side is simulated.
- **Voice:** voice results use synthetic voices on a laptop.
- **Limitations stay in:** there are no camera-through-the-glass results and no user study.
- **Uncertainty:** report Wilson intervals, the seed spread and the print-level uncertainty (±12 pt).

## Checklist

- [ ] Author details, ORCIDs, corresponding author
- [ ] Page limit met (conference: compress Tables III–V or move DMWR to a short paragraph if needed)
- [ ] Figures at 300 dpi or vector. The PNG charts are 1,280–1,920 px wide. For print, use the vector PDFs in `figures/publication/` where they exist, or regenerate the charts with `scripts/eval/make_publication_figures.py` in the research repository
- [ ] All references complete. BibTeX reports gaps that exist in the source bibliography; they were left empty rather than guessed:
  - **No year:** Vosk, Piper, eSpeak NG, EasyOCR, ONNX Runtime, PaddleOCR, SymSpell, HarfBuzz, Silero VAD, OrCam, Envision, Seeing AI, BanglaTaka (Mendeley). For software, add the version and year you used.
  - **No authors:** `readingaid_rpi2023`, `thirdeye2019`, `banglaocr2021`, `at_review2025`.
  - **No pages:** several conference entries (COCO, RAF-DB, MobileNetV3, EfficientNetV2, ICDAR 2019 MLT, MVP-N and others).
  - Check each against its DOI or publisher page.
- [ ] Data and code availability statement with final links or DOIs
- [ ] Ethics statement (no human-subject results reported)
- [ ] Cover letter that states the contributions and the limitations honestly
- [ ] Optional: a second reader checks every number against the master table
