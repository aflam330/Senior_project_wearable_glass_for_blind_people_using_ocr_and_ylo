# Suggested Paper Outlines

Use these structures when drafting manuscripts from `PAPER1_Currency_Detection.md` and `PAPER2_Smart_Glass_System.md`.

---

## Paper 1 — Currency Detection (CV / ML venue)

**Length target:** 6–10 pages (conference) or short journal article  
**Tone:** methods + experiments heavy  

### Outline

1. **Title & authors**
2. **Abstract** (copy from Paper 1 doc; polish once)
3. **Keywords**
4. **Introduction**
   - Need for real-time BDT localisation
   - Classification vs detection gap
   - Synthetic-to-real motivation
   - Contributions list (4–5 bullets)
5. **Related work**
   - Pull from `RELATED_WORK_BIBLIOGRAPHY.md` (currency-tagged + NSTU-BDTAKA)
   - BDT classification systems [4,5,7]
   - BDT / multi-currency detection [1,2]
   - Domain randomisation / synthetic detection [3]
   - Datasets: BanglaTaka [6], COCO [8]
6. **Methodology**
   - 6.1 Problem formulation (detection, 9 classes)
   - 6.2 Source data (BanglaTaka + COCO)
   - 6.3 Copy-paste compositing & domain randomisation
   - 6.4 YOLOv8s architecture & training setup
   - 6.5 Real-time application (webcam, brighten, TTS)
7. **Experiments**
   - Dataset splits & protocol
   - Overall metrics table
   - Per-class table
   - Confusion matrix figure
   - Training curves
   - Comparison with [1][2]
   - Inference speed
   - Qualitative webcam results
8. **Discussion**
   - Why compositing works
   - Synthetic-test caveat
   - Failure modes
9. **Limitations & future work**
10. **Conclusion**
11. **References**
12. **Appendix (optional):** hyperparameter full table, script list

### Must-include figures
1. System / pipeline diagram (`architecture.png` or redraw)  
2. Compositing examples (`compositing.png`)  
3. Training curves  
4. Confusion matrix  
5. Live webcam qualitative grid  

### Must-include tables
1. Source note counts per denomination  
2. Synthetic split sizes  
3. Training hyperparameters  
4. Overall + per-class metrics  
5. Comparison with prior work  

### Honesty checklist
- [ ] State test set is held-out **synthetic** composites  
- [ ] Do not claim counterfeit detection  
- [ ] Report hardware for FPS claim (RTX 4060 Laptop)  
- [ ] Cite BanglaTaka + COCO licenses/terms as required by venue  

---

## Paper 2 — Smart Glass System (assistive tech / embedded AI venue)

**Length target:** 8–12 pages system paper  
**Tone:** design + architecture + accessibility UX; add whatever evaluation you can run  

### Outline

1. **Title & authors**
2. **Abstract** (from Paper 2 doc)
3. **Keywords**
4. **Introduction**
   - Accessibility gap in Bangladesh / Bangla
   - Offline + low-cost motivation
   - System overview paragraph
   - Contributions
5. **Related work**
   - Pull from `RELATED_WORK_BIBLIOGRAPHY.md` (smart-glass / OCR / object / TTS tags)
   - Especially sheet #2 (MDPI smart glass, 77 cites), #55, #49, RPi aids (#12/#14/#40)
   - Bangla OCR (#23) & TTS papers
   - Currency recognition for VI (cite Paper 1 if published/submitted)
6. **System design**
   - 6.1 Design requirements (offline, buttons, Bangla-first)
   - 6.2 Hardware platform & wiring
   - 6.3 Software architecture & threading
   - 6.4 Mode design (OCR / Object / Currency)
   - 6.5 TTS & language segmentation
   - 6.6 Deployment (install + systemd)
7. **Implementation details**
   - OCR preprocess choices (CLAHE, no binarize, capture/read split)
   - Object cooldown UX
   - Currency two-stage pipeline
   - Windows harness for development
8. **Evaluation** *(fill with your experiments)*
   - Latency per mode on Pi 5
   - Functional tests / demo scenarios
   - Optional user study
   - Optional currency accuracy (esp. if YOLO integrated)
9. **Discussion**
   - Design trade-offs (CPU, EasyOCR size, button UX)
   - Comparison vs cloud systems
10. **Limitations & future work**
11. **Conclusion**
12. **References**
13. **Appendix:** BOM, GPIO pin table, install steps

### Must-include figures
1. Hardware photo + block diagram  
2. Software architecture diagram  
3. Mode-state / button interaction diagram  
4. Example OCR / object / currency usage photos  
5. (Optional) Latency bar chart on Pi 5  

### Must-include tables
1. Hardware BOM  
2. Button map  
3. Models & thresholds  
4. Mode behaviour summary  
5. Evaluation results (once measured)  

### Cross-paper strategy
| If… | Then… |
|-----|-------|
| Paper 1 accepted/submitted first | Paper 2 cites it as the currency backend / planned backend |
| Same venue / simultaneous | Share authors; differentiate novelty (data+detector vs wearable system) |
| YOLO not integrated yet | Paper 2 describes HSV+MobileNet honestly; future work = Paper 1 model |

---

## One-paragraph “elevator” for each paper

**Paper 1:** We turn a clean BanglaTaka classification dataset into a large detection dataset by pasting notes onto COCO backgrounds with heavy domain randomisation, then fine-tune YOLOv8s to localise nine BDT denominations in real time with near-perfect synthetic-test mAP and a webcam TTS demo.

**Paper 2:** We build a fully offline Raspberry Pi 5 smart glass for Bangla-speaking blind users that cycles among OCR, object announcement, and BDT currency recognition using physical buttons and neural/offline speech synthesis — without requiring cloud services.
