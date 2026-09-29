# Literature novelty matrix

This is a positioning document, not a claim of priority. No method below is described as “first” unless a systematic review (not performed here) supports that wording.

| Paper | Year | Task | Dataset | Method | View selection | Uncertainty | Quality | Diversity | Info gain | Edge | Limitation vs this work |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Han et al., Trusted Multi-view Classification (TMC / ETMC) | 2021–2022 | Multi-view classification | Benchmark MVC sets | Dirichlet evidence + Dempster–Shafer fusion | All views used | Yes (evidential) | Implicit via evidence | No explicit geometry | No | No | Fuses a fixed view set; no sequential acquisition or acquisition cost |
| Wang et al., SeqMvRL | 2025 | Sequential multi-view representation | MVC / sentiment-style views | RL next-view selector + pairwise integrator | RL reward on cluster compactness | No auth uncertainty | No banknote quality head | Conflict/redundancy via reward | No explicit ΔH | No | Reward is clustering, not authentication reliability or assistive cost |
| G+D / multi-camera multi-spectral document inspection | 2016–2019 | Banknote / document authentication | Proprietary notes | Multi-spectral capture + print-method detectors | Hardware-fixed cameras | Limited | Optical quality of capture | Multi-spectral, not embedding geometry | No | Embedded inspection systems | Parallel multi-sensor, not sequential information-gain policy on a monocular phone/Pi camera |
| Active view selection with neural uncertainty maps (PUN) | 2025 | 3D reconstruction | Object-centric views | Uncertainty maps, drop low-uncertainty views | Uncertainty + angular redundancy | Yes (reconstruction) | No authentication quality | Angular / low-uncertainty filter | Implicit | No | Different task (NeRF/PSNR), not binary note authenticity |
| Standard temperature scaling (Guo et al.) | 2017 | Calibration | ImageNet etc. | Single T on logits | n/a | Confidence calibration | No | No | No | n/a | HER adds quality- and entropy-conditioned residuals; still compared against T |
| EWC (Kirkpatrick et al.) | 2017 | Continual learning | Permuted / sequential tasks | Fisher penalty | n/a | No | No | No | No | n/a | QWER-VPC adds quality-weighted prototypes; compared, not replaced as a claim of dominance |
| FedAvg (McMahan et al.) | 2017 | Federated averaging | Language / vision | n_k-weighted mean | n/a | No | No | No | No | Comms | QDW-Fed reweights by usable quality × embedding volume; **simulated clients only** |
| Prior CAMVA in this repo (quality-attention) | 2026 | JaalTaka authentication | JaalTaka 1390 notes / 6 views | Scalar quality + attention; confidence stop | Confidence / quality order | Softmax conf | Scalar usability | None | None | Intended | Measured worse than CNN+ViT on 1–5 views; artifacts kept in `results/camva/` |

## How Q-DUIG-CAMVA is modified relative to the nearest ideas

1. **Quality (RSQA)** — not a scalar gate or hand-crafted blur threshold. A 12-factor descriptor is an *input* to a learned bilinear residual attention. Thresholds, if any, are fit on **validation** usability vs leave-one-out agreement.
2. **Diversity (CVR)** — not cosine. Residual energy after projection onto the span of selected embeddings × incremental log-det volume.
3. **Uncertainty / calibration (HER)** — temperature plus quality- and entropy-conditioned residuals, fit on val NLL. Temperature, entropy mapping, confidence, and MC dropout are retained as controls.
4. **Information gain (PCR-IG)** — supervised surrogate of binary-entropy reduction from prefix→prefix+1 rollouts on **train/val only**.
5. **Policy (CRIQP)** — STOP iff predicted utility ≤ 0: `U = IG · usable · (1 − redundancy) − λ ΔC`. Cost weights chosen on the **validation Pareto** frontier.
6. **Fusion (HGEF)** — hypernetwork gate + residual shift from auxiliary evidence, not mean or concat-MLP alone.
7. **Continual (QWER-VPC)** and **federated (QDW-Fed)** are additional methods compared against FT/replay/EWC and FedAvg/FedAvg+EWC. Federated results are labeled **SIMULATED**.

## Topics searched

Multi-view authentication; sequential visual acquisition; active perception; information-gain image acquisition; uncertainty-aware classification; quality-aware fusion; redundancy-aware fusion; banknote authentication; assistive currency recognition; edge visual authentication.

## What is not claimed

- First sequential multi-view method in computer vision.
- First banknote authenticator.
- First quality-aware or uncertainty-aware fusion.
- Automatic superiority over the CNN+ViT baseline (the previous CAMVA run lost on 1–5 views).
