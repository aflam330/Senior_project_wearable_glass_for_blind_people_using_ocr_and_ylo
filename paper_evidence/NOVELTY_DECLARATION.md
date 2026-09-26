# Novelty declaration

Labels describe how the implementation relates to published ideas. They are not performance claims. Experimental numbers are only in `NOVEL_ALGORITHMS_COMPARISON.md` and the per-algorithm results files.

| algorithm | label | why |
|---|---|---|
| OGPD | EXTENDS | Learning using privileged information (Vapnik and Vashist, 2009) and generalized distillation (Lopez-Paz et al., 2015, arXiv:1511.03643) already train with extra inputs that are absent at test time. Multi-view privileged distillation is also published (Lambert et al. style multi-view LUPI, arXiv:1903.03694; MPIRL). This run distills a shortest label-consistent prefix into a stop head. It does not claim a new distillation theory. |
| VCIE | ADAPTED | The Set Transformer (Lee et al., ICML 2019, arXiv:1810.00825) and Deep Sets (Zaheer et al., 2017) are permutation-invariant set encoders. AttSets (Yang et al., 2018) already pools multi-view features with attention. This run uses two self-attention blocks and attention pooling with no view-index embedding. |
| APC | EXTENDS | Curriculum learning (Bengio et al., 2009) and self-paced learning (Kumar, Packer, and Koller, 2010) are prior work. Multi-view self-paced learning (Xu, Tao, and Xu, IJCAI 2015) and self-paced multi-view co-training (JMLR 2020) already treat views as a difficulty axis. The schedule here increases prefix length. |
| SFAQ | ADAPTED | No-reference image quality (BRISQUE, Mittal et al., 2012; NIQE) is prior work. Security-feature detection is NOT_MEASURED: JaalTaka has no hologram, thread, watermark, microprint, or serial labels. The trained head uses generic image statistics only. |
| IGCR | EXTENDS | InfoNCE (van den Oord et al., 2018) is prior work. The added term treats two views of the same note as a positive pair. It is not a labeled-entropy information-gain model. |
| UGF | ADAPTED | Gated and uncertainty-weighted fusion is prior work (attention fusion and uncertainty-aware multi-view models). The gate is a two-layer map of uncertainty, quality, and a diversity proxy. |
| NDAL | ADAPTED | Focal loss (Lin et al., 2017) and hard-example reweighting are prior work. The weight is the detached per-note loss, normalized inside the batch. |
| SFPL | ADAPTED | FedAvg (McMahan et al., 2017) and EWC (Kirkpatrick et al., 2017) are prior work. This run is SIMULATED: one process, client ids only change the allowed prefix length. FedAvg and FedAvg+EWC arms are NOT_MEASURED. |
| CVS | EXTENDS | Leave-one-out intervention scores are a standard ablation. The head is trained to match the change in the class logit when a view is masked. A causal graph and a do-calculus identification result are NOT_MEASURED. |
| MTPT | ADAPTED | Multi-task learning (Caruana, 1997) is prior work. Measured heads are authenticity and a generic quality proxy. Denomination and emotion are NOT_MEASURED. |

The prefix-robust Q-DUIG result remains the strongest measured accuracy result in this repository (see `FINAL_RESULTS.md`). These ten runs do not replace that result. No algorithm is declared first or state of the art.

Searches used to place the labels: oracle distillation and privileged information for view selection; set transformer and permutation-invariant multi-view encoders; curriculum and self-paced multi-view selection; security-feature and banknote image quality; contrastive information gain and mutual information for views; uncertainty-gated and Bayesian multi-view fusion; difficulty-aware and focal losses; federated learning with heterogeneous views; causal and interventional view selection; multi-task prefix transformers.
## v2 additions

PRAVT, CRIS, SAVS, MAVT, and VAT were trained after the v1 runs. PRAVT adds a reversed-order loss. CRIS adds a KL bottleneck on the fused vector. SAVS is sharpness-aware minimization on the same classifier. MAVT adds an in-batch prototype loss. VAT samples prefix length with probability proportional to 1/k and predicts that length. None of these are claimed as first or state of the art. VCIE, SFPL, and MTPT v2 use a residual around the set encoder because v1 predicted only the majority class.

## A* positioning (searched, not a systematic review of ten databases)

Web-index queries returned papers from CVF (ICCV), arXiv, OpenReview, AAAI, IJCAI, MDPI, PMC, and Research Square. Scopus, IEEE Xplore, ACM Digital Library, and Web of Science were not queried as separate authenticated databases. A paper that those indexes hold and the web index missed would change a label. No method below is called first.

| algorithm | label | prior work | difference |
|---|---|---|---|
| PRMVT | EXTENDS | Missing-view training already exists: Robust MVAE drops a random subset of views and reconstructs the rest (OpenReview, RMAE); dual-masked VAEs handle view-missing and noise (IJCAI 2025); RML simulates unusable views (Xu et al., ICCV 2025). | The measured failure is narrower: a classifier trained at a fixed view count of 6 drops to 0.5865 accuracy at 1 view on JaalTaka. The fix is supervised prefix coverage, not a new missing-view theory. |
| NDAL | ADAPTED | Focal loss (Lin et al., ICCV 2017, arXiv:1708.02002) already down-weights easy examples. Class-wise difficulty weighting is also published (arXiv:2207.14499). | This run multiplies focal loss (gamma 1.5) by a clamped per-note weight. It is a variant, not a new loss. |
| PRAVT | ADAPTED | View-level adversarial training is published: RDML (IJCAI 2025, arXiv:2505.04046) and the DAVE preprint (Research Square rs-8025005), which uses PGD on views. | This run adds cross-entropy on `views.flip(1)`. That is an order reversal, not a projected-gradient attack. |
| VAT | EXTENDS | Stochastic view dropout is in RMAE and in the DAVE preprint. | This run samples the prefix length with probability proportional to 1/k and adds a head for that length. The name VCDS is local. The support argument is in `THEORETICAL_ANALYSIS.md`. |
| UGF | ADAPTED | Evidential fusion (Han et al., TMC/ETMC, 2021–2022) and uncertainty-guided fusion (DAVE preprint) already weight views by reliability. | The gate here is a small map on the shared MobileNet+TinyViT features. |
| CRIS | ADAPTED | Multi-view information bottleneck is published (Federici et al., arXiv:2002.07017; AAAI multi-view IB, ojs.aaai.org/17210). | This run adds a KL term of weight 0.01 on a bottleneck of the fused vector. |
| SAVS | ADAPTED | Sharpness-aware minimization (Foret et al., ICLR 2021) is prior work. A later note (arXiv:2405.20439) studies SAM and redundant features. | One SAM step with rho 0.05. The name says view selection; the code is SAM on the classifier. |
| MAVT | ADAPTED | Prototypical networks (Snell, Swersky, and Zemel, 2017) and multi-task learning (Caruana, 1997) are prior work. | In-batch prototype NLL during training. Test uses the classifier only. |
| VCIE | ADAPTED | Deep Sets and the Set Transformer, as in the v1 row. | v2 adds a residual around the set encoder after v1 collapsed to the majority class. |
| MTPT | ADAPTED | Caruana, 1997, as in the v1 row. | v2 uses the same residual set encoder. Denomination and emotion heads were not trained. |
| CVS | EXTENDS | Leave-one-out ablations are standard. Causal multi-view feature selection (CAUSA, arXiv:2509.13763) is a different method. | The loss is KL between the full-view logit and the logit with one view dropped, weight 0.05. No causal graph was fit. |
| OGPD | EXTENDS | Generalized distillation and multi-view LUPI (Lopez-Paz et al., arXiv:1511.03643; arXiv:1903.03694). | v1 is the standing result. v2, with a smaller distillation weight, was worse. See `FINAL_RESULTS.md`. |
