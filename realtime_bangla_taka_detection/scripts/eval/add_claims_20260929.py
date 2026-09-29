"""Add the 2026-09-29 results to CLAIM_REGISTRY.json, each with a json_key so the value is checked.

Values are read from the artifacts, never typed in. Existing claim_ids are replaced, not duplicated.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
REG = WORK / "paper_evidence" / "CLAIM_REGISTRY.json"
R = ROOT / "results"


def get(path: Path, key: list):
    node = json.loads(path.read_text(encoding="utf-8"))
    for k in key:
        node = node[k]
    return node


def claim(cid, text, path: Path, key: list, dataset, split, metric, seed="42", sample="physical_note", script=""):
    return {"claim_id": cid, "claim": text, "value": get(path, key), "json_key": key,
            "source_artifact": str(path.relative_to(WORK)), "dataset": dataset, "split": split,
            "sample_definition": sample, "training_seed": seed, "evaluation_seed": seed, "metric": metric,
            "status": "VERIFIED", "script": script}


def main() -> None:
    new = []
    arms = {"fixed": "fixed 6-view, per-count BN", "fixed_shbn": "fixed 6-view, shared BN",
            "prefix_shbn": "prefix, shared BN"}
    for arm, desc in arms.items():
        for s in (42, 43, 44):
            for k in (1, 6):
                p = R / f"same_arch/{arm}/s2/seed{s}/test_views/{k}view/test_metrics.json"
                new.append(claim(f"C_SAMEARCH_{arm.upper()}_S{s}_K{k}", f"Same network, {desc}, seed {s}, {k}-view test accuracy",
                                 p, ["accuracy"], "JaalTaka", "test", "accuracy", str(s), script="scripts/train/run_same_arch.py"))
    pol = R / "jaal_whole" / "policy.json"
    for key, text in ((["jaaltaka_test_real_views", "tau"], "Safe jaal policy threshold (max VAL counterfeit score, S4)"),
                      (["jaaltaka_test_real_views", "counterfeit_passed"], "Safe jaal policy: JaalTaka TEST counterfeit notes passed as likely genuine (of 88)"),
                      (["jaaltaka_test_real_views", "genuine_passed"], "Safe jaal policy: JaalTaka TEST genuine notes confirmed (of 120)")):
        new.append(claim("C_JAAL_SAFE_" + key[-1].upper(), text, pol, key, "JaalTaka", "test", "count", script="scripts/eval/jaal_policy.py"))
    app = R / "jaal_whole" / "app_check.json"
    for key, cid, text, ds in ((["summary", "spoken_jaal"], "C_JAAL_APP_SPOKEN_COUNTERFEIT", "App code path: times 'counterfeit' was spoken on 1,889 whole-note photos", "whole-note photos"),
                               (["summary", "sets", "cf/counterfeit", "likely_genuine"], "C_JAAL_APP_CF_PASSED", "App code path: counterfeit originals said likely genuine (of 20 checked)", "Counterfeit Currency Image Dataset"),
                               (["summary", "sets", "bm/genuine", "likely_genuine"], "C_JAAL_APP_BM_CONFIRMED", "App code path: Bangla Money genuine confirmed (of 277 checked)", "Bangla Money"),
                               (["summary", "sets", "bt/genuine", "likely_genuine"], "C_JAAL_APP_BT_CONFIRMED", "App code path: BanglaTaka genuine confirmed (of 296 checked)", "BanglaTaka")):
        new.append(claim(cid, text, app, key, ds, "external", "count", sample="image", script="scripts/eval/eval_jaal_safe_app.py"))
    summ = R / "jaal_whole" / "summary.json"
    for c in ("S0", "S4", "R2", "B"):
        new.append(claim(f"C_JAAL_WHOLE_AUC_{c}", f"Whole-note ROC-AUC of checker {c} on counterfeit-set originals", summ,
                         ["auc_cf_originals", c], "Counterfeit Currency Image Dataset", "external", "roc_auc", sample="image",
                         script="scripts/eval/jaal_whole_note.py"))
    new.append(claim("C_JAAL_META_SHORTCUT", "Metadata-only classifier: share of counterfeit images it misses (leave one group out)", summ,
                     ["sets", "cf/counterfeit", "META", "wrong_rate"], "Counterfeit Currency Image Dataset", "external", "miss_rate",
                     sample="image", script="scripts/eval/jaal_whole_note.py"))
    new.append(claim("C_JAAL_B_BT_FALSE_COUNTERFEIT", "Approach B (trained on the counterfeit set): BanglaTaka genuine called counterfeit", summ,
                     ["sets", "bt/genuine", "B", "wrong_rate"], "BanglaTaka", "external", "error_rate", sample="image",
                     script="scripts/eval/jaal_whole_note.py"))
    rej = R / "safety" / "rejection_seed42.json"
    gate = R / "safety" / "quality_gate_seed42.json"
    for k in ("1", "6"):
        for cond in ("clean", "low_light_0.35", "low_light_0.2", "brightness_2.2"):
            tag = cond.upper().replace(".", "P")
            new.append(claim(f"C_SAFETY_REJ_K{k}_{tag}", f"Confidence rejection, {k} view, {cond}: wrong verdicts of 208", rej,
                             ["views", k, cond, "wrong_verdicts"], "JaalTaka", "test", "count", script="scripts/eval/eval_safety_rejection.py"))
            new.append(claim(f"C_SAFETY_GATE_K{k}_{tag}", f"Quality gate + confidence, {k} view, {cond}: wrong verdicts of 208", gate,
                             ["views", k, cond, "gate_and_confidence", "wrong_verdicts"], "JaalTaka", "test", "count",
                             script="scripts/eval/eval_quality_gate.py"))
        new.append(claim(f"C_SAFETY_GATE_K{k}_CLEAN_ANSWERED", f"Quality gate + confidence, {k} view, clean: notes answered of 208", gate,
                         ["views", k, "clean", "gate_and_confidence", "answered"], "JaalTaka", "test", "count",
                         script="scripts/eval/eval_quality_gate.py"))
    mv = R / "mvpn" / "vcds.json"
    if mv.is_file():
        for key in ("concat/fixed", "concat/prefix", "attn/fixed", "attn/prefix"):
            for k in ("1", "6"):
                new.append(claim(f"C_MVPN_{key.replace('/', '_').upper()}_K{k}", f"MVP-N {key} head, {k}-view test accuracy, mean of 3 seeds",
                                 mv, ["summary", key, k, "mean"], "MVP-N", "test", "accuracy", seed="42,43,44", sample="object_set",
                                 script="scripts/eval/mvpn_vcds.py"))
        ref = R / "mvpn" / "reference.json"
        for k in ("1", "6"):
            new.append(claim(f"C_MVPN_REFERENCE_K{k}", f"MVP-N per-view reference, {k}-view test accuracy", ref, ["test", k],
                             "MVP-N", "test", "accuracy", seed="deterministic", sample="object_set", script="scripts/eval/mvpn_reference.py"))
    ss = R / "jaal_whole" / "serial_split.json"
    if ss.is_file():
        new.append(claim("C_JAAL_SERIAL_LOOKUP_ACC", "Serial-lookup rule, JaalTaka test accuracy (205 reconstructed notes)", ss,
                         ["serial_lookup_test_accuracy"], "JaalTaka", "test", "accuracy", script="scripts/eval/jaaltaka_serial_audit.py"))
        new.append(claim("C_JAAL_SERIAL_UNSEEN", "JaalTaka test counterfeits with a serial unseen (or unread) in TRAIN counterfeits", ss,
                         ["unseen_or_unread"], "JaalTaka", "test", "count", script="scripts/eval/jaaltaka_serial_audit.py"))
        sa = R / "jaal_whole" / "serial_audit.json"
        new.append(claim("C_JAAL_SERIAL_500CF_TOP", "Counterfeit 500 BDT JaalTaka notes carrying the most common serial", sa,
                         ["summary", "500/counterfeit", "top3", 0, 1], "JaalTaka", "all", "count", script="scripts/eval/jaaltaka_serial_audit.py"))
    sm = R / "jaal_whole" / "serial_mask_test.json"
    if sm.is_file():
        for which in ("unmasked", "serial_masked"):
            new.append(claim(f"C_JAAL_SERIALMASK_K1_{which.upper()}", f"PRMVT 1-view test accuracy, serial region {which}", sm,
                             ["by_k", "1", which, "accuracy"], "JaalTaka", "test", "accuracy", script="scripts/eval/serial_mask_test.py"))
    hs = R / "watermark" / "hybrid_serial_split.json"
    if hs.is_file():
        for k in ("k1", "k6"):
            for m in ("network", "hybrid"):
                new.append(claim(f"C_SERIALSPLIT_{m.upper()}_{k.upper()}", f"Serial-disjoint split, prefix network{' + watermark' if m == 'hybrid' else ''}, {k} test accuracy",
                                 hs, [k, m, "accuracy"], "JaalTaka (serial-disjoint)", "test", "accuracy", script="scripts/eval/hybrid_serial_split.py"))
            new.append(claim(f"C_SERIALSPLIT_HYBRID_{k.upper()}_FCR", f"Serial-disjoint split, hybrid {k}: genuine called counterfeit (of 121)",
                             hs, [k, "hybrid", "false_counterfeit_on_genuine"], "JaalTaka (serial-disjoint)", "test", "count",
                             script="scripts/eval/hybrid_serial_split.py"))
    hss = R / "watermark" / "hybrid_serial_split_seeds.json"
    if hss.is_file():
        for key in ("k1/network", "k1/hybrid", "k6/network", "k6/hybrid"):
            new.append(claim(f"C_SERIALSPLIT3_{key.replace('/', '_').upper()}", f"Serial-disjoint split, {key} accuracy, mean over seeds 42-44",
                             hss, ["summary", key, "mean"], "JaalTaka (serial-disjoint)", "test", "accuracy", seed="42,43,44",
                             script="scripts/eval/hybrid_serial_split.py"))
    mb = R / "watermark" / "mobilenet.json"
    if mb.is_file():
        new.append(claim("C_WM_MOBILENET_ACC", "Watermark MobileNetV3-Small, unseen-print test accuracy", mb, ["test", "accuracy"],
                         "JaalTaka (serial-disjoint)", "test", "accuracy", script="scripts/train/train_watermark_mobilenet.py"))
        new.append(claim("C_WM_MOBILENET_AUC", "Watermark MobileNetV3-Small, unseen-print test AUC", mb, ["test", "auc"],
                         "JaalTaka (serial-disjoint)", "test", "roc_auc", script="scripts/train/train_watermark_mobilenet.py"))
    hf = R / "watermark" / "hybrid_final_seeds.json"
    if hf.is_file():
        for key in ("k1/hybrid", "k6/hybrid"):
            new.append(claim(f"C_FINAL_HYBRID_{key.replace('/', '_').upper()}", f"Final hybrid (MobileNetV2 watermark, chosen on VAL), unseen prints, {key}, mean of 3 seeds",
                             hf, ["summary", key, "mean"], "JaalTaka (serial-disjoint)", "test", "accuracy", seed="42,43,44",
                             script="scripts/eval/hybrid_v2_serial_split.py"))
    m2 = R / "watermark" / "mobilenetv2.json"
    if m2.is_file():
        new.append(claim("C_WM_MOBILENETV2_ACC", "Watermark MobileNetV2 (chosen on VAL AUC), unseen-print test accuracy", m2, ["test", "accuracy"],
                         "JaalTaka (serial-disjoint)", "test", "accuracy", script="scripts/train/train_watermark_mobilenet.py"))
        new.append(claim("C_WM_MOBILENETV2_INT8_AGREE", "Watermark MobileNetV2 INT8: share of test decisions equal to FP32", m2,
                         ["export", "watermark_mobilenetv2_int8.onnx (static, calibrated)", "same_decision_as_fp32"], "JaalTaka (serial-disjoint)",
                         "test", "agreement", script="scripts/export/quantize_watermark_int8.py"))
    da = R / "jaal_whole" / "domain_adapt.json"
    if da.is_file():
        for mth in ("S4_adabn", "R2_coral"):
            new.append(claim(f"C_DA_{mth.upper()}_AUC", f"Whole-note domain adaptation {mth}: AUC on counterfeit-set originals", da,
                             ["methods", mth, "auc_cf_originals"], "Counterfeit Currency Image Dataset", "external", "roc_auc",
                             sample="image", script="scripts/eval/domain_adapt_whole.py"))
    br = R / "serial_split" / "beat_resnet.json"
    if br.is_file():
        for key in ("k1/FT_RESNET", "k6/FT_RESNET", "k1/FUSION", "k6/FUSION", "k1/ENSEMBLE", "k6/ENSEMBLE", "k1/HYBRID_FINAL", "k6/HYBRID_FINAL"):
            new.append(claim(f"C_BEAT_{key.replace('/', '_').upper()}", f"Unseen prints, {key} accuracy, mean of 3 seeds", br,
                             ["summary", key, "mean"], "JaalTaka (serial-disjoint)", "test", "accuracy", seed="42,43,44",
                             script="scripts/eval/beat_resnet.py"))
    pw = R / "user_study" / "power.json"
    if pw.is_file():
        new.append(claim("C_USERSTUDY_MIN_DZ_105", "Smallest paired effect dz detectable with 80 % power, n = 105", pw, ["105", "min_dz_80"],
                         "user study design", "design", "effect_size", seed="n/a", sample="participant", script="scripts/eval/user_study_power.py"))
    hj = R / "watermark" / "hybrid.json"
    if hj.is_file():
        for key in ("W_deep", "S_list", "HYBRID_k1", "HYBRID_k6"):
            new.append(claim(f"C_WM_{key.upper()}_ACC", f"Seed-42 split, {key} test accuracy", hj, [key, "test", "accuracy"],
                             "JaalTaka", "test", "accuracy", script="scripts/eval/watermark_hybrid.py"))
        new.append(claim("C_WM_HAND_AUC", "Seed-42 split, hand-crafted watermark test AUC", hj, ["W_hand", "test", "auc"],
                         "JaalTaka", "test", "roc_auc", script="scripts/eval/watermark_hybrid.py"))
    ssp = R / "watermark" / "serial_split_eval.json"
    if ssp.is_file():
        for k in ("k1", "k6"):
            new.append(claim(f"C_SERIALSPLIT_PROBE_{k.upper()}", f"Serial-disjoint split, ResNet-50 probe {k} accuracy", ssp, ["PROBE", k, "accuracy"],
                             "JaalTaka (serial-disjoint)", "test", "accuracy", seed="deterministic", script="scripts/eval/watermark_serial_split.py"))
    cvp = R / "sota" / "cv_probe.json"
    if cvp.is_file():
        for d in ("note", "serial"):
            for k in ("1", "6"):
                new.append(claim(f"C_CV_{d.upper()}_K{k}", f"5-fold CV over 1,390 notes ({d} folds), ResNet-50 probe {k}-view pooled accuracy",
                                 cvp, [d, "pooled_accuracy_all_1390_notes", k], "JaalTaka", "cross-validation", "accuracy",
                                 seed="deterministic", script="scripts/eval/cv_probe.py"))
    ff = R / "mvpn" / "fusion_fix.json"
    if ff.is_file():
        for key in ("meanpool/fixed", "meanpool/prefix"):
            for k in ("1", "6"):
                new.append(claim(f"C_MVPN_{key.replace('/', '_').upper()}_K{k}", f"MVP-N {key} head, {k}-view accuracy, 3 seeds", ff,
                                 [key, k, "mean"], "MVP-N", "test", "accuracy", seed="42,43,44", sample="object_set",
                                 script="scripts/eval/mvpn_fusion_fix.py"))
    aa = R / "jaal_whole" / "approach_a" / "eval.json"
    if aa.is_file():
        new.append(claim("C_JAAL_A_REAL_CF_PASSED", "Approach A: real counterfeit originals said likely genuine (of 20)", aa,
                         ["real_whole_note", "A", "cf/counterfeit", "said_likely_genuine"], "Counterfeit Currency Image Dataset",
                         "external", "count", sample="image", script="scripts/eval/eval_approach_a.py"))
        new.append(claim("C_JAAL_A_REAL_AUC", "Approach A: real whole-note AUC on counterfeit-set originals", aa,
                         ["real_whole_note", "A", "auc_cf_originals"], "Counterfeit Currency Image Dataset", "external", "roc_auc",
                         sample="image", script="scripts/eval/eval_approach_a.py"))
        new.append(claim("C_JAAL_A_SYNTH_TEST_ACC", "Approach A: synthetic whole-note test accuracy", aa,
                         ["synthetic_test_A", "accuracy_0.5"], "JaalTaka (reconstructed)", "test", "accuracy",
                         script="scripts/eval/eval_approach_a.py"))
    probes = R / "sota" / "backbone_probes.json"
    if probes.is_file():
        for name in json.loads(probes.read_text(encoding="utf-8")):
            for k in ("1", "6"):
                new.append(claim(f"C_PROBE_{name.upper()}_K{k}", f"Frozen {name} linear probe, {k}-view test accuracy", probes,
                                 [name, "test_acc", k], "JaalTaka", "test", "accuracy", seed="deterministic",
                                 script="scripts/eval/eval_backbone_probes.py"))
    data = json.loads(REG.read_text(encoding="utf-8"))
    claims = data["claims"] if isinstance(data, dict) else data
    ids = {c["claim_id"] for c in new}
    kept = [c for c in claims if c.get("claim_id") not in ids]
    kept += new
    if isinstance(data, dict):
        data["claims"] = kept
    else:
        data = kept
    REG.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print("added/updated", len(new), "total", len(kept))


if __name__ == "__main__":
    main()
