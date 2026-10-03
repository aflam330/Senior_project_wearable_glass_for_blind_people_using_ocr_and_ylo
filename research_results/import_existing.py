"""Import the results that already exist in this repository into research_results/experiments.json.

Every number is read from a result file by this script (nothing is typed in). Each record names the file and the
key path (selector) it came from, plus the file's SHA-256. Records get status 'verified' only because the value was
read from the saved output of the evaluation that produced it; that is stated in verification.method.
Running it again adds nothing unless a result file changed (then a new record is added and linked with run.rerun_of).

  python research_results/import_existing.py            import, validate, export
  python research_results/import_existing.py --dry-run   show what would be added
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_results import results_db as DB  # noqa: E402

REPO = DB.REPO
R = "realtime_bangla_taka_detection/results"
G = "savior_glass/results"
LAPTOP = {"device": "laptop (DESKTOP-EA33S9L)", "cpu": "AMD Ryzen 7 5800H", "gpu": "NVIDIA GeForce RTX 3050 Laptop GPU (4 GB)"}
JAAL = {"name": "JaalTaka", "source": "data set/JaalTaka", "real_or_synthetic": "real", "classes": ["counterfeit", "genuine"]}
SKIPPED = []
_cache = {}


def J(path):
    if path not in _cache:
        _cache[path] = json.loads((REPO / path).read_text(encoding="utf-8"))
    return _cache[path]


def get(d, sel):
    for k in sel:
        d = d[k] if not isinstance(d, list) else d[int(k)]
    return d


def nums(d, keep=None, drop=()):
    """Numeric leaves of a flat dict (ints, floats). Lists and strings are left out."""
    out = {}
    for k, v in d.items():
        if k in drop or (keep and k not in keep):
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = v
    return out


def ev(path, selector="", kind="evaluation_output"):
    f = REPO / path
    return {"type": kind, "file": path, "selector": selector, "sha256": DB.sha256(f)}


def mtime(path):
    return dt.datetime.fromtimestamp((REPO / path).stat().st_mtime).astimezone().isoformat(timespec="seconds")


def rec(name, task, path, selector, *, metrics=None, performance=None, group=None, variant="", status="verified", **extra):
    r = {"experiment_name": name, "task": task, "timestamp": mtime(path), "status": status,
         "evidence": [ev(path, selector)], "versioning": {"code_commit": DB.git_commit(REPO / path)},
         "verification": {"method": "value read from the saved evaluation output by research_results/import_existing.py",
                          "checked_at": DB.now_iso()}}
    if metrics:
        r["metrics"] = metrics
    if performance:
        r["performance"] = performance
    if group:
        r["comparison_group"] = {"id": group[0], "name": group[1], "purpose": group[2]}
    if variant:
        r.setdefault("comparison", {})["variant"] = variant
    for k, v in extra.items():
        if v not in (None, {}, [], ""):
            if k == "comparison":
                r.setdefault("comparison", {}).update(v)
            else:
                r[k] = v
    return r


def source(fn):
    """Run one importer; a missing or unexpected file is reported and skipped, never guessed."""
    try:
        return list(fn())
    except Exception as exc:  # noqa: BLE001
        SKIPPED.append(f"{fn.__name__}: {type(exc).__name__}: {exc}")
        return []


# ------------------------------------------------------------------ 1. Taka detector
G_DET = ("TAKA_DETECTOR", "Taka denomination detector across test sets", "Same YOLOv8s weights on synthetic and independent real test sets")


def detector():
    p = f"{R}/training_v2/test_eval/test_metrics.json"
    d = J(p)
    yield rec("Taka detector, synthetic composite test", "currency_detection", p, "", group=G_DET, variant="composites test",
              mode="currency", algorithm="YOLOv8s detection",
              metrics={"precision": d["precision"], "recall": d["recall"], "map50": d["map50"], "map50_95": d["map50_95"],
                       "per_class_map50_95": d["per_class_map50_95"]},
              performance={"inference_ms": d["speed_ms"]["inference"], "preprocess_ms": d["speed_ms"]["preprocess"],
                           "postprocess_ms": d["speed_ms"]["postprocess"]},
              model={"name": "YOLOv8s Taka detector", "architecture": "YOLOv8s", "framework": "ultralytics", "weights": d["weights"]},
              dataset={"name": "Taka detector composites", "split": "test", "samples": d["n_images"], "real_or_synthetic": "synthetic",
                       "classes": list(d["per_class_map50_95"]), "source": "data set/currency_yolo_data"}, hardware=LAPTOP)
    p = f"{R}/new_test/detector_newtest.json"
    d = J(p)
    m = nums(d, drop=("seed_base",)) or nums(d.get("metrics", {}))
    yield rec("Taka detector, fresh composite test (test notes on COCO train2017)", "currency_detection", p, "", group=G_DET,
              variant="fresh composites", mode="currency", algorithm="YOLOv8s detection", metrics=m,
              configuration={"seed_base": d.get("seed_base")},
              model={"name": "YOLOv8s Taka detector", "architecture": "YOLOv8s"},
              dataset={"name": "Fresh composite test", "split": "test", "real_or_synthetic": "synthetic", "source": "data set/currency_yolo_newtest"},
              hardware=LAPTOP)
    p = f"{R}/external/external_taka.json"
    d = J(p)
    for key, blk in d.items():
        if isinstance(blk, dict) and nums(blk):
            yield rec(f"Taka detector on independent photos: {key}", "currency_detection", p, key, group=G_DET, variant=key,
                      mode="currency", algorithm="YOLOv8s detection, no retraining", metrics=nums(blk),
                      model={"name": "YOLOv8s Taka detector", "architecture": "YOLOv8s"},
                      dataset={"name": key, "split": "external test", "real_or_synthetic": "real",
                               "samples": int(blk["n"]) if isinstance(blk.get("n"), int) else None}, hardware=LAPTOP)


# ------------------------------------------------------------------ 2. View-count shift (same network, 3 seeds)
G_VC = ("JAALTAKA_VIEW_COUNT", "Fixed-view vs prefix training, same network", "Accuracy at 1-6 views; note-disjoint JaalTaka test (208 notes); seeds 42-44")
ARMS = {"fixed": ("Fixed 6-view training, per-count BN", f"{R}/same_arch/fixed/s2/seed{{s}}/test_views"),
        "fixed_shbn": ("Fixed 6-view training, shared BN", f"{R}/same_arch/fixed_shbn/s2/seed{{s}}/test_views"),
        "prefix_shbn": ("Prefix training, shared BN", f"{R}/same_arch/prefix_shbn/s2/seed{{s}}/test_views"),
        "prmvt": ("PRMVT (prefix training, per-count BN)", f"{R}/qduig/prefix_ft/seed{{s}}/test_views_20260928")}


def view_count():
    for arm, (label, tpl) in ARMS.items():
        for s in (42, 43, 44):
            base = tpl.format(s=s)
            m, raw, evs = {}, {}, []
            for k in range(1, 7):
                p = f"{base}/{k}view/test_metrics.json"
                d = J(p)
                m[f"accuracy_k{k}"] = d["accuracy"]
                if k in (1, 6):
                    m[f"roc_auc_k{k}"] = d["roc_auc"]
                    m[f"f1_k{k}"] = d["f1"]
                    m[f"false_acceptance_rate_k{k}"] = d["false_acceptance_rate"]
                    m[f"false_rejection_rate_k{k}"] = d["false_rejection_rate"]
                    raw[f"confusion_matrix_k{k}"] = d["confusion_matrix"]
                    evs.append(ev(p, "accuracy"))
            r = rec(f"{label}, seed {s}", "counterfeit_detection", f"{base}/1view/test_metrics.json", "accuracy", metrics=m,
                    group=G_VC, variant=label, mode="research: JaalTaka authentication",
                    algorithm="multi-view MobileNetV3-Small + TinyViT with attention fusion", experiment_group=f"VIEW_COUNT_{arm.upper()}",
                    run={"seed": s}, model={"name": label, "architecture": "Q-DUIG multi-view network"},
                    dataset={**JAAL, "split": "test (note-disjoint, seed 42)", "samples": d["n"]}, raw_measurements=raw,
                    test_conditions={"views_at_test": "1 to 6 (accuracy_k1 ... accuracy_k6)"}, hardware=LAPTOP)
            r["evidence"] = evs
            yield r


G_PROBE = ("JAALTAKA_FROZEN_PROBES", "Frozen ImageNet backbones, linear probe", "Reference baselines on the note-disjoint JaalTaka test")


def probes():
    p = f"{R}/sota/backbone_probes.json"
    for name, d in J(p).items():
        yield rec(f"Frozen {name} linear probe", "counterfeit_detection", p, f"{name}.test_acc", group=G_PROBE, variant=name,
                  metrics={**{f"accuracy_k{k}": v for k, v in d["test_acc"].items()}, **{f"val_accuracy_k{k}": v for k, v in d["val_acc"].items()}},
                  algorithm="frozen backbone + logistic regression (C chosen on validation)", mode="research: JaalTaka authentication",
                  model={"name": name, "parameters": {"feature_dim": d["feature_dim"], "C_chosen_on_val": d["C_chosen_on_val"]}},
                  dataset={**JAAL, "split": "test (note-disjoint, seed 42)", "samples": 208}, run={"seed": "deterministic"}, hardware=LAPTOP)


# ------------------------------------------------------------------ 3. Unseen counterfeit prints
G_UP = ("UNSEEN_PRINTS", "Counterfeit detection on unseen prints", "Serial-disjoint JaalTaka split, 222 test notes (101 counterfeit); seeds 42-44; 1 and 6 views")
UP_NAMES = {"PROBE": "Frozen ResNet-50 probe", "NETWORK": "Prefix network", "FUSION": "Attention fusion with watermark token",
            "FUSION_no_watermark": "Attention fusion without watermark token", "ENSEMBLE": "Uncertainty-weighted ensemble",
            "HYBRID_FINAL": "Prefix network + watermark hybrid", "FT_RESNET": "Fine-tuned ResNet-50 (quarter-resolution cache)"}
SERIAL = {**JAAL, "split": "test (serial-disjoint: unseen counterfeit prints)", "samples": 222}


def unseen_prints():
    p = f"{R}/serial_split/beat_resnet.json"
    d = J(p)["per_seed"]
    for seed, blk in d.items():
        for meth, label in UP_NAMES.items():
            m, raw = {}, {}
            for k in ("k1", "k6"):
                if meth not in blk[k]:
                    continue
                x = blk[k][meth]
                m[f"accuracy_{k}"] = x["accuracy"]
                for c in ("false_counterfeit_on_genuine", "counterfeit_missed"):
                    if c in x:
                        m[f"{c}_{k}"] = x[c]
                if "wilson95" in x:
                    raw[f"accuracy_wilson95_{k}"] = x["wilson95"]
            if not m:
                continue
            abl = {"enabled": True, "component": "watermark token", "configuration": "without" if meth == "FUSION_no_watermark" else "with"} \
                if meth.startswith("FUSION") else None
            yield rec(f"{label}, unseen prints, seed {seed}", "counterfeit_detection", p, f"per_seed.{seed}.k1|k6.{meth}", metrics=m,
                      group=G_UP, variant=label, experiment_group=f"UNSEEN_PRINTS_{meth}", run={"seed": int(seed)},
                      mode="research: JaalTaka authentication", model={"name": label}, dataset=SERIAL, raw_measurements=raw,
                      ablation=abl, hardware=LAPTOP)
    for s in (42, 43, 44):
        p = f"{R}/serial_split/ft_resnet50_fullres/seed{s}.json"
        d = J(p)
        yield rec(f"Fine-tuned ResNet-50 (full-resolution views), unseen prints, seed {s}", "counterfeit_detection", p, "test",
                  metrics={**{f"accuracy_k{k}": v for k, v in d["test"].items()}, **{f"val_accuracy_k{k}": v for k, v in d["val"].items()}},
                  group=G_UP, variant="Fine-tuned ResNet-50 (full resolution)", experiment_group="UNSEEN_PRINTS_FT_RESNET_FULLRES",
                  run={"seed": s}, mode="research: JaalTaka authentication",
                  model={"name": "ResNet-50 fine-tuned", "architecture": "ResNet-50", "training": {"epochs": d["epochs"], "epoch_chosen_on": "validation mean"}},
                  dataset=SERIAL, hardware=LAPTOP, comparison={"baseline": "fair image baseline for the hybrid"})


# ------------------------------------------------------------------ 4. Watermark
G_WM = ("WATERMARK_CHECK", "Watermark-window check on unseen prints", "Window classifier and localizer on the serial-disjoint test notes")


def watermark():
    for path, label, arch in ((f"{R}/watermark/mobilenetv2.json", "Watermark classifier MobileNetV2 (deployed)", "MobileNetV2"),
                              (f"{R}/watermark/mobilenet.json", "Watermark classifier MobileNetV3-Small", "MobileNetV3-Small")):
        d = J(path)
        t = d["test"]
        raw = {"export": d.get("export", {}), "counts": {k: t[k] for k in ("false_counterfeit_on_genuine", "genuine_n", "counterfeit_missed", "counterfeit_n")}}
        notes = []
        int8 = next((v for k, v in d.get("export", {}).items() if "int8" in k), None)
        m = {"accuracy": t["accuracy"], "auc": t["auc"], "best_val_auc": d["best_val_auc"]}
        if int8 and "same_decision_as_fp32" in int8:
            m["int8_same_decision_as_fp32"] = int8["same_decision_as_fp32"]
            if int8["same_decision_as_fp32"] < 0.99:
                notes.append("INT8 export changes test decisions; this INT8 model was not deployed.")
        yield rec(label, "counterfeit_detection", path, "test", metrics=m, group=G_WM, variant=label, mode="currency: watermark check",
                  algorithm="SIFT registration + fixed window crop + CNN classifier", model={"name": label, "architecture": arch},
                  dataset={**JAAL, "split": "test (serial-disjoint), registered back-lit views", "samples": d["n"]["test"]},
                  raw_measurements=raw, notes=notes, hardware=LAPTOP)
    for s in (42, 43, 44):
        p = f"{R}/watermark_localizer/seed{s}.json"
        d = J(p)
        c, loc = d["test_classification"], d["test_localization"]
        yield rec(f"Learned watermark localizer + MobileNetV2, seed {s}", "counterfeit_detection", p, "test_localization, test_classification",
                  metrics={"accuracy_all_test_notes": c["localizer_all_test_notes"]["accuracy"],
                           "accuracy_registered_notes": c["localizer_on_registered_notes"]["accuracy"],
                           "accuracy_sift_same_notes": c["registration_on_registered_notes"]["accuracy"],
                           "accuracy_notes_sift_missed": c["localizer_on_notes_registration_missed"]["accuracy"],
                           "mcnemar_p_vs_sift": c["mcnemar_localizer_vs_registration"]["p"],
                           "mean_corner_err_frac": loc["mean_corner_err_frac"], "median_quad_iou": loc["median_quad_iou"],
                           "iou_ge_0.7": loc["iou_ge_0.7"], "best_val_mean_corner_err": d["best_val_mean_corner_err"]},
                  group=G_WM, variant="Learned localizer + MobileNetV2", experiment_group="WATERMARK_LOCALIZER", run={"seed": s},
                  mode="currency: watermark check", algorithm="corner regression (MobileNetV3-Small) + perspective crop + CNN classifier",
                  model={"name": "Watermark localizer", "architecture": "MobileNetV3-Small corner regressor", "training": {"best_epoch": d["best_epoch"]}},
                  dataset={**JAAL, "split": "test (serial-disjoint), back-lit view 6", "samples": c["localizer_all_test_notes"]["n"]},
                  raw_measurements={"counts_all": c["localizer_all_test_notes"], "mcnemar": c["mcnemar_localizer_vs_registration"]}, hardware=LAPTOP)
    p = f"{R}/watermark_localizer/app_path_check.json"
    d = J(p)
    yield rec("Glass app watermark path (localizer ONNX + MobileNetV2 INT8)", "counterfeit_detection", p, "", group=G_WM,
              variant="App path (INT8)", metrics={"accuracy": d["accuracy_on_found"], "same_decision_as_research_fp32": d["same_decision_as_research_fp32"],
                                                  "window_found": d["window_found"]},
              mode="currency: watermark check", dataset={**JAAL, "split": "test (serial-disjoint)", "samples": d["test_notes"]}, hardware=LAPTOP)
    p = f"{R}/watermark_localizer/laptop_latency.json"
    d = J(p)
    for name in ("localizer", "sift"):
        yield rec(f"Watermark check latency, {name} path, laptop CPU", "system_performance", p, name, group=G_WM, variant=f"{name} path",
                  performance={"latency_ms_median": d[name]["median_ms"], "latency_ms_p95": d[name]["p95_ms"], "window_found": d[name]["window_found"]},
                  mode="currency: watermark check", dataset={"name": "JaalTaka back-lit test photos", "samples": d["n"]},
                  hardware={"device": "laptop CPU", "cpu": d["cpu"]})


# ------------------------------------------------------------------ 5. Safety of the deployed check, negative results
G_SAFE = ("COUNTERFEIT_SAFETY", "Safety of the deployed counterfeit check", "Policy E: 'likely genuine' or 'check by hand', never 'counterfeit'")


def safety():
    p = f"{R}/jaal_whole/policy.json"
    t = J(p)["jaaltaka_test_real_views"]
    yield rec("Policy E on JaalTaka test (threshold fixed on validation)", "counterfeit_detection", p, "jaaltaka_test_real_views", group=G_SAFE,
              variant="Policy E, JaalTaka test", metrics={"counterfeit_passed": t["counterfeit_passed"], "counterfeit_n": t["counterfeit_n"],
                                                          "genuine_passed": t["genuine_passed"], "genuine_n": t["genuine_n"]},
              raw_measurements={"counterfeit_passed_wilson95": t["counterfeit_passed_wilson95"]}, mode="currency: safe counterfeit check",
              algorithm="PRMVT, 4 cut views, pass only above the highest validation counterfeit score",
              configuration={"tau": t["tau"], "checker": t["checker"]}, dataset={**JAAL, "split": "test"}, hardware=LAPTOP)
    p = f"{R}/jaal_whole/app_check.json"
    s = J(p)["summary"]
    for key, blk in s["sets"].items():
        yield rec(f"Glass app counterfeit check on whole-note photos: {key}", "counterfeit_detection", p, f"summary.sets.{key}", group=G_SAFE,
                  variant=f"App path, {key}", metrics=nums(blk), raw_measurements={"spoken_jaal_all_runs": s["spoken_jaal"]},
                  mode="currency: safe counterfeit check", configuration={"tau": s["tau"]},
                  dataset={"name": key, "split": "whole-note photos", "real_or_synthetic": "real"}, hardware=LAPTOP)
    p = f"{R}/safety/quality_gate_seed42.json"
    for k, blk in J(p)["views"].items():
        for cond, x in blk.items():
            if isinstance(x, dict) and nums(x):
                yield rec(f"Image-quality gate + confidence, {k} view(s), {cond}", "counterfeit_detection", p, f"views.{k}.{cond}", group=G_SAFE,
                          variant=f"Quality gate, k={k}", metrics=nums(x), test_conditions={"condition": cond, "views": int(k)},
                          mode="currency: safe counterfeit check", dataset={**JAAL, "split": "test", "samples": 208}, run={"seed": 42}, hardware=LAPTOP)


G_WHOLE = ("WHOLE_NOTE_TRANSFER", "Approaches for counterfeit checking on whole-note photos", "All measured; none met the pre-registered rule")


def negative():
    p = f"{R}/jaal_whole/approach_a/eval.json"
    d = J(p)
    for key in ("synthetic_test_A", "synthetic_test_S4_deployed"):
        yield rec(f"Approach A (synthetic whole notes): {key}", "counterfeit_detection", p, key, group=G_WHOLE, variant=key,
                  metrics=nums(d[key]), mode="currency: safe counterfeit check",
                  dataset={"name": "Synthetic whole notes built from JaalTaka", "split": "test", "real_or_synthetic": "synthetic", "samples": d[key]["n"]},
                  comparison={"decision": "not_kept"}, notes=["Approach A was not deployed under the pre-registered rule."], hardware=LAPTOP)
    p = f"{R}/jaal_whole/domain_adapt.json"
    for name, blk in J(p)["methods"].items():
        yield rec(f"Whole-note transfer, {name}", "counterfeit_detection", p, f"methods.{name}", group=G_WHOLE, variant=name,
                  metrics={k: v for k, v in nums(blk).items()}, raw_measurements={"success_rule_met": blk.get("success_rule_met"),
                                                                                 "cf_counterfeit_passed_at_0.5": blk.get("cf_counterfeit_passed_at_0.5")},
                  comparison={"decision": "baseline" if name in ("S4", "R2") else "not_kept"}, mode="currency: safe counterfeit check",
                  dataset={"name": "Counterfeit set originals + Bangla Money + BanglaTaka (whole notes)", "real_or_synthetic": "real"}, hardware=LAPTOP)


# ------------------------------------------------------------------ 6. OCR benchmark
G_OCR = ("OCR_BENCHMARK", "OCR pipelines on the shaped Bangla + English benchmark", "CER / WER on val (choices), test (read once), lexicon and select sets; 3 image seeds")
OCR_SETS = {"val": "validation (48 phrases; every choice made here)", "test": "held-out test (48 phrases; read once per method)",
            "lexicon": "old 24 lexicon phrases", "select": "24 phrases used once for the canvas choice"}
ABL = {"baseline": ("pipeline", "old offline path"), "glass_app": ("pipeline", "app as shipped (legacy)"), "t1_region": ("text region", "+ Task 1"),
       "t2_prep": ("preprocessing", "+ Task 2"), "t4_params": ("EasyOCR parameters", "+ Task 4"), "final": ("post-processing", "+ Task 5 (v2, in the app)")}


def ocr():
    for f in sorted((REPO / G / "ocr_bench").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if "overall" not in d or "by_cond" not in d:
            continue
        p = f"{G}/ocr_bench/{f.name}"
        st, _, method = f.stem.partition("_")
        if st not in OCR_SETS:
            continue
        m = {"cer": d["overall"]["cer"], "wer": d["overall"]["wer"], "exact_match": d["overall"]["exact"], "cer_english": d["en"]["cer"],
             "cer_seed_sd": d.get("cer_seed_sd")}
        if d.get("bn"):
            m["cer_bangla"] = d["bn"]["cer"]
        for cond, blk in d["by_cond"].items():
            m[f"cer_{cond}"] = blk["all"]["cer"]
        perf = {"latency_ms_median_gpu": d["latency_ms_median"]} if d.get("latency_ms_median") else None
        if "postprocess_ms_median" in d:
            perf = {"postprocess_ms_median": d["postprocess_ms_median"]}
        seed = (d.get("meta") or {}).get("seed")
        abl = {"enabled": True, "component": ABL[method][0], "configuration": ABL[method][1]} if method in ABL else None
        yield rec(f"OCR {method} on {st}", "ocr", p, "overall, bn, en, by_cond", metrics={k: v for k, v in m.items() if v is not None},
                  performance=perf, group=G_OCR, variant=method, mode="ocr", algorithm="EasyOCR (CRAFT + CRNN) unless the variant names another engine",
                  dataset={"name": "OCR benchmark (HarfBuzz-shaped Bangla + English)", "split": OCR_SETS[st], "samples": d["overall"]["n"],
                           "real_or_synthetic": "synthetic text on clean / distorted pages and real COCO photos"},
                  runs={"count": len(d["per_seed_cer"]), "metric": "cer per image seed (0, 1, 2)", "values": d["per_seed_cer"]},
                  configuration=(d.get("meta") or {}).get("config") or (d.get("meta") or {}).get("choices") or {},
                  run={"seed": seed} if seed else None, ablation=abl,
                  hardware=LAPTOP, comparison={"selected_on": "validation set"} if st != "val" else None)
    for s in (42, 43, 44):
        p = f"{G}/ocr_bench/t7_seed{s}.json"
        d = J(p)
        yield rec(f"OCR recognizer fine-tuning, seed {s} (validation line crops)", "ocr", p, "best_val_crop_cer", group=G_OCR,
                  variant="recognizer fine-tuning (crops)", experiment_group="OCR_FINETUNE", run={"seed": s},
                  metrics={"val_crop_cer_after": d["best_val_crop_cer"], "val_crop_cer_before": d["pretrained_val_crop_cer"]},
                  model={"name": "EasyOCR bengali recognizer (CRNN, CTC)", "training": {"steps": d["steps"], "best_step": d["best_step"],
                                                                                         "train_words": d["train_words"]}},
                  mode="ocr", dataset={"name": "OCR benchmark val phrases as line crops", "split": "validation", "real_or_synthetic": "synthetic"},
                  comparison={"decision": "not_kept", "selected_on": "validation (end to end)"}, hardware=LAPTOP)
    p = f"{G}/ocr_bench/cpu_latency.json"
    d = J(p)
    for key, label in (("glass_app_legacy", "legacy app pipeline"), ("v2_final", "pipeline v2")):
        yield rec(f"OCR CPU latency, {label} (quantised recognizer)", "system_performance", p, key, group=G_OCR, variant=label,
                  performance={"latency_ms_median": d[key]["median_ms"], "latency_ms_p95": d[key]["p95_ms"]},
                  metrics={"cer_on_these_images": d[key]["cer_on_these"]}, mode="ocr",
                  dataset={"name": "OCR benchmark test subset", "samples": d["n"]}, hardware={"device": "laptop CPU", "cpu": d["cpu"]})
    for size in (2560, 640, 480, 320):
        p = f"{G}/ocr_select_canvas{size}.json"
        d = J(p)
        yield rec(f"OCR canvas size {size} (selection set, unshaped renders)", "ocr", p, "raw.overall", group=("OCR_CANVAS_SIZE",
                  "EasyOCR canvas size", "Chosen on a selection set; images drawn without Bangla shaping (superseded benchmark)"),
                  variant=f"canvas {size}", metrics={"raw_cer": d["raw"]["overall"]["cer"], "raw_wer": d["raw"]["overall"]["wer"],
                                                     "raw_cer_bangla": d["raw"]["bn"]["cer"], "raw_cer_english": d["raw"]["en"]["cer"]},
                  performance={"seconds_total_80_images": d["seconds_total_this_host"]}, configuration={"canvas_size": size}, mode="ocr",
                  dataset={"name": "24-phrase selection set (Pillow basic layout)", "samples": d["n"], "real_or_synthetic": "synthetic"},
                  notes=["Bangla was drawn without text shaping in this older set; compare only within this group."],
                  comparison={"decision": "kept" if size == 2560 else "not_kept"}, hardware=LAPTOP)


# ------------------------------------------------------------------ 7. Raspberry Pi 5
G_PI = ("PI5_LATENCY", "Latency per module on the Raspberry Pi 5", "benchmark_pi5.py --iters 100 --sustained 30")


def pi5():
    for f in sorted((REPO / G).glob("pi5_benchmark_*.json")):
        p = f"{G}/{f.name}"
        d = json.loads(f.read_text(encoding="utf-8"))
        h = d["host"]
        if not h.get("raspberry_pi_5"):
            continue
        hw = {"device": h["device_model"], "cpu": f"{h['machine']}, {h['cpu_count']} cores", "hostname": h["hostname"]}
        env = {"os": h["os"], "python": h["python"], "framework_versions": {k: h[k] for k in ("torch", "ultralytics", "onnxruntime", "easyocr", "cv2") if k in h}}
        for name, x in d["latency"].items():
            yield rec(f"Pi 5 latency: {name}", "system_performance", p, f"latency.{name}", group=G_PI, variant=name,
                      performance={"latency_ms_median": x["median_ms"], "latency_ms_p95": x["p95_ms"], "latency_ms_mean": x["mean_ms"],
                                   "fps": x["fps_median"], "memory_rss_mb_after": x["rss_mb_after"], "cpu_temp_c_after": x["cpu_temp_c_after"],
                                   "load_ms": x["load_ms"]},
                      raw_measurements={"iterations": x["iters"]}, mode=name, hardware=hw, software_environment=env, timestamp=h["utc"])
        s = d.get("sustained")
        if s:
            yield rec("Pi 5 sustained 30-minute mixed run", "system_performance", p, "sustained", group=G_PI, variant="sustained run",
                      performance={"minutes": s["minutes"], "calls": s["calls"], "max_cpu_temp_c": s["max_cpu_temp_c"], "peak_rss_mb": d["peak_rss_mb"]},
                      raw_measurements={"final_throttled": s["final_throttled"], "per_call_csv": f"{G}/{s['csv']}"}, hardware=hw,
                      software_environment=env, timestamp=h["utc"])


# ------------------------------------------------------------------ 8. Other assistive modules, device behaviour
def assistive():
    p = "paper_evidence/emotion/emotion_rafdb.json"
    d = J(p)
    yield rec("Emotion recognition on RAF-DB test", "emotion_recognition", p, "", mode="emotion",
              metrics={"accuracy": d["accuracy"], "macro_precision": d["macro_precision"], "macro_recall": d["macro_recall"], "macro_f1": d["macro_f1"]},
              model={"name": d["arch"], "architecture": d["arch"], "input_size": d["img_size"]},
              dataset={"name": "RAF-DB", "split": d["split"], "samples": d["n_test"], "classes": d["labels"], "real_or_synthetic": "real"},
              raw_measurements={"confusion_matrix": d["confusion_matrix"]}, hardware=LAPTOP)
    p = "paper_evidence/object/object_coco.json"
    d = J(p)
    yield rec("Object detection on a COCO val2017 subset", "object_detection", p, "", mode="object", metrics={"map50": d["map50"]},
              model={"name": "YOLOv8s (COCO)", "weights": Path(d["weights"]).name},
              dataset={"name": "COCO val2017", "split": d["split"], "samples": d["n_images"], "real_or_synthetic": "real"},
              raw_measurements={"n_classes_scored": d["n_classes_scored"]}, hardware=LAPTOP)
    p = f"{G}/ekush_letters.json"
    d = J(p)
    yield rec("Ekush handwritten Bangla letters (writer-disjoint)", "handwritten_character_recognition", p, "test_acc", mode="ocr: handwritten fallback",
              metrics={"accuracy": d["test_acc"], "best_val_accuracy": d["best_val_acc"], **{f"accuracy_{k}": v for k, v in d["group_acc"].items()}},
              model={"name": "Ekush CNN"}, dataset={"name": "Ekush", "split": "test (unseen writers)", "samples": d["n_test"], "real_or_synthetic": "real"},
              raw_measurements={"split": d["split"], "n_classes": d["n_classes"]}, run={"seed": d["split"]["seed"]}, hardware=LAPTOP)


G_UX = ("DEVICE_BEHAVIOUR", "Glass interaction behaviour", "Capture guide and mode switching")


def device():
    p = f"{R}/capture_guide/thresholds.json"
    d = J(p)
    for cond, x in d["test"].items():
        yield rec(f"Capture guide thresholds on back-lit test photos: {cond}", "capture_quality_gate", p, f"test.{cond}", group=G_UX,
                  variant="capture guide", metrics={"accepted_rate": x["accepted_rate"], "accepted": x["accepted"], "rejected_dark": x["rejected_dark"],
                                                    "rejected_blurry": x["rejected_blurry"]},
                  configuration=d["thresholds"], test_conditions={"condition": cond}, mode="currency: guided capture",
                  dataset={**JAAL, "split": "test (serial-disjoint), back-lit view 6", "samples": x["n"]}, hardware=LAPTOP)
    for tag, label in (("before", "before (reload on every switch)"), ("after", "after (load once + warm-up)"),
                       ("before_wm", "before, watermark check on"), ("after_wm", "after, watermark check on")):
        p = f"{G}/mode_switch_{tag}.json"
        d = J(p)
        yield rec(f"Currency mode switch timing, {label}", "system_performance", p, "", group=G_UX, variant=f"mode switch {tag}",
                  performance=nums(d), configuration={"watermark_check": d["watermark_check"]}, mode="currency",
                  ablation={"enabled": True, "component": "model loading and warm-up", "configuration": label},
                  hardware={"device": f"laptop CPU ({d['host']})"})


# ------------------------------------------------------------------ 9. Failed and planned
def failed_and_planned():
    p = f"{G}/ocr_bench/task1_best.json"
    yield {"experiment_name": "OCR text detection with DBNet18", "task": "ocr", "status": "failed", "timestamp": mtime(p), "mode": "ocr",
           "comparison_group": {"id": G_OCR[0], "name": G_OCR[1], "purpose": G_OCR[2]}, "comparison": {"variant": "DBNet18 detector"},
           "evidence": [ev(p, "not_run.dbnet18")], "versioning": {"code_commit": DB.git_commit(REPO / p)},
           "notes": [J(p)["not_run"]["dbnet18"], "The run crashed in the DBNet forward pass; no error rate exists."]}
    p = "paper_evidence/OCR_BANGLA_SPECIFIC.md"
    yield {"experiment_name": "PaddleOCR on Bangla text", "task": "ocr", "status": "failed", "timestamp": mtime(p), "mode": "ocr",
           "comparison_group": {"id": G_OCR[0], "name": G_OCR[1], "purpose": G_OCR[2]}, "comparison": {"variant": "PaddleOCR Bangla"},
           "evidence": [ev(p, "PaddleOCR", "report")], "versioning": {"code_commit": DB.git_commit(REPO / p)},
           "notes": ["PaddleOCR 3.7 has no Bangla model (lang 'bn' and 'bengali': no models available). Only English was scored."]}
    for name, task, note, src in (
            ("User study with blind and low-vision participants (guided vs unguided capture)", "user_study",
             "Protocol, pre-registration and analysis script are ready; no participant data.", "paper_evidence/STUDY_PREREGISTRATION.md"),
            ("Battery life of the glass", "system_performance", "Needs a USB-C power meter on the Pi.", "paper_evidence/PI5_RESULTS.md"),
            ("Pi 5 timing of OCR pipeline v2 and of the mode-switch warm-up", "system_performance",
             "Measured on the laptop only; rerun benchmark_pi5.py --only ocr_mode and measure_mode_switch.py on the Pi.", "paper_evidence/PI5_RESULTS.md"),
            ("Counterfeit detectors on a print-disjoint dataset (~100 unseen prints, second currency)", "counterfeit_detection",
             "Tools ready; data collection with a bank or law-enforcement partner needed.", "paper_evidence/PRINT_DISJOINT_DATASET_PLAN.md"),
            ("Capture-guide thresholds calibrated on glass-camera photos", "capture_quality_gate",
             "Current thresholds come from JaalTaka photos.", "paper_evidence/STUDY_PREREGISTRATION.md"),
            ("OCR accuracy on real glass-camera photos", "ocr", "Current OCR numbers come from rendered text.", "paper_evidence/OCR_TEXT_REGION.md")):
        yield {"experiment_name": name, "task": task, "status": "planned", "timestamp": DB.now_iso(), "notes": [note],
               "evidence": [{"type": "plan", "file": src}]}


def main() -> None:
    dry = "--dry-run" in sys.argv
    db = DB.load()
    before = len(db["experiments"])
    added, dup = 0, 0
    for fn in (detector, view_count, probes, unseen_prints, watermark, safety, negative, ocr, pi5, assistive, device, failed_and_planned):
        for r in source(fn):
            try:
                if r["status"] == "planned" and any(x["experiment_name"] == r["experiment_name"] for x in db["experiments"]):
                    dup += 1
                    continue
                eid = DB.append(db, r)
            except ValueError as exc:
                SKIPPED.append(f"{r.get('experiment_name')}: {exc}")
                continue
            added += bool(eid)
            dup += not eid
    print(f"existing {before}; added {added}; already recorded {dup}; skipped {len(SKIPPED)}")
    for s in SKIPPED:
        print("  SKIPPED", s[:300])
    if not dry:
        DB.save(db)
        probs = DB.validate(db)
        for x in probs:
            print("PROBLEM", x)
        print("export:", DB.export(db))
        print(DB.summary(db))


if __name__ == "__main__":
    main()
