# Print-disjoint counterfeit dataset: collection plan and tools (2026-10-01)

**Why this comes first.** JaalTaka's note split lets the same counterfeit print appear in train and test, so its published accuracy is a test on reprints (`JAALTAKA_SERIAL_AUDIT.md`). The serial-disjoint split leaves 101 test counterfeits, which are about 24 prints (m_eff = 23.6, Theorem 16). A reviewer counts prints, not photos.

At 90 % print-level accuracy the Wilson 95 % interval is:

| Test prints | Wilson 95 % interval at 90 % |
|---:|---|
| 24 | ±12 points |
| 100 | ±6 points |

A new model scored on the same 24 prints cannot move a main-track review.

## What to collect

| Item | Target | Minimum |
|---|---|---|
| Counterfeit prints in the TEST split | ~100 | 50 |
| Counterfeit prints in total (test fraction 0.5 for counterfeits) | ~200 | 100 |
| Photos per note | phone room, phone backlight, glass room, glass backlight | glass backlight + phone room |
| Genuine notes | matched per denomination, at least as many notes as counterfeit prints | same |
| Second currency | one of INR or USD, same grouping and photo set | 30 test prints |
| Denominations | every denomination for which counterfeits are available | 500 and 1,000 BDT |

- **Print id.** Counterfeit notes with the same serial, or the same plate defects, share one print id. This is the partner's grouping. When unsure, merge prints: over-merging is safe, and splitting one print into two is a leak.
- **Genuine notes.** Each genuine note is its own print unless serials show a bundle.
- **Partner.** Real counterfeits come from a bank currency-management unit or law enforcement. Their written permission, their print grouping and their chain of custody belong in the paper's data section.
- **Second currency.** It was out of scope under the earlier "Bangladeshi Taka only" rule. Adding it is your decision. Nothing was downloaded.

## Tools (in `realtime_bangla_taka_detection/scripts/dataset/`)

**`print_capture.py` stores and guards the photos.**
- **Per photo it records:** note id, print id, currency, denomination, label, camera (phone or glass), lighting (room or backlight), blur (Laplacian variance), exposure (mean, black and white clipping), SHA-256 and time.
- **It refuses** (the self-test `test_print_capture.py` covers each case):
  - a print with no split;
  - a print photographed into a second split;
  - a note moving to a second print;
  - a label or currency that contradicts the plan;
  - a byte-identical duplicate;
  - ids that are not file-safe.

**`plan` fixes the split before any training.**
- It reads `prints.csv` (`print_id,currency,denomination,label`) and assigns each print to train, validation or test, stratified by currency × denomination × label.
- It never moves a print that already has a split. New prints can be added later.
- It prints the SHA-256 of `splits.json`: put that hash in the paper.

**`check` and `export`.**
- `check` re-verifies every rule over the whole manifest.
- `export` writes `train.txt`, `val.txt`, `test.txt` and `datasheet_counts.json`, and refuses to export if any rule is violated.

**Localizer tools.**
- `annotate_watermark.py` records four clicks per back-lit photo for the watermark window. It works for any currency and hardcodes no box.
- `manifest_to_localizer_labels.py` turns the manifest and those clicks into labels for `scripts/train/train_watermark_localizer.py <seed> <labels_dir>`.
- That path was run end to end on a 40-photo throwaway set: capture, convert, train, test, ONNX export. It proves only that the code runs.

**Example session:**
```bash
cd realtime_bangla_taka_detection
python scripts/dataset/print_capture.py plan  --root D:/jaal_v2 --prints prints.csv --test 0.5
python scripts/dataset/print_capture.py add   --root D:/jaal_v2 --image IMG_0012.jpg \
    --note-id N0001 --print-id P017 --currency BDT --denomination 500 --label counterfeit \
    --camera phone --lighting backlight
python scripts/dataset/print_capture.py add   --root D:/jaal_v2 --camera-index 0 ...   # grab from the glass camera
python scripts/dataset/print_capture.py check --root D:/jaal_v2
python scripts/dataset/print_capture.py export --root D:/jaal_v2
```

## Rules for the analysis on the new set

1. Train on train, choose epochs, thresholds and combiners on validation, and read test once per seed. Use three seeds.
2. Report print-level results.
   - Accuracy on test prints (a print counts as right when the majority of its notes are right).
   - Note-level results too, with a print-clustered bootstrap interval.
3. The baseline is a fine-tuned ResNet-50 on the full note at full resolution. It is the recipe in `scripts/train/ft_resnet_serial.py` with the `views256_full` cache.
4. The comparison is a McNemar test on test prints, done per currency.
5. If the watermark localizer only ties the ResNet, the paper is the dataset and the audit (Datasets and Benchmarks), not a new architecture.
