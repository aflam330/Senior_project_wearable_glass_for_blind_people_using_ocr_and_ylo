# Serial-anomaly head (2026-09-30)

A serial blacklist and a duplicate-serial rule were evaluated on the serial-disjoint split. They catch counterfeits whose serial was already seen in training. On prints whose serial was held out, they catch 0 of 19.

That is the expected result: an unseen print has an unseen serial, so a list of known serials cannot fire. The serial signal is therefore not trained as a fused head. Adding it to the watermark hybrid did not change decisions. Detail: `SERIAL_WATERMARK_DETECTOR.md` and `QUALITY_LOOP_FINAL.md` iteration 12.

The number to report is the watermark hybrid, not a serial detector.
