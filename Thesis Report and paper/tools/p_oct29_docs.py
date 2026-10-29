# 29 October 2026: bring the thesis hardware text and the paper package (project summary, results table, both papers)
# up to the results of that day: the C270 USB webcam of the second prototype (Camera Module 2 on the first), the third
# Pi run (benchmark_report/2026-10-29/REPORT.md), the Pi now on the laptop's full-precision Taka detector
# (raw/currency_current_code.json, raw/field/20261029_213948_memo), the UV answer under the lamp and the watermark step
# off (paper_evidence/PI_FIXES_2026-10-29.md). Run once; every replacement must match exactly once.
import pathlib
import shutil

T = pathlib.Path(__file__).resolve().parent.parent            # Thesis Report and paper
REP = T / "thesis report" / "My report"
PKG = T / "tools" / "paper_package" / "src"

shutil.copy(REP / "DIagram fig tab" / "hw_webcam.jpg", REP / "hw_webcam.jpg")

THIRD_RUN_EN = ("A third run on 29 October, with a fan and a heat sink added and the C270 webcam, read a 3-line sign in "
                "5.4~s, three times faster than on 1 October, with the currency check at 422~ms; under 30 minutes of "
                "continuous load the processor's sensor reached 87.0~$^\\circ$C with the soft temperature limit active "
                "in 55\\% of samples. The application's log of that day moved the Pi to the laptop's full-precision Taka "
                "detector: 532~ms a check, and a note named in 5 of 8 presses against 2 of 10 before.")

EDITS = {
    REP / "methodology.tex": [
        ("640$\\times$480 colour frames from a Raspberry Pi Camera Module~2 (Sony IMX219, fixed focus) read through the Picamera2 library \\cite{picamera2}, or from a USB webcam.",
         "640$\\times$480 colour frames from a USB webcam (a Logitech C270 in the second prototype) read through OpenCV, or from a Raspberry Pi Camera Module~2 (Sony IMX219, fixed focus; first prototype) read through the Picamera2 library \\cite{picamera2}."),
        ("{hw_camera.jpg}\\vspace{1pt} & Camera Module 2 & 8-megapixel Sony IMX219 sensor with a fixed-focus lens; used at 640$\\times$480 pixels &",
         "{hw_webcam.jpg}\\vspace{1pt} & USB webcam & Logitech C270 board on a USB port of the Pi (second prototype); the first prototype used a Camera Module~2 (Sony IMX219, fixed focus); used at 640$\\times$480 pixels &"),
        ("Camera Module v2, by the Raspberry Pi Foundation, CC BY-SA 4.0; ", ""),
        ("Push-buttons, headphone, spectacle frame and enclosure: our second prototype.",
         "Webcam board, push-buttons, headphone, spectacle frame and enclosure: our second prototype."),
        ("The Camera Module~2 (IMX219) is connected through the CSI ribbon cable and read with the Picamera2 library; the OpenCV video interface receives no frames from this camera on the Raspberry Pi~5. A USB webcam can be used instead: the camera manager tries Picamera2 first, then OpenCV, and the choice can be forced with one configuration value.",
         "The second prototype uses a Logitech C270 USB webcam, read through OpenCV; on the Pi it gives 7.6 frames per second at 640$\\times$480, 1.3~s after it is opened. The first prototype used a Camera Module~2 (IMX219) on the CSI ribbon cable, read with the Picamera2 library, because the OpenCV video interface received no frames from that camera on the Raspberry Pi~5. The camera manager tries Picamera2 first, then OpenCV, and the choice can be forced with one configuration value."),
        ("The check is switched off by default and its wording is cautious",
         "Since 29 October 2026 the check runs whenever a currency press is taken under the lamp, which is recognised from the colour of the picture (Section~\\ref{sec:pi5_third}), and its wording is cautious"),
    ],
    REP / "implementation.tex": [
        ("\\subsection*{2. Camera Module~2}\nThe camera is connected to the CSI connector and read with Picamera2.",
         "\\subsection*{2. Camera}\nThe second prototype's Logitech C270 webcam is plugged into a USB port of the Pi and read through OpenCV. The first prototype's Camera Module~2 was connected to the CSI connector and read with Picamera2."),
        ("Running the compressed (INT8) Taka detector, the watermark finder and checker, and a back-up emotion model (FER+)",
         "Running the watermark finder and checker, the 8-bit text detector and a back-up emotion model (FER+); the compressed (INT8) Taka detector remains an option on the Pi"),
    ],
    PKG / "latex" / "q1_journal" / "main.tex": [
        ("Camera Module 2 & Sony IMX219, fixed focus, used at 640$\\times$480 & Text, notes, scenes \\\\",
         "USB webcam & Logitech C270 (second prototype; Camera Module 2 in the first), used at 640$\\times$480 & Text, notes, scenes \\\\"),
        ("Table~\\ref{tab:pi} gives two runs.", THIRD_RUN_EN.replace("~$^", "\\,$^").replace("~ms", "\\,ms").replace("~s", "\\,s")
         + " Table~\\ref{tab:pi} gives the first two runs."),
    ],
    PKG / "latex" / "ieee_conference" / "main.tex": [
        ("carries a Camera Module~2 (640$\\times$480, fixed focus)",
         "carries a USB webcam (Logitech C270, 640$\\times$480; a Camera Module~2 on the first prototype)"),
        ("A thermal governor was added in response; its effect has not yet been measured.",
         "A thermal governor was added in response; its effect has not yet been measured. "
         + THIRD_RUN_EN.replace("~$^", "\\,$^").replace("~ms", "\\,ms").replace("~s", "\\,s")),
    ],
    PKG / "01_PROJECT_FULL_INFORMATION.md": [
        ("| Camera Module 2 | Sony IMX219, 8 MP, fixed-focus; used at 640×480 | Pictures of text, notes, surroundings |",
         "| USB webcam | Logitech C270 on a USB port (second prototype); Camera Module 2, Sony IMX219, in the first; used at 640×480 | Pictures of text, notes, surroundings |"),
        ("announce thresholds 0.75 (laptop) / 0.80 (Pi INT8)",
         "announce threshold 0.75 with the full-precision detector on laptop and Pi since 29 October (0.80 for the optional Pi INT8 copy)"),
        ("guided watermark check; optional UV fibre check (off by default)",
         "guided watermark check (off by default since 29 October: it timed out with the C270); UV fibre check whenever a press is taken under the lamp"),
        ("7. **Deployment:** off by default, with cautious wording.",
         "7. **Deployment:** runs whenever a currency press is taken under the lamp (mean blue > 2.136 × mean green; 47 of 47 UV test photos recognised, 0 of 2,282 ordinary photos taken), with cautious wording; the value is not spoken under the lamp."),
        ("| | First run, 1 Oct (cool board) | Second run, 13 Oct (hot, throttled) |\n|---|---|---|",
         "| | First run, 1 Oct (cool board) | Second run, 13 Oct (hot, throttled) | Third run, 29 Oct (fan and heat sink) |\n|---|---|---|---|"),
        ("| Currency check | 327 ms | 399 ms |", "| Currency check | 327 ms | 399 ms | 422 ms (532 ms with the full-precision detector now used) |"),
        ("| Taka detector | 254 ms | 306 ms |", "| Taka detector | 254 ms | 306 ms | 339 ms hand-held INT8; 483 ms full precision |"),
        ("| Watermark check | 50 ms | 64 ms |", "| Watermark check | 50 ms | 64 ms | 51 ms |"),
        ("| Authenticator, 1 / 6 views | 37 / 90 ms | 43 / 98 ms |", "| Authenticator, 1 / 6 views | 37 / 90 ms | 43 / 98 ms | 35 ms (1 view) |"),
        ("| Objects | YOLOv8n 207 ms | YOLOv8s 562 ms |", "| Objects | YOLOv8n 207 ms | YOLOv8s 562 ms | YOLOv8s 462 ms |"),
        ("| Emotion | — (the 48 ms was a frame without a face) | 680 ms with a face |", "| Emotion | — (the 48 ms was a frame without a face) | 680 ms with a face | 536 ms with a face |"),
        ("| Text reading, one capture | 18.7 s | 18.3 s |", "| Text reading, one capture | 18.7 s | 18.3 s | 5.4 s (3-line sign); 12-line page 15.4 s, first line at 4.4 s |"),
        ("| Memory | peak 1.48 GB | — |", "| Memory | peak 1.48 GB | — | app peak 2.76 GB |"),
        ("| Temperature | peak 77.1 °C, no throttling | peak 88.7 °C, throttled in 76% of text-test samples, clock 1.5 GHz |",
         "| Temperature | peak 77.1 °C, no throttling | peak 88.7 °C, throttled in 76% of text-test samples, clock 1.5 GHz | sensor peak 87.0 °C in 30 min of load, soft limit in 55% of samples; board surface 57.8–62.3 °C (thermometer) |"),
    ],
}

P_MD = ("| P06 | Raspberry Pi 5 (29 Oct 2026) | Median (p95) latency, fan and heat sink, C270 | currency 422 (519) ms; "
        "hand-held INT8 detector 339 (551) ms; watermark 51 (63) ms; safe-check 37 (51) ms; objects YOLOv8s 462 (635) ms; "
        "emotion with a face 536 (844) ms; 3-line sign 5.4 s; 12-line page 15.4 s, first line 4.4 s | 100 calls per row | "
        "30-min run: mean 82.8 C, peak 87.0 C, soft limit 55% of samples; power 3.7 W idle, 7.8 W sustained | "
        "Sec. pi5_third; benchmark_report/2026-10-29/REPORT.md | current |\n"
        "| P07 | Raspberry Pi 5 (29 Oct 2026) | Currency on the full-precision detector now used | 532 (627) ms a check; "
        "detector 483 ms; note named in 5 of 8 app presses (2 of 10 with the INT8 copy at 0.80); 0.81 s press to speech | "
        "100 calls; app session of 6 min | board 56 C at start | Sec. pi5_third; raw/currency_current_code.json | current |\n"
        "| F01 | UV answer under the lamp | Picture taken under the lamp | rule mean blue > 2.136 x mean green; 47 / 47 UV test "
        "photos recognised, 0 / 2,282 ordinary photos taken; genuine notes \"likely genuine\" 37 / 47 test (83 / 118 all) | "
        "uv_notes test split; currency_yolo_data test | phone photos, not the C270 | Sec. pi5_third; results/uv_check/uv_frame_*.json | current |\n"
        "| F02 | Text reading | Sharpest frame of 1 s after a press | CER 50.4% -> 3.5% (press most shaken); 28.0% -> 3.8% "
        "(all shaken alike) | 144 validation images | simulated motion blur | Sec. pi5_third; results/ocr_bench/best_frame.json | current |\n")
P_CSV = ("P06,Raspberry Pi 5 (29 Oct 2026),\"Median (p95) latency, fan and heat sink, C270\",\"currency 422 (519) ms; hand-held INT8 detector 339 (551) ms; "
         "watermark 51 (63) ms; safe-check 37 (51) ms; objects YOLOv8s 462 (635) ms; emotion with a face 536 (844) ms; 3-line sign 5.4 s; 12-line page 15.4 s, first line 4.4 s\","
         "100 calls per row,\"30-min run: mean 82.8 C, peak 87.0 C, soft limit 55%; power 3.7 W idle, 7.8 W sustained\",Sec. pi5_third; benchmark_report/2026-10-29/REPORT.md,current\n"
         "P07,Raspberry Pi 5 (29 Oct 2026),Currency on the full-precision detector now used,\"532 (627) ms a check; detector 483 ms; note named in 5 of 8 app presses "
         "(2 of 10 with the INT8 copy at 0.80); 0.81 s press to speech\",100 calls; app session of 6 min,board 56 C at start,Sec. pi5_third; raw/currency_current_code.json,current\n"
         "F01,UV answer under the lamp,Picture taken under the lamp,\"rule mean blue > 2.136 x mean green; 47 / 47 UV test photos recognised, 0 / 2,282 ordinary photos taken; "
         "genuine notes likely genuine 37 / 47 test (83 / 118 all)\",uv_notes test split; currency_yolo_data test,\"phone photos, not the C270\",Sec. pi5_third; results/uv_check/uv_frame_*.json,current\n"
         "F02,Text reading,Sharpest frame of 1 s after a press,\"CER 50.4% -> 3.5% (press most shaken); 28.0% -> 3.8% (all shaken alike)\",144 validation images,simulated motion blur,"
         "Sec. pi5_third; results/ocr_bench/best_frame.json,current\n")


def edit(p, pairs):
    s = p.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in s else "\n"
    s = s.replace("\r\n", "\n")
    for a, b in pairs:
        assert s.count(a) == 1, (p.name, s.count(a), a[:70])
        s = s.replace(a, b)
    p.write_bytes(s.replace("\n", nl).encode("utf-8"))


for path, pairs in EDITS.items():
    edit(path, pairs)

# results table: new rows after P05
md = PKG / "02_RESULTS_MASTER_TABLE.md"
s = md.read_bytes().decode("utf-8").replace("\r\n", "\n")
i = s.index("| P05 |"); j = s.index("\n", i) + 1
md.write_bytes((s[:j] + P_MD + s[j:]).encode("utf-8"))
csv = PKG / "02_RESULTS_MASTER_TABLE.csv"
s = csv.read_bytes().decode("utf-8").replace("\r\n", "\n")
i = s.index("\nP05,") + 1; j = s.index("\n", i) + 1
csv.write_bytes((s[:j] + P_CSV + s[j:]).encode("utf-8"))
print("ok")
