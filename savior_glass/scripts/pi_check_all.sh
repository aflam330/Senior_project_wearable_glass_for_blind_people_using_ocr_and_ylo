#!/bin/sh
# One run on the Raspberry Pi 5 that checks the changes of 14 October 2026 and times what the laptop cannot.
#
#   cd savior_glass && . ~/.venvs/savior_glass/bin/activate
#   sh scripts/pi_check_all.sh            # about 40 minutes; let the board cool first and close VS Code
#   sh scripts/pi_check_all.sh quick      # checks and the voice only, a few minutes
#
# Everything is written under benchmark_report/raw/pi_<date>/ ; commit and push that folder afterwards:
#   cd .. && git add benchmark_report savior_glass/results && git commit -m "Pi 5 run" && git push
ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/../benchmark_report/raw/pi_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT"
BENCH="$ROOT/../benchmark_report/scripts/ocr_response_bench.py"
temp() { vcgencmd measure_temp 2>/dev/null || echo "temp=unknown"; }
run() { name="$1"; shift; echo "== $name  ($(temp))"; "$@" > "$OUT/$name.txt" 2>&1; echo "   exit $? -> $OUT/$name.txt"; tail -n 2 "$OUT/$name.txt" | sed 's/^/   /'; }

echo "Start: $(date)  $(temp)" | tee "$OUT/start.txt"
vcgencmd get_throttled 2>/dev/null | tee -a "$OUT/start.txt"

run preflight        python scripts/pi5_preflight.py
run speech_models    python scripts/get_speech_models.py --check
run test_speech      python scripts/test_tts_queue.py
run test_voice_cmd   python scripts/test_voice_commands.py
run test_single_pass python scripts/test_ocr_single_pass.py
# the voice: rendered and timed without sound, with the Piper voices and with espeak-ng
run speech_piper     python scripts/speech_check.py --quiet
cp results/speech_check_*.json "$OUT/speech_check_piper.json" 2>/dev/null
TTS_VOICE=espeak python scripts/speech_check.py --quiet > "$OUT/speech_espeak.txt" 2>&1
cp results/speech_check_*.json "$OUT/speech_check_espeak.json" 2>/dev/null
arecord -l > "$OUT/microphones.txt" 2>&1
aplay -l > "$OUT/sound_devices.txt" 2>&1

if [ "${1:-}" = "quick" ]; then echo "Quick run done: $OUT"; exit 0; fi

# Text reading through the call the app makes on ACTION, for the settings that can only be decided on this board.
# A minute of rest between runs, so one run does not heat the next.
for cfg in "onnx_int8 1200" "onnx 1200" "onnx_int8 880" "onnx 880"; do
    set -- $cfg
    sleep 60
    echo "== text reading: detector $1, read width $2  ($(temp))"
    OCR_DETECTOR="$1" OCR_READ_WIDTH="$2" python "$BENCH" "$OUT/ocr_response_$1_$2.json" > "$OUT/ocr_response_$1_$2.txt" 2>&1
    echo "   exit $? ; $(tail -n 1 "$OUT/ocr_response_$1_$2.txt")"
done
echo "End: $(date)  $(temp)" | tee "$OUT/end.txt"
vcgencmd get_throttled 2>/dev/null | tee -a "$OUT/end.txt"
echo "Done. Results: $OUT"
