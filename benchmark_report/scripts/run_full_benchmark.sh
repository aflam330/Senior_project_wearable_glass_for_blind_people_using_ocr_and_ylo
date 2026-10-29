#!/bin/bash
# The whole Raspberry Pi 5 measurement in one go, logged for a dated report in benchmark_report/<date>/.
#
#   bash benchmark_report/scripts/run_full_benchmark.sh        # about 1 h 45 min
#
# Steps, with the sampler (temperature, throttle flags, clock, CPU, memory, board power every 2 s) running throughout:
#   idle 2 min; preflight; the hardware-free tests; scripts/benchmark_pi5.py --iters 100 --sustained 30 (the protocol
#   in HOW_TO_RUN.md); currency mode switch; extras_bench.py (emotion with a face, both object detectors, speech,
#   camera); 60 s of real webcam frames through the modes; the OCR response test for the four detector / read-width
#   settings of scripts/pi_check_all.sh; idle 3 min.
# Between the heavy steps it waits until the board is back under COOL_C (at most 10 min), so each starts alike.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GLASS="$REPO/savior_glass"
PY="${PY:-$HOME/.venvs/savior_glass/bin/python}"
RUN="$REPO/benchmark_report/$(date +%Y-%m-%d)"
OUT="$RUN/raw"
COOL_C="${COOL_C:-63}"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
mkdir -p "$OUT/tests"
PHASE="$OUT/phase.txt"
STEPS="$OUT/steps.csv"
TESTS="$OUT/tests.csv"

temp_c() { awk '{printf "%.1f", $1 / 1000}' /sys/class/thermal/thermal_zone0/temp; }
stamp() { while IFS= read -r l; do printf '%s %s\n' "$EPOCHREALTIME" "$l"; done; }

{
    echo "date: $(date -Is)"; uname -a; cat /proc/device-tree/model | tr -d '\0'; echo
    vcgencmd version; vcgencmd get_throttled; vcgencmd measure_temp
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
    echo "max freq: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq) kHz"
    free -m; df -h "$REPO" /
    echo "--- top processes at start"; ps -eo pcpu,pmem,rss,comm --sort=-pcpu | head -15
    echo "--- python packages"; "$PY" -m pip list 2>/dev/null | grep -iE "^(torch|onnxruntime|opencv|easyocr|ultralytics|numpy|psutil) "
    echo "--- camera"; v4l2-ctl --list-devices 2>/dev/null
    echo "--- git"; git -C "$REPO" log -1 --format='%H %cd %s'; git -C "$REPO" status --short
} > "$OUT/environment.txt" 2>&1

echo idle_before > "$PHASE"
"$PY" "$REPO/benchmark_report/scripts/sys_sampler.py" "$OUT/system_timeline.csv" "$PHASE" &
SAMPLER=$!
trap 'kill -TERM $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null' EXIT
echo "step,start_epoch,end_epoch,exit_code,temp_before_c,temp_after_c" > "$STEPS"
echo "test,exit_code,seconds" > "$TESTS"

# step <name> <command...>: output to raw/<name>.txt with a time stamp on every line, one row in steps.csv
step() {
    local name="$1"; shift
    local t0 tb rc
    t0="$EPOCHREALTIME"; tb="$(temp_c)"
    echo "$name" > "$PHASE"
    echo "== $name  ($(date +%H:%M:%S), $tb C)"
    "$@" 2>&1 | stamp > "$OUT/$name.txt"
    rc=${PIPESTATUS[0]}
    echo "$name,$t0,$EPOCHREALTIME,$rc,$tb,$(temp_c)" >> "$STEPS"
    echo "   exit $rc, $(temp_c) C"
    echo idle > "$PHASE"
}

cool() {
    local t0="$EPOCHREALTIME" waited=0
    echo cooldown > "$PHASE"
    while awk -v t="$(temp_c)" -v c="$COOL_C" 'BEGIN { exit !(t > c) }' && [ $waited -lt 600 ]; do
        sleep 10; waited=$((waited + 10))
    done
    echo "cooldown,$t0,$EPOCHREALTIME,0,,$(temp_c)" >> "$STEPS"
    echo "   cooled for ${waited}s, now $(temp_c) C"
}

cd "$GLASS"
step idle_before sleep 120
step preflight "$PY" scripts/pi5_preflight.py

echo tests > "$PHASE"
t0="$EPOCHREALTIME"; tb="$(temp_c)"
echo "== tests  ($(date +%H:%M:%S), $tb C)"
for t in scripts/test_thermal.py scripts/test_multi_note.py scripts/test_capture_guide.py scripts/test_field_log.py \
         scripts/test_tts_queue.py scripts/test_uv_check.py scripts/test_voice_commands.py scripts/test_buttons_haptics.py \
         scripts/test_live_diag.py scripts/test_ocr_single_pass.py \
         ../research_results/test_hooks.py ../research_results/test_results_db.py \
         ../realtime_bangla_taka_detection/scripts/test_roboeye_modules.py \
         ../realtime_bangla_taka_detection/scripts/dataset/test_print_capture.py; do
    name="$(basename "$t" .py)"
    s="$EPOCHREALTIME"
    timeout 900 "$PY" "$t" > "$OUT/tests/$name.txt" 2>&1
    rc=$?
    echo "$name,$rc,$(awk -v a="$s" -v b="$EPOCHREALTIME" 'BEGIN { printf "%.1f", b - a }')" >> "$TESTS"
    echo "   $name exit $rc"
done
echo "tests,$t0,$EPOCHREALTIME,0,$tb,$(temp_c)" >> "$STEPS"

cool
# the per-benchmark phase comes from the script's own "timing <name>" lines
t0="$EPOCHREALTIME"; tb="$(temp_c)"
echo "== main_benchmark  ($(date +%H:%M:%S), $tb C)"
"$PY" scripts/benchmark_pi5.py --iters 100 --warmup 3 --sustained 30 2>&1 | while IFS= read -r l; do
    case "$l" in
        "timing "*) n="${l#timing }"; echo "main_benchmark_${n%% *}" > "$PHASE" ;;
        "sustained run"*) echo main_sustained > "$PHASE" ;;
    esac
    printf '%s %s\n' "$EPOCHREALTIME" "$l"
done > "$OUT/main_benchmark.txt"
rc=${PIPESTATUS[0]}
echo "main_benchmark,$t0,$EPOCHREALTIME,$rc,$tb,$(temp_c)" >> "$STEPS"
echo "   exit $rc, $(temp_c) C"
latest="$(ls -t results/pi5_benchmark_*.json | head -1)"
cp "$latest" "$OUT/"; cp "${latest%.json}.csv" "$OUT/" 2>/dev/null

cool
step mode_switch "$PY" scripts/measure_mode_switch.py "$OUT/mode_switch_currency.json"
step extras "$PY" "$REPO/benchmark_report/scripts/extras_bench.py" "$OUT/extras.json" "$PHASE"
step live_camera "$PY" scripts/test_live_camera.py --seconds 60
cp results/live_camera_test.json "$OUT/" 2>/dev/null

for cfg in "onnx_int8 1200" "onnx 1200" "onnx_int8 880" "onnx 880"; do
    set -- $cfg
    cool
    step "ocr_$1_$2" env OCR_DETECTOR="$1" OCR_READ_WIDTH="$2" "$PY" "$REPO/benchmark_report/scripts/ocr_response_bench.py" \
        "$OUT/ocr_response_$1_$2.json" "$PHASE"
done

step idle_after sleep 180
echo "Done: $OUT"
