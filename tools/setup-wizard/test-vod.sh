#!/usr/bin/env bash
# tools/setup-wizard/test-vod.sh
# Host-only test for the vision additions of setup.sh (VOD-6, widened by SDD D27; no GPU, no build, no model, no
# download): the projector columns of the model table, vision_offered, the vision append of
# complete_row_flags, and the dry-run emission with vision on / off / refused, for every model entry.
#
# Run: bash tools/setup-wizard/test-vod.sh
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SETUP="$REPO/setup.sh"

CUT_AT=$(grep -n '^# --- main' "$SETUP" | head -1 | cut -d: -f1)
[ -n "$CUT_AT" ] || { echo '[FAIL] main marker not found in setup.sh'; exit 1; }
CUT_FILE=$(mktemp)
head -n $((CUT_AT - 1)) "$SETUP" | tr -d '\r' > "$CUT_FILE"
# shellcheck disable=SC1090
source "$CUT_FILE"
rm -f "$CUT_FILE"

FAIL=0
N=0
check() { # check WHAT STATUS(0/1)
    N=$((N+1))
    if [ "$2" = 0 ]; then echo "  PASS $1"; else echo "  FAIL $1"; FAIL=$((FAIL+1)); fi
}
count() { # count TEXT FIXED-STRING -> occurrences
    printf '%s' "$1" | grep -o -F -- "$2" | wc -l | tr -d ' '
}

# --- the table: indexes 0..6 = qwen38 q4k_m, qwen36 q4k_m, flint q5k_xl, qwen38 q5k_xl, qwen36 q5k_xl, qwen36 q8k_xl, apex35 q5k_m
for i in 0 1 2 3 4 5 6; do
    k="${M_KEYS[$i]}"; q="${M_QUANTS[$i]}"
    case "$k" in
        qwen36)
            [ "${M_MMPROJ[$i]}" = 'mmproj-BF16.gguf' ]; check "$k/$q lists mmproj-BF16.gguf" $?
            want=902822624
            [ "${M_REPOS[$i]}" = 'unsloth/Qwen3.6-35B-A3B-MTP-GGUF' ] && want=902822528
            [ "${M_MMPROJ_BYTES[$i]}" = "$want" ]; check "$k/$q projector size matches its repo (${M_REPOS[$i]})" $?
            [ -z "${M_MMPROJ_REPOS[$i]}" ]; check "$k/$q projector comes from the model's own repo" $? ;;
        qwen38)
            [ "${M_MMPROJ[$i]}" = 'mmproj-BF16.gguf' ] && [ "${M_MMPROJ_BYTES[$i]}" = 931146432 ] && [ -z "${M_MMPROJ_REPOS[$i]}" ] && [ "${M_REPOS[$i]}" = 'unsloth/Qwen3.8-27B-GGUF' ]; check "$k/$q lists mmproj-BF16.gguf (unsloth/Qwen3.8-27B-GGUF)" $? ;;
        apex35)
            [ "${M_MMPROJ[$i]}" = 'mmproj-Apexintelligence-AI_Apex-35B-V1.0-bf16.gguf' ] && [ "${M_MMPROJ_BYTES[$i]}" = 902822560 ] && [ "${M_MMPROJ_REPOS[$i]}" = 'bartowski/Apexintelligence-AI_Apex-35B-V1.0-GGUF' ]; check "$k/$q lists the Apex projector from the bartowski repo (the model's own repo publishes none)" $? ;;
        *)
            [ -z "${M_MMPROJ[$i]}" ]; check "$k/$q lists no projector (text-only model)" $? ;;
    esac
done
[ "${#M_MMPROJ[@]}" = "${#M_KEYS[@]}" ] && [ "${#M_MMPROJ_BYTES[@]}" = "${#M_KEYS[@]}" ] && [ "${#M_MMPROJ_REPOS[@]}" = "${#M_KEYS[@]}" ]; check 'the three projector arrays have one element per model entry' $?

# --- vision_offered over every recipe row
ROWS='qwen38 full 262k q4k_m qwen38 16g 262k q4k_m qwen38 8g 128k q4k_m flint full 262k q5k_xl qwen36 full 262k q8k_xl qwen36 16g 262k q8k_xl qwen36 8g 262k q4k_m qwen36 16g 262k q4k_m apex35 full 262k q5k_m'
i=0
while [ $i -lt 9 ]; do
    m=$(echo "$ROWS" | cut -d' ' -f$((i*4+1)))
    v=$(echo "$ROWS" | cut -d' ' -f$((i*4+2)))
    c=$(echo "$ROWS" | cut -d' ' -f$((i*4+3)))
    q=$(echo "$ROWS" | cut -d' ' -f$((i*4+4)))
    recipe_exact "$m" "$v" "$c" "$q" || { echo "[FAIL] row $((i+1)) not found"; exit 1; }
    mi "$m" "$q"
    want=0    # SDD D27: every model with a projector, the MTP / draft rows included
    [ "$m" = flint ] && want=1
    vision_offered; got=$?
    [ "$got" = "$want" ]; check "$R_ID ($m/$q) vision offered = $([ $want = 0 ] && echo yes || echo no)" $?

    for cc in 12.0 6.1 ''; do
        OFF=$(complete_row_flags "$R_FLAGS" "$R_ENV" "$cc" 0 2>/dev/null)
        ON=$(complete_row_flags "$R_FLAGS" "$R_ENV" "$cc" 1 2>/dev/null)
        [ "$(count "$OFF" '--mmproj')" = 0 ]; check "$R_ID cc='$cc' vision off: no vision flag" $?
        [ "$(count "$ON" '--mmproj "$MMPROJ" --mmproj-on-demand')" = 1 ] && [ "$(count "$ON" '--mmproj')" = 2 ]; check "$R_ID cc='$cc' vision on: --mmproj \"\$MMPROJ\" --mmproj-on-demand exactly once" $?
        [ "$ON" = "$OFF"' --mmproj "$MMPROJ" --mmproj-on-demand' ]; check "$R_ID cc='$cc' vision on: the rest equals the vision-off flags" $?
        AGAIN=$(complete_row_flags "$ON" "$R_ENV" "$cc" 1 2>/dev/null)
        [ "$AGAIN" = "$ON" ]; check "$R_ID cc='$cc' idempotent with vision" $?
    done
    i=$((i+1))
done

# --- dry-run end to end (setup.sh itself; --dry-run starts nothing, downloads nothing)
DR_COMMON=(--dry-run --non-interactive --model qwen36 --vram 16g --context 262k --skip-build --model-dir "${TMPDIR:-/tmp}/swz-vod-models" --port 8090)
OUT=$(bash "$SETUP" "${DR_COMMON[@]}" 2>&1); RC=$?
[ "$RC" = 0 ]; check 'dry run (default): exit 0' $?
printf '%s' "$OUT" | grep -q 'MMPROJ=".*mmproj-BF16\.gguf"'; check 'dry run (default): the emitted script defines MMPROJ' $?
LAUNCH=$(printf '%s\n' "$OUT" | grep -E '^\s+"\$EXE" ' | head -5)
[ "$(printf '%s\n' "$LAUNCH" | grep -c .)" = 1 ] && [ "$(count "$LAUNCH" '--mmproj "$MMPROJ" --mmproj-on-demand')" = 1 ]; check 'dry run (default): one launch line, carrying the vision flags once' $?
printf '%s' "$OUT" | grep -q 'projector mmproj-BF16.gguf from unsloth/Qwen3.6-35B-A3B-GGUF'; check 'dry run (default): the projector download is announced from the model repo' $?
OUT=$(bash "$SETUP" "${DR_COMMON[@]}" --no-vision 2>&1); RC=$?
[ "$RC" = 0 ] && [ "$(count "$OUT" '--mmproj')" = 0 ] && ! printf '%s' "$OUT" | grep -q 'mmproj-BF16'; check 'dry run --no-vision: exit 0, no vision flag, no projector line' $?
bash "$SETUP" "${DR_COMMON[@]}" --mmproj-file "${TMPDIR:-/tmp}/swz-vod-missing.gguf" >/dev/null 2>&1; RC=$?
[ "$RC" = 1 ]; check 'dry run --mmproj-file with a missing file: exit 1' $?
FAKE="${TMPDIR:-/tmp}/swz-vod-fake-mmproj.gguf"; echo x > "$FAKE"
OUT=$(bash "$SETUP" "${DR_COMMON[@]}" --mmproj-file "$FAKE" 2>&1); RC=$?
[ "$RC" = 0 ] && printf '%s' "$OUT" | grep -q -F "$FAKE" && ! printf '%s' "$OUT" | grep -q 'not downloading the projector'; check 'dry run --mmproj-file: the local file is used, nothing is downloaded' $?
bash "$SETUP" "${DR_COMMON[@]}" --mmproj-file "$FAKE" --no-vision >/dev/null 2>&1; RC=$?
[ "$RC" = 1 ]; check 'dry run --mmproj-file with --no-vision: exit 1 (contradiction)' $?
rm -f "$FAKE"
# SDD D27: every model entry with a projector, default (vision on) and --no-vision; the MTP row R5 and the dense rows included
dr() { # dr MODEL VRAM CTX QUANT [extra...] -> OUT, RC
    local m="$1" v="$2" c="$3" q="$4"; shift 4
    local qa=(); [ -n "$q" ] && qa=(--quant "$q")
    OUT=$(bash "$SETUP" --dry-run --non-interactive --model "$m" --vram "$v" --context "$c" ${qa[@]+"${qa[@]}"} --skip-build --model-dir "${TMPDIR:-/tmp}/swz-vod-models" --port 8090 "$@" 2>&1); RC=$?
}
CASES=(
    'qwen38|16g|262k|q4k_m|mmproj-BF16.gguf|unsloth/Qwen3.8-27B-GGUF'
    'qwen38|full|262k|q5k_xl|mmproj-BF16.gguf|unsloth/Qwen3.8-27B-GGUF'
    'qwen36|full|262k|q8k_xl|mmproj-BF16.gguf|unsloth/Qwen3.6-35B-A3B-MTP-GGUF'
    'apex35|full|262k|q5k_m|mmproj-Apexintelligence-AI_Apex-35B-V1.0-bf16.gguf|bartowski/Apexintelligence-AI_Apex-35B-V1.0-GGUF'
)
for cs in "${CASES[@]}"; do
    IFS='|' read -r cm cv cc cq cf cr <<< "$cs"
    tag="$cm/$cq at $cv"
    dr "$cm" "$cv" "$cc" "$cq"
    LAUNCH=$(printf '%s\n' "$OUT" | grep -E '^\s+"\$EXE" ' | head -5)
    [ "$RC" = 0 ] && [ "$(printf '%s\n' "$LAUNCH" | grep -c .)" = 1 ] && [ "$(count "$LAUNCH" '--mmproj "$MMPROJ" --mmproj-on-demand')" = 1 ]; check "dry run $tag (default): exit 0, one launch line carrying the vision flags once" $?
    printf '%s' "$OUT" | grep -q -F "projector $cf from $cr"; check "dry run $tag (default): the download is announced as $cf from $cr" $?
    dr "$cm" "$cv" "$cc" "$cq" --no-vision
    [ "$RC" = 0 ] && [ "$(count "$OUT" '--mmproj')" = 0 ] && ! printf '%s' "$OUT" | grep -q 'projector'; check "dry run $tag --no-vision: exit 0, no vision flag, no projector line" $?
done
# the emitted script says when --mmproj-on-demand is not a measured combination (Qwen3.6-35B-A3B and its Apex fine-tune are)
dr qwen38 16g 262k q4k_m;  printf '%s' "$OUT" | grep -q 'vision on this model is NOT measured'; check 'dry run qwen38: the emitted script says that vision is not measured on this model' $?
dr qwen36 16g 262k q4k_m;  ! printf '%s' "$OUT" | grep -q 'vision on this model is NOT measured'; check 'dry run qwen36: no "not measured" line' $?
dr apex35 full 262k q5k_m; ! printf '%s' "$OUT" | grep -q 'vision on this model is NOT measured'; check 'dry run apex35: no "not measured" line (same architecture as Qwen3.6-35B-A3B)' $?
# flint is text-only: no projector is published anywhere
dr flint full 262k q5k_xl
[ "$RC" = 0 ] && printf '%s' "$OUT" | grep -q 'vision: not available for flint' && [ "$(count "$OUT" '--mmproj "$MMPROJ"')" = 0 ] && [ "$(count "$OUT" '--mmproj-on-demand')" = 0 ]; check 'dry run flint: exit 0, "vision: not available", no vision flag' $?
dr flint full 262k q5k_xl --mmproj-file "${TMPDIR:-/tmp}/swz-vod-missing.gguf"
[ "$RC" = 1 ]; check 'dry run flint --mmproj-file: exit 1' $?

echo
if [ "$FAIL" -gt 0 ]; then echo "VOD sh: $FAIL of $N check(s) FAILED"; exit 1; fi
echo "VOD sh: all $N checks passed"
exit 0
