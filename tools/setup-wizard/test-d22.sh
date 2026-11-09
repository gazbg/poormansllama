#!/usr/bin/env bash
# tools/setup-wizard/test-d22.sh
# Host-only unit test for the D22 row-flag completion (no GPU, no build,
# no model): cut setup.sh at the '# --- main' marker, source the function
# part, and check complete_row_flags against every recipe row and a matrix
# of compute capabilities.
#
# Run: bash tools/setup-wizard/test-d22.sh
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SETUP="$REPO/setup.sh"

CUT_AT=$(grep -n '^# --- main' "$SETUP" | head -1 | cut -d: -f1)
[ -n "$CUT_AT" ] || { echo '[FAIL] main marker not found in setup.sh'; exit 1; }
CUT_FILE=$(mktemp)
head -n $((CUT_AT - 1)) "$SETUP" > "$CUT_FILE"
# shellcheck disable=SC1090
source "$CUT_FILE"
rm -f "$CUT_FILE"

FAIL=0
check() { # check WHAT STATUS(0/1)
    if [ "$2" = 0 ]; then echo "  PASS $1"; else echo "  FAIL $1"; FAIL=$((FAIL+1)); fi
}
prearm_always() {
    # exactly one --lm-prefill-prearm, and its value is always
    local n v
    n=$(echo "$1" | tr ' ' '\n' | grep -cx -- '--lm-prefill-prearm')
    [ "$n" = 1 ] || return 1
    v=$(echo "$1" | sed -n 's/.*--lm-prefill-prearm \([a-z]*\)\( \|$\).*/\1/p')
    [ "$v" = always ]
}
has_int8() {
    echo "$1" | tr ' ' '\n' | grep -qx -- '--lm-prefill-attn-int8'
}

echo "  main marker at line $CUT_AT"

# iterate over the exact rows like the wizard does: (model, vram, ctx, quant)
ROWS='qwen38 full 262k q4k_m qwen38 16g 262k q4k_m qwen38 8g 128k q4k_m flint full 262k q5k_xl qwen36 full 262k q8k_xl qwen36 16g 262k q8k_xl qwen36 8g 262k q4k_m qwen36 16g 262k q4k_m apex35 full 262k q5k_m'
i=0
while [ $i -lt 9 ]; do
    m=$(echo "$ROWS" | cut -d' ' -f$((i*4+1)))
    v=$(echo "$ROWS" | cut -d' ' -f$((i*4+2)))
    c=$(echo "$ROWS" | cut -d' ' -f$((i*4+3)))
    q=$(echo "$ROWS" | cut -d' ' -f$((i*4+4)))
    recipe_exact "$m" "$v" "$c" "$q" || { echo "[FAIL] row $((i+1)) not found"; exit 1; }
    for cc in 12.0 8.9 6.1 5.2 3.5 ''; do
        R_OUT=$(complete_row_flags "$R_FLAGS" "$R_ENV" "$cc" 2>/dev/null)
        prearm_always "$R_OUT"; check "$R_ID cc='$cc' prearm=always, exactly once" $?
        case "$cc" in
            6.1|5.2|3.5)
                has_int8 "$R_OUT"; check "$R_ID cc='$cc' int8 flag added" $?
                ;;
            *)
                has_int8 "$R_OUT"; [ $? -ne 0 ]; check "$R_ID cc='$cc' no int8 flag" $?
                ;;
        esac
        R_OUT2=$(complete_row_flags "$R_OUT" "$R_ENV" "$cc" 2>/dev/null)
        [ "$R_OUT2" = "$R_OUT" ]; check "$R_ID cc='$cc' idempotent" $?
    done
    i=$((i+1))
done

# the "env already set" branch (AC-3): no current row carries the env, so
# drive the pure function with a synthetic env
R_OUT=$(complete_row_flags "$R_FLAGS" 'LLAMA_FATTN_TILE_Q8QK=1 LLAMA_FATTN_GQA8_MAXROWS=4' 6.1 2>/dev/null)
has_int8 "$R_OUT"; [ $? -ne 0 ]; check 'env-set: no int8 flag appended' $?
prearm_always "$R_OUT"; check 'env-set: prearm still normalized' $?
R_OUT=$(complete_row_flags "$R_FLAGS" 'LLAMA_FATTN_TILE_Q8QK=1' 12.0 2>/dev/null)
prearm_always "$R_OUT"; [ $? -eq 0 ] && { has_int8 "$R_OUT"; [ $? -ne 0 ]; }
check 'env-set + new arch: prearm normalized, no flag' $?

# detect_gpu_arch test hook
SETUP_WIZARD_GPU_CC=6.1
arch=$(detect_gpu_arch)
[ "$arch" = 6.1 ]; check 'detect_gpu_arch honors the hook' $?
unset SETUP_WIZARD_GPU_CC

echo
if [ "$FAIL" -gt 0 ]; then echo "D22 sh: $FAIL check(s) FAILED"; exit 1; fi
echo "D22 sh: all 9x6x4 + 3 checks passed"
exit 0
