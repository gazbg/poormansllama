#!/usr/bin/env bash
# tools/setup-wizard/test-d24.sh
# Host-only unit test for SDD D24: every MoE model shares the 8g profile (R7), whatever its name and
# whether or not it has a benchmark. No GPU, no build, no model: cut setup.sh at the '# --- main'
# marker, source the function part, and check recipe_for / is_moe_model against a matrix.
#
# Run: bash tools/setup-wizard/test-d24.sh
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
TOTAL=0
check() { # check WHAT STATUS(0/1)
    TOTAL=$((TOTAL+1))
    if [ "$2" = 0 ]; then echo "  PASS $1"; else echo "  FAIL $1"; FAIL=$((FAIL+1)); fi
}
kase() { # kase MODEL QUANT VRAM CTX WANT_ID
    local got=none
    R_ID=''
    if recipe_for "$1" "$3" "$4" "$2" 2>/dev/null; then got="$R_ID"; fi
    [ "$got" = "$5" ]; check "$1/$2 $3 $4 -> $5 (got $got)" $?
}

# which models are MoE
is_moe_model qwen38; [ $? -ne 0 ]; check 'qwen38 is not MoE' $?
is_moe_model qwen36; check 'qwen36 is MoE' $?
is_moe_model flint;  check 'flint is MoE' $?
is_moe_model apex35; check 'apex35 is MoE' $?
is_moe_model nosuch; [ $? -ne 0 ]; check 'unknown model is not MoE' $?

# 8g, 262k: every MoE model gets R7 (exact, default-quant fallback, or the shared profile)
kase qwen36 q4k_m  8g 262k R7
kase qwen36 q5k_xl 8g 262k R7
kase qwen36 q8k_xl 8g 262k R7
kase flint  q5k_xl 8g 262k R7
kase apex35 q5k_m  8g 262k R7
kase apex35 q6k_l  8g 262k R7

# the profile is the same flag text for every MoE model
recipe_for qwen36 8g 262k q4k_m >/dev/null 2>&1; A="$R_FLAGS"
recipe_for flint  8g 262k q5k_xl >/dev/null 2>&1; B="$R_FLAGS"
recipe_for apex35 8g 262k q6k_l >/dev/null 2>&1; C="$R_FLAGS"
[ -n "$A" ] && [ "$A" = "$B" ] && [ "$B" = "$C" ]; check 'all MoE models at 8g/262k get the identical flag text' $?

# the shared profile flags itself, so the caller can warn (and a per-model row does not)
recipe_for apex35 8g 262k q6k_l >/dev/null 2>&1; [ "$R_CLASS_FALLBACK" = 1 ]; check 'apex35 8g: class-profile flag set' $?
recipe_for qwen36 8g 262k q4k_m >/dev/null 2>&1; [ "$R_CLASS_FALLBACK" = 0 ]; check 'qwen36 q4k_m 8g: exact row, no class flag' $?

# not widened: other targets and contexts keep their rows or refuse
kase apex35 q5k_m  8g   128k none
kase flint  q5k_xl 8g   128k none
kase apex35 q5k_m  16g  262k none
kase flint  q5k_xl 16g  262k none
kase apex35 q5k_m  full 262k R9
kase flint  q5k_xl full 262k R4
kase qwen36 q4k_m  16g  262k R8
kase qwen36 q8k_xl full 262k R5

# dense models are not MoE: their 8g row stays R3 (128k only)
kase qwen38 q4k_m  8g 128k R3
kase qwen38 q4k_m  8g 262k none
kase qwen38 q5k_xl 8g 262k none

# an unknown model never gets a profile
kase nosuch q4k_m 8g 262k none

echo
if [ "$FAIL" -gt 0 ]; then echo "D24 sh: $FAIL of $TOTAL check(s) FAILED"; exit 1; fi
echo "D24 sh: all $TOTAL checks passed"
exit 0
