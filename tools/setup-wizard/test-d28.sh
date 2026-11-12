#!/usr/bin/env bash
# tools/setup-wizard/test-d28.sh
# Host-only test for SDD D28 in setup.sh (no GPU, no build, no model, no download): the emitted run-server.sh carries a
# default cap of 16000 new tokens per response (-n), the MAX_TOKENS variable at the top, and a notice line printed just
# before the server starts that tells the user to edit it.
#
# Run: bash tools/setup-wizard/test-d28.sh
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
check() { N=$((N+1)); if [ "$2" = 0 ]; then echo "  PASS $1"; else echo "  FAIL $1"; FAIL=$((FAIL+1)); fi; }
count() { printf '%s' "$1" | grep -o -F -- "$2" | wc -l | tr -d ' '; }

# --- has_predict_flag (pure)
has_predict_flag '-c 262144 -ctk q8_0 -n 4000 -fa on';   check 'has_predict_flag: -n 4000'          $?
has_predict_flag '-c 1 --n-predict 4000';                 check 'has_predict_flag: --n-predict 4000' $?
has_predict_flag '-c 1 --predict=4000';                   check 'has_predict_flag: --predict=4000'   $?
has_predict_flag '-c 1 -n=4000';                          check 'has_predict_flag: -n=4000'          $?
! has_predict_flag '-c 262144 -np 1 -ngl 99 -ncmoe 37 --no-warmup -t 16 -nkvo';   check 'has_predict_flag: -np / -ngl / -ncmoe / --no-warmup / -nkvo are not -n' $?
! has_predict_flag '';                                    check 'has_predict_flag: empty flags'      $?

# --- every kind of row: the emitted script (dry run prints it)
CASES=(
    'qwen38|16g|262k|q4k_m'
    'qwen38|8g|128k|q4k_m'
    'flint|full|262k|q5k_xl'
    'qwen36|8g|262k|q4k_m'
    'qwen36|16g|262k|q8k_xl'
    'apex35|full|262k|q5k_m'
)
for cs in "${CASES[@]}"; do
    IFS='|' read -r cm cv cc cq <<< "$cs"
    tag="$cm/$cq at $cv"
    OUT=$(bash "$SETUP" --dry-run --non-interactive --model "$cm" --vram "$cv" --context "$cc" --quant "$cq" --skip-build --model-dir "${TMPDIR:-/tmp}/swz-d28-models" --port 8090 2>&1); RC=$?
    [ "$RC" = 0 ]; check "$tag: dry run exit 0" $?
    [ "$(count "$OUT" 'MAX_TOKENS=16000')" = 1 ]; check "$tag: MAX_TOKENS=16000 defined once at the top of the script" $?
    LAUNCH=$(printf '%s\n' "$OUT" | grep -E '^\s+"\$EXE" ' | head -5)
    [ "$(printf '%s\n' "$LAUNCH" | grep -c .)" = 1 ] && [ "$(count "$LAUNCH" '-n "$MAX_TOKENS"')" = 1 ]; check "$tag: one launch line, carrying -n MAX_TOKENS once" $?
    NOTICE_LN=$(printf '%s\n' "$OUT" | grep -n 'echo "max new tokens per response: \$MAX_TOKENS' | head -1 | cut -d: -f1)
    LAUNCH_LN=$(printf '%s\n' "$OUT" | grep -n -E '^\s+"\$EXE" ' | head -1 | cut -d: -f1)
    DEF_LN=$(printf '%s\n' "$OUT" | grep -n 'MAX_TOKENS=16000' | head -1 | cut -d: -f1)
    [ -n "$NOTICE_LN" ] && [ -n "$LAUNCH_LN" ] && [ -n "$DEF_LN" ] && [ "$DEF_LN" -lt "$NOTICE_LN" ] && [ "$NOTICE_LN" -lt "$LAUNCH_LN" ] && [ $((LAUNCH_LN - NOTICE_LN)) -le 2 ]; check "$tag: the definition comes first, the edit notice directly before the launch line" $?
    printf '%s' "$OUT" | grep 'echo "max new tokens' | grep -q 'Edit MAX_TOKENS at the top of this script if you need more'; check "$tag: the notice tells the user to edit MAX_TOKENS" $?
done

echo
if [ "$FAIL" -gt 0 ]; then echo "D28 sh: $FAIL of $N check(s) FAILED"; exit 1; fi
echo "D28 sh: all $N checks passed"
exit 0
