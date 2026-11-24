#!/bin/bash
# Host-only unit test for SDD D31: the web-UI flags of the wizard's cmake configure (setup.sh).
# The prebuilt UI archive of the pinned version is gone from the Hugging Face bucket and the 'latest' archive fails this engine's
# asset check, so the configure must not depend on the download: npm present -> build the UI from source, npm absent -> no embedded UI.
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SETUP="$REPO/setup.sh"
PASS=0; FAIL=0
check() {   # check NAME EXPECTED ACTUAL
    if [ "$2" = "$3" ]; then PASS=$((PASS + 1)); echo "  PASS $1"; else FAIL=$((FAIL + 1)); echo "  FAIL $1 (expected '$2', got '$3')"; fi
}

CUT_AT=$(grep -n '^# --- main' "$SETUP" | head -1 | cut -d: -f1)
[ -n "$CUT_AT" ] || { echo '[FAIL] main marker not found in setup.sh'; exit 1; }
CUT_FILE=$(mktemp)
head -n $((CUT_AT - 1)) "$SETUP" > "$CUT_FILE"
# shellcheck disable=SC1090
source "$CUT_FILE"
rm -f "$CUT_FILE"
set +e

# a PATH with a fake npm, and one without
FAKE=$(mktemp -d); printf '#!/bin/sh\nexit 0\n' > "$FAKE/npm"; chmod +x "$FAKE/npm"
NONE=$(mktemp -d)

check 'npm present: build the UI from source'  '-DLLAMA_BUILD_UI=ON -DLLAMA_USE_PREBUILT_UI=OFF' "$( unset SETUP_WIZARD_NO_NPM; PATH="$FAKE:$PATH" ui_cmake_flags )"
check 'npm absent: no embedded UI, no download' '-DLLAMA_USE_PREBUILT_UI=OFF'                      "$( unset SETUP_WIZARD_NO_NPM; PATH="$NONE" ui_cmake_flags )"
check 'SETUP_WIZARD_NO_NPM simulates no npm'    '-DLLAMA_USE_PREBUILT_UI=OFF'                      "$( SETUP_WIZARD_NO_NPM=1 PATH="$FAKE:$PATH" ui_cmake_flags )"
# the download is never allowed in either case
for v in '' 1; do
    f=$( if [ -n "$v" ]; then export SETUP_WIZARD_NO_NPM=1; fi; PATH="$FAKE:$PATH" ui_cmake_flags )
    case "$f" in *'LLAMA_USE_PREBUILT_UI=OFF'*) check "no prebuilt download (no-npm='$v')" yes yes ;; *) check "no prebuilt download (no-npm='$v')" yes no ;; esac
done

# the note the user sees
n1=$( unset SETUP_WIZARD_NO_NPM; PATH="$FAKE:$PATH" ui_cmake_note 2>&1 )
case "$n1" in *'built from this tree'*) check 'note with npm' yes yes ;; *) check 'note with npm' yes no ;; esac
n2=$( SETUP_WIZARD_NO_NPM=1 ui_cmake_note 2>&1 )
case "$n2" in *'WITHOUT the embedded web UI'*) check 'warning without npm' yes yes ;; *) check 'warning without npm' yes no ;; esac

# the dry run shows the flags in the configure command
D=$(bash "$SETUP" --dry-run --non-interactive --model qwen36 --quant q5k_xl --vram 8g --context 262k --skip-model --no-start --no-vision --full-build --clone-dir "$REPO" 2>&1)
case "$D" in *'-DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_UI=ON -DLLAMA_USE_PREBUILT_UI=OFF)'*|*'-DCMAKE_BUILD_TYPE=Release -DLLAMA_USE_PREBUILT_UI=OFF)'*) check 'dry-run configure line carries the UI flags' yes yes ;; *) check 'dry-run configure line carries the UI flags' yes no ;; esac

rm -rf "$FAKE" "$NONE"
echo
if [ "$FAIL" -eq 0 ]; then echo "D31 sh: all $PASS checks passed"; exit 0; else echo "D31 sh: $FAIL of $((PASS + FAIL)) check(s) FAILED"; exit 1; fi
