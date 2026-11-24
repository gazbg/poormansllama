#!/bin/bash
# Host-only unit test for SDD D30: the GPU selection of setup.sh (menu logic, --gpus, the emitted card flags).
# No GPU, no model, no network: nvidia-smi is replaced by SETUP_WIZARD_GPUS (lines "index|name|total MiB|free MiB").
# Same shape as test-d24.sh: cut setup.sh at the '# --- main' marker, source the function part, check the pure functions,
# then run the whole wizard once per case with --dry-run and read the emitted command.
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
set +e   # the sourced part sets -u only; failures below are tested explicitly

# three 8 GiB cards: 0 and 1 idle, 2 busy (another process holds 6 GiB)
export SETUP_WIZARD_GPUS=$'0|NVIDIA GeForce GTX 1070|8192|8150\n1|NVIDIA GeForce GTX 1070|8192|8150\n2|NVIDIA GeForce GTX 1070|8192|2100'

check 'need free 8g'   7864  "$(gpu_need_free 8g)"
check 'need free 16g'  9830  "$(gpu_need_free 16g)"
check 'need free full' 21626 "$(gpu_need_free full)"

sel() {  # sel INPUT VRAM -> "GPU_SEL/GPU_COUNT" or FAIL
    GPU_SEL=''; GPU_COUNT=1
    if gpu_parse_selection "$1" "$2" 2>/dev/null; then echo "$GPU_SEL/$GPU_COUNT"; else echo FAIL; fi
}
check 'one card (0)'               '0/1'   "$(sel 0 8g)"
check 'one card (CUDA1)'           '1/1'   "$(sel CUDA1 8g)"
check 'two cards (0,1)'            '0,1/2' "$(sel 0,1 8g)"
check 'two cards, spaces, reversed' '0,1/2' "$(sel '1 0' 8g)"
check 'two cards, CUDA names'      '0,1/2' "$(sel 'CUDA0,CUDA1' 8g)"
check 'duplicates collapse'        '0/1'   "$(sel '0 0' 8g)"
check 'all = the cards with room'  '0,1/2' "$(sel all 8g)"
check 'ALL (upper case)'           '0,1/2' "$(sel ALL 8g)"
check 'card with too little free memory refused' FAIL "$(sel 2 8g)"
check 'a good and a full card refused'           FAIL "$(sel 0,2 8g)"
check 'no such card'               FAIL "$(sel 7 8g)"
check 'not a number'               FAIL "$(sel x 8g)"
check 'empty input'                FAIL "$(sel '' 8g)"

# select_gpus: --gpus, --device, the R7 limit
run_select() {  # run_select RID GPUS DEVICE -> "GPU_SEL/GPU_COUNT/DEVICE/rc"
    ( R_ID="$1"; GPUS="$2"; DEVICE="$3"; GPU_SEL=''; GPU_COUNT=1; NI=1; select_gpus 8g >/dev/null 2>&1; rc=$?; echo "$GPU_SEL/$GPU_COUNT/$DEVICE/$rc" )
}
check '--gpus 0,1 on R7'            '0,1/2/0/0' "$(run_select R7 0,1 '')"
check '--gpus 1 on R7'              '1/1/1/0'   "$(run_select R7 1 '')"
( R_ID=R5; GPUS=0,1; DEVICE=''; GPU_SEL=''; GPU_COUNT=1; NI=1; select_gpus 8g ) >/dev/null 2>&1; check '--gpus 0,1 on R5 refused (exit 2)' 2 "$?"
check '--device wins when no --gpus' '/1/2/0'    "$(run_select R7 '' 2)"
check 'non-interactive, nothing given: automatic pick unchanged' '/1//0' "$(run_select R7 '' '')"

# the card part of the command line
check 'flags, one card'  '--device CUDA1'     "$( DEVICE=1; GPU_COUNT=1; VRAM=8g; gpu_run_flags )"
check 'flags, two cards' '--gpu-instances 2'  "$( DEVICE=0; GPU_COUNT=2; VRAM=8g; gpu_run_flags )"

# the whole wizard, dry run: the emitted command
emit() {  # emit ARGS... -> the run-server.sh text of a dry run
    bash "$SETUP" --dry-run --non-interactive --model qwen36 --quant q5k_xl --context 262k --skip-build --skip-model --no-start --no-vision --clone-dir "$REPO" "$@" 2>&1
}
E2=$(emit --vram 8g --gpus 0,1)
case "$E2" in *'--gpu-instances 2'*) check 'wizard, 2 cards: --gpu-instances 2' yes yes ;; *) check 'wizard, 2 cards: --gpu-instances 2' yes no ;; esac
case "$E2" in *'export CUDA_VISIBLE_DEVICES=0,1'*) check 'wizard, 2 cards: CUDA_VISIBLE_DEVICES=0,1' yes yes ;; *) check 'wizard, 2 cards: CUDA_VISIBLE_DEVICES=0,1' yes no ;; esac
case "$E2" in *'export CUDA_DEVICE_ORDER=PCI_BUS_ID'*) check 'wizard, 2 cards: PCI bus order' yes yes ;; *) check 'wizard, 2 cards: PCI bus order' yes no ;; esac
case "$E2" in *' --device CUDA'*) check 'wizard, 2 cards: no --device' no yes ;; *) check 'wizard, 2 cards: no --device' no no ;; esac
case "$E2" in *'--lm-prefill-free-sched'*) check 'wizard, 2 cards: --lm-prefill-free-sched' yes yes ;; *) check 'wizard, 2 cards: --lm-prefill-free-sched' yes no ;; esac
E1=$(emit --vram 8g --gpus 1)
case "$E1" in *'--device CUDA1'*) check 'wizard, card 1: --device CUDA1' yes yes ;; *) check 'wizard, card 1: --device CUDA1' yes no ;; esac
case "$E1" in *'--gpu-instances'*) check 'wizard, one card: no --gpu-instances' no yes ;; *) check 'wizard, one card: no --gpu-instances' no no ;; esac
case "$E1" in *'CUDA_VISIBLE_DEVICES'*) check 'wizard, one card: no CUDA_VISIBLE_DEVICES' no yes ;; *) check 'wizard, one card: no CUDA_VISIBLE_DEVICES' no no ;; esac
EA=$(emit --vram 8g)
case "$EA" in *'--device CUDA0'*) check 'wizard, automatic: --device CUDA0 as before' yes yes ;; *) check 'wizard, automatic: --device CUDA0 as before' yes no ;; esac
emit --vram 8g --gpus 0,2 >/dev/null 2>&1; check 'wizard, busy card refused (exit 2)' 2 "$?"

echo
if [ "$FAIL" -eq 0 ]; then echo "D30 sh: all $PASS checks passed"; exit 0; else echo "D30 sh: $FAIL of $((PASS + FAIL)) check(s) FAILED"; exit 1; fi
