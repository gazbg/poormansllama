#!/usr/bin/env bash
# ============================================================================
# setup.sh - poormansllama setup wizard (Linux)
#
# Purpose: check the environment (git, cmake, CUDA toolkit, gcc, NVIDIA GPU),
# auto-install missing toolchain pieces via apt (y/n confirmed), clone the
# repo if not already inside a checkout, then ask ALL settings up front
# (model, quant, local path, VRAM target, context; SDD D20) - the unmeasured-
# combo refusal (exit 3) happens BEFORE any build or download. The build
# (llama-server with GGML_CUDA=ON, LLAMA_LAYER_STREAMING=ON) and the model
# download (Hugging Face, resume-capable) then run IN PARALLEL with one
# combined status line, fail-fast. One of the seven MEASURED recipe rows
# (R1-R7) is emitted as a ready-to-run run-server.sh in the repo root.
# Optionally starts the server.
#
# Mirror of setup.ps1 (Windows). Design: docs/setup-wizard-sdd.md
# Plan + acceptance: docs/plans/setup-wizard-plan.md
#
# Safety: downloads only from huggingface.co; no driver install (a missing
# GPU/driver is fatal with manual guidance); auto-install only behind y/n and
# only via the apt packages in the SDD. The HF token, if given, is never
# printed and never written to any file.
#
# Exit codes: 0 ok / 1 usage / 2 environment / 3 no measured recipe /
# 4 build failure / 5 model download failure / 6 user abort
#
# Test hook (documented, mirrors LLAMA_LM_PEER_FAIL):
#   SETUP_WIZARD_FORCE_MISSING='cmake,git' makes named tools report missing
#   so the install path is testable; it only affects the report, never the
#   install.
# ============================================================================
set -u

# --- options ---------------------------------------------------------------
MODEL=''
VRAM=''
CONTEXT=''
PORT=''   # empty = ask (default 8090); an explicit --port suppresses the question (D22)
BIND=''
DEVICE=''     # CUDA index for device-pinned rows R5/R7; empty = auto (SDD D17)
QUANT=''      # q4k_m|q5k_xl|q8k_xl|q5k_m (SDD D19, D23); empty = the model's default
MODEL_FILE=''
MODEL_DIR=''
CLONE_DIR=''   # empty = in-repo detection / default path (see invoke_clone)
BRANCH='claude/flint-support'
REPO_URL='https://github.com/gazbg/poormansllama.git'
CUDA_ARCH=''
FULL_CLONE=0
FULL_BUILD=0
SKIP_BUILD=0
SKIP_MODEL=0
SKIP_DOWNLOAD=0
NO_START=0
DRY_RUN=0
NI=0
YES_INSTALL=0
HF_TOKEN="${HF_TOKEN:-}"
MAX_DL_MIB=0
# VOD-6 / SDD D27 (docs/plans/vision-on-demand-plan.md): vision support. The projector comes from the model's own
# Hugging Face repo (M_MMPROJ_REPOS names another repo when the model's own publishes none); the emitted command
# gets --mmproj "$MMPROJ" --mmproj-on-demand.
NO_VISION=0
MMPROJ_FILE=''   # --mmproj-file: a local projector (implies vision)
VISION=0         # set by the settings phase
MMPROJ_PATH=''
# D20 state: set by resolve_model_file (settings phase), consumed by the
# parallel phase. Initialized so set -u is happy on every path.
LOCAL_FILE=''
MODEL_FILE_TARGET=''
BUILD_PID=''

usage() {
    cat <<'EOF'
usage: setup.sh [options]
  --model qwen38|qwen36|flint|apex35   model to use (asked if omitted)
  --quant q4k_m|q5k_xl|q8k_xl|q5k_m   quant build (qwen38/qwen36: q4k_m q5k_xl q8k_xl; apex35: q5k_m; default: the row's own)
  --vram full|16g|8g            VRAM target (asked if omitted)
  --context 262k|128k           context size (asked if omitted)
  --port N                      server port (default 8090, asked if omitted; D22)
  --host ADDR                   bind address (default: the recipe row's own)
  --device N                    CUDA index for device-pinned rows R5/R7 (default: auto)
  --model-file PATH             explicit local GGUF (still needs --model)
  --mmproj-file PATH            local multimodal projector GGUF (implies vision)
  --no-vision                   no vision support (no projector download, no --mmproj flags)
  --model-dir DIR               where to look for / download the model
  --clone-dir DIR               clone target (default ~/poormansllama)
  --branch B                    clone branch (default claude/flint-support)
  --repo-url U                  clone URL (default the GitHub repo)
  --cuda-arch N                 CUDA arch (default: detected from nvidia-smi)
  --full-clone                  clone full history instead of --depth 1
  --full-build                  rebuild even if the binary is newer than sources
  --skip-build                  emit build commands only
  --skip-model                  skip model lookup/download (recipe still checked)
  --skip-download               refuse to download (fail if no local model)
  --no-start                    do not ask to start the server at the end
  --dry-run                     no writes, no installs, no downloads
  --non-interactive             never ask; defaults apply
  --yes-install                 allow the apt install in non-interactive mode
  --hf-token TOK                Hugging Face token (env HF_TOKEN also honored)
  --max-download-mib N          test hook: stop the transfer at N MiB
  -h, --help                    this text
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --quant) QUANT="$2"; shift 2 ;;
        --vram) VRAM="$2"; shift 2 ;;
        --context) CONTEXT="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --host) BIND="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --model-file) MODEL_FILE="$2"; shift 2 ;;
        --mmproj-file) MMPROJ_FILE="$2"; shift 2 ;;
        --no-vision) NO_VISION=1; shift ;;
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --clone-dir) CLONE_DIR="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --repo-url) REPO_URL="$2"; shift 2 ;;
        --cuda-arch) CUDA_ARCH="$2"; shift 2 ;;
        --full-clone) FULL_CLONE=1; shift ;;
        --full-build) FULL_BUILD=1; shift ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --skip-model) SKIP_MODEL=1; shift ;;
        --skip-download) SKIP_DOWNLOAD=1; shift ;;
        --no-start) NO_START=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --non-interactive) NI=1; shift ;;
        --yes-install) YES_INSTALL=1; shift ;;
        --hf-token) HF_TOKEN="$2"; shift 2 ;;
        --max-download-mib) MAX_DL_MIB="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

trap 'echo; echo "[abort] interrupted" >&2; exit 6' INT

# --- small helpers ----------------------------------------------------------
step() { printf '\n== %s ==\n' "$1"; }
say()  { printf '  %s\n' "$1"; }
ok()   { printf '  [ok]   %s\n' "$1"; }
warn() { printf '  [warn] %s\n' "$1"; }
fail() { printf '  [FAIL] %s\n' "$1" >&2; exit "${2:-1}"; }

force_missing() {  # force_missing NAME -> 0 if the hook forces NAME missing
    [ -n "${SETUP_WIZARD_FORCE_MISSING:-}" ] || return 1
    case ",${SETUP_WIZARD_FORCE_MISSING}," in
        *,$(echo "$1" | tr '[:upper:]' '[:lower:]'),*) return 0 ;;
        *) return 1 ;;
    esac
}

# confirm PROMPT [default y|n] -> returns 0/1
confirm() {
    local prompt="$1" def="${2:-y}" a
    if [ "$NI" = 1 ]; then
        [ "$def" = "y" ]
        return $?
    fi
    printf '  %s [%s] ' "$prompt" "$([ "$def" = "y" ] && echo y/N || echo Y/n)"
    read -r a || a=""
    a="${a:-$def}"
    case "$a" in y|Y|yes|YES|'') return 0 ;; *) return 1 ;; esac
}

# ask_choice "Question" DEFAULT KEY1 KEY2 ... -> echoes the chosen key.
# The menu prints go to stderr (>&2) so a $(...) call captures ONLY the final
# selection echoed to stdout; on stdout the user would never see the menu
# (PR #16 fix, preserved). The explicit DEFAULT (SDD D19) fixes the old first-key
# default, which made the model menu's interactive default qwen38 while the
# non-interactive default was qwen36 - the two disagreed.
ask_choice() {
    local q="$1" def="$2"; shift 2
    local keys=("$@") local i a dno=1
    if [ "$NI" = 1 ]; then echo "$def"; return; fi
    for i in "${!keys[@]}"; do
        printf '    %d) %s%s\n' "$((i+1))" "${keys[$i]}" "$( [ "${keys[$i]}" = "$def" ] && echo ' (default)' )" >&2
        [ "${keys[$i]}" = "$def" ] && dno=$((i+1))
    done
    while :; do
        printf '  %s (1-%d, default %d) ' "$q" "${#keys[@]}" "$dno" >&2
        read -r a || a=""
        a="${a:-$dno}"   # empty/EOF returns the marked default (a key NAME would not match the case patterns)
        if [[ "$a" =~ ^[0-9]+$ ]] && [ "$a" -ge 1 ] && [ "$a" -le "${#keys[@]}" ]; then
            echo "${keys[$((a-1))]}"; return
        fi
        for k in "${keys[@]}"; do
            [ "$a" = "$k" ] && { echo "$k"; return; }
        done
        printf '  (choose a number or one of: %s)\n' "${keys[*]}" >&2
    done
}

# --- model table (SDD 4.5; links verified 2026-11-04, Q5_K_XL 2026-11-05) ----
# SDD D19: flat over (model, quant); the Default entries are the files the
# recipe rows were measured on. MI_IDX indexes this flat list.
# SDD D23 (2026-11-09): entry 7 is Apex-35B-V1.0 Q5_K_M (qwen35moe, 1 MTP layer); its size and the
# resolve-endpoint Content-Length were checked against the Hugging Face API (25,347,532,640 bytes).
# The Q6_K_L build was entry 8 and was removed on 2026-11-09.
M_KEYS=(qwen38 qwen36 flint qwen38 qwen36 qwen36 apex35)
M_QUANTS=(q4k_m q4k_m q5k_xl q5k_xl q5k_xl q8k_xl q5k_m)
M_REPOS=('unsloth/Qwen3.8-27B-GGUF' 'unsloth/Qwen3.6-35B-A3B-GGUF' 'unsloth/flint-GGUF' 'unsloth/Qwen3.8-27B-GGUF' 'unsloth/Qwen3.6-35B-A3B-GGUF' 'unsloth/Qwen3.6-35B-A3B-MTP-GGUF' 'Apexintelligence-AI/Apex-35B-V1.0-GGUF')
M_FILES=('Qwen3.8-27B-UD-Q4_K_M.gguf' 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf' 'flint-UD-Q5_K_XL.gguf' 'Qwen3.8-27B-UD-Q5_K_XL.gguf' 'Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf' 'Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf' 'Apex-35B-V1.0-Q5_K_M.gguf')
M_SIZES_GIB=(15.4 20.6 27.9 19.4 24.7 36.4 23.6)
M_SIZES_GIB_INT=(16 21 28 20 25 37 24)
M_KINDS=('dense hybrid' 'MoE' 'SWA MoE' 'dense hybrid' 'MoE' 'MoE + MTP' 'MoE + MTP')
M_DIRS=("$HOME/models/qwen-3.8" "$HOME/models/qwen-3.6" "$HOME/models/flint" "$HOME/models/qwen-3.8" "$HOME/models/qwen-3.6" "$HOME/models/qwen-3.6" "$HOME/models/apex")
M_DEFAULTS=(1 1 1 0 0 0 1)
# VOD-6: the vision projector of each entry (same repo as the model file; sizes read from the Hugging
# Face tree API on 2026-11-09). Empty = no projector published (flint is text-only; SDD D27).
M_MMPROJ=('mmproj-BF16.gguf' 'mmproj-BF16.gguf' '' 'mmproj-BF16.gguf' 'mmproj-BF16.gguf' 'mmproj-BF16.gguf' 'mmproj-Apexintelligence-AI_Apex-35B-V1.0-bf16.gguf')
M_MMPROJ_BYTES=(931146432 902822624 0 931146432 902822624 902822528 902822560)
M_MMPROJ_REPOS=('' '' '' '' '' '' 'bartowski/Apexintelligence-AI_Apex-35B-V1.0-GGUF')   # empty = the model's own repo

# vision_offered: is vision offered for the chosen model entry and recipe row? Needs a projector in the
# table and a recipe that does not carry --mmproj itself (the MTP / draft rows are included since D27: the
# owner's own Q8_K_XL MTP command ran with a projector). Uses MI_IDX and R_FLAGS. Pure: returns 0 (offered) / 1.
vision_offered() {
    [ -n "${M_MMPROJ[$MI_IDX]}" ] || return 1
    case " $R_FLAGS " in *' --mmproj '*) return 1 ;; esac
    return 0
}

mi() {  # mi KEY [QUANT] | NUM -> sets MI_IDX or exits 1 (SDD D19)
    case "$1" in
        1|2|3|4|5|6|7) MI_IDX=$(( $1 - 1 )) ;;
        qwen38|qwen36|flint|apex35)
            local k="$1" i q="${2:-}" avail=''
            for i in 0 1 2 3 4 5 6; do
                [ "${M_KEYS[$i]}" = "$k" ] || continue
                if [ -z "$q" ]; then
                    [ "${M_DEFAULTS[$i]}" = 1 ] && { MI_IDX=$i; return 0; }
                elif [ "${M_QUANTS[$i]}" = "$q" ]; then MI_IDX=$i; return 0; fi
                avail="$avail ${M_QUANTS[$i]}"
            done
            if [ -z "$q" ]; then
                fail "unknown model '$k' (choose qwen38, qwen36, flint, apex35, or 1-7)" 1
            else
                fail "no '$q' build for $k (available:$avail)" 1
            fi ;;
        *) fail "unknown model '$1' (choose qwen38, qwen36, flint, apex35, or 1-7)" 1 ;;
    esac
}

# --- recipe table (SDD 4.6; rows verbatim from the measured commands) -------
# recipe_exact MODEL VRAM CTX QUANT -> sets R_ID, R_FLAGS, R_BIND, R_MEASURED,
# R_NOTE, R_ENV, R_DEVICE; returns 1 if that exact row is not measured (SDD D21:
# rows are keyed on the quant too). recipe_for adds the default-quant fallback.
recipe_exact() {
    R_ID=''; R_FLAGS=''; R_BIND=''; R_MEASURED=''; R_NOTE=''; R_ENV=''; R_DEVICE=''
    local m="$1" v="$2" c="$3" q="$4"
    if [ "$m" = qwen38 ] && [ "$v" = full ] && [ "$c" = 262k ] && [ "$q" = q4k_m ]; then
        R_ID=R1; R_BIND=127.0.0.1
        R_FLAGS='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 -t 16 --no-warmup --lm-prefill-host-layers 0-37 --lm-decode-stream dynamic --lm-decode-slots 2 --lm-prefill --spec-type draft-mtp --spec-draft-p-min 0.6 --spec-draft-n-min 0 --spec-draft-n-max 3 --spec-mtp-defer -ctkd q8_0 -ctvd q8_0 -lv 4'
        R_MEASURED='PP 1093.25 t/s, TG 14.95 t/s (owner 2026-10-29, RTX 5090 24 GiB)'
        return 0
    fi
    if [ "$m" = qwen38 ] && [ "$v" = 16g ] && [ "$c" = 262k ] && [ "$q" = q4k_m ]; then
        R_ID=R2; R_BIND=127.0.0.1
        R_FLAGS='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 -t 16 --no-warmup --lm-prefill-host-layers 0-63 --lm-decode-stream dynamic --lm-decode-slots 2 --lm-prefill --spec-type draft-mtp --spec-draft-p-min 0.6 --spec-draft-n-min 0 --spec-draft-n-max 3 --spec-mtp-defer -ctkd q8_0 -ctvd q8_0 -lv 4'
        R_MEASURED='PP 1007.87 t/s, TG 9.96 t/s (owner 2026-10-29, RTX 5090 24 GiB)'
        return 0
    fi
    if [ "$m" = qwen38 ] && [ "$v" = 8g ] && [ "$c" = 128k ] && [ "$q" = q4k_m ]; then
        R_ID=R3; R_BIND=127.0.0.1
        R_FLAGS='-c 128000 -ctk q4_0 -ctv q4_0 -fa on -np 1 -t 16 --no-warmup --lm-prefill-host-layers 0-63 --lm-decode-stream dynamic --lm-decode-slots 2 --lm-prefill --lm-prefill-prearm always -lv 4'
        R_MEASURED='PP 36.95 t/s, TG 9.92 t/s (owner 2026-10-29, RTX 5090 24 GiB)'
        return 0
    fi
    if [ "$m" = flint ] && [ "$v" = full ] && [ "$c" = 262k ] && [ "$q" = q5k_xl ]; then
        R_ID=R4; R_BIND=0.0.0.0
        R_FLAGS='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 24 -b 262000 -ub 2048 --lm-prefill-ubatch 2048 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 3 -lv 4 --lm-prefill-copy-piece 64'
        R_MEASURED='PP 2422.94 t/s, TG 53.80 t/s (owner 2026-11-02, RTX 5090 24 GiB)'
        return 0
    fi
    if [ "$m" = qwen36 ] && [ "$v" = full ] && [ "$c" = 262k ] && [ "$q" = q8k_xl ]; then
        R_ID=R5; R_BIND=0.0.0.0
        R_FLAGS='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 24 -b 65536 --lm-prefill-ubatch 2048 --lm-prefill-segment 65536 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 3 -lv 4 --hybrid-moe-prefill-free-all --spec-type draft-mtp --spec-draft-p-min 0.1 --spec-draft-n-min 0 --spec-draft-n-max 3 --spec-mtp-defer -ctkd q8_0 -ctvd q8_0 --mtp-draft-vocab 32768 --lm-prefill-fuse --no-warmup'
        R_DEVICE=auto
        R_MEASURED='PP 3600 t/s @50k, 2350 t/s @200k; TG 77 t/s @50k, 66 t/s @200k (owner, GTX 5090 Laptop, Windows)'
        R_ENV='LLAMA_MOE_HYBRID_LEAN_STALL_US=10000 LLAMA_FATTN_GQA8_MAXROWS=4'
        R_NOTE='needs the MTP build of this quant (unsloth/Qwen3.6-35B-A3B-MTP-GGUF). Measured on Windows with one MoE model only; Linux and other models are not measured with these flags'
        return 0
    fi
    if [ "$m" = qwen36 ] && [ "$v" = 16g ] && [ "$c" = 262k ] && [ "$q" = q8k_xl ]; then
        R_ID=R6; R_BIND=0.0.0.0
        R_FLAGS='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe -fitt 8 --lm-prefill --lm-prefill-ckpt-chunk 24 -b 32768 --lm-prefill-ubatch 2048 --lm-prefill-segment 32768 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 3 -lv 4 --hybrid-moe-prefill-free-all --spec-type draft-mtp --spec-draft-p-min 0.1 --spec-draft-n-min 0 --spec-draft-n-max 3 --spec-mtp-defer -ctkd q8_0 -ctvd q8_0 --mtp-draft-vocab 32768 --lm-prefill-fuse'
        R_DEVICE=auto
        R_MEASURED='PP 2000 t/s, TG 51 t/s @200k (owner, GTX 5090 Laptop, Windows)'
        R_ENV='LLAMA_MOE_HYBRID_LEAN_STALL_US=10000 LLAMA_FATTN_GQA8_MAXROWS=4'
        R_NOTE='needs the MTP build of this quant (unsloth/Qwen3.6-35B-A3B-MTP-GGUF). Measured on Windows with one MoE model only; Linux and other models are not measured with these flags'
        return 0
    fi
    if [ "$m" = qwen36 ] && [ "$v" = 8g ] && [ "$c" = 262k ] && [ "$q" = q4k_m ]; then
        R_ID=R7; R_BIND=0.0.0.0
        R_FLAGS='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 24 -b 32768 -ub 512 --lm-prefill-ubatch 2048 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 2 -lv 4 --hybrid-moe-prefill-free-all --lm-prefill-segment 16384'
        R_DEVICE=auto
        R_MEASURED='PP 590 t/s, TG 33 t/s @20k (owner, GTX 1070 8 GiB, Windows)'
        R_ENV='LLAMA_MOE_HYBRID_LEAN_STALL_US=10000 LLAMA_FATTN_GQA8_MAXROWS=4'
        R_NOTE='measured at 20k context on an 8 GiB card, Windows, one MoE model only; the row is 262k context as the owner ran the command'
        return 0
    fi
    if [ "$m" = qwen36 ] && [ "$v" = 16g ] && [ "$c" = 262k ] && [ "$q" = q4k_m ]; then
        R_ID=R8; R_BIND=127.0.0.1
        R_FLAGS='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe --lm-prefill --lm-prefill-canonical --lm-prefill-fuse --lm-prefill-prearm auto --lm-prefill-ubatch 2048 --lm-prefill-segment 32768 -b 32768 --hybrid-moe-prefill-ubatch 0 -fitt 16000 --ctx-checkpoints 32'
        R_MEASURED='G42 canonical: TG 3.0-3.7x stock, PP 3.65-4.11x stock (docs/guide/USAGE.md, build s53)'
        R_NOTE='measured on Qwen3.6-35B-A3B-MTP-UD-Q4_K_S; the wizard offers the closest HF file, Q4_K_M (SDD D13). Kept as R8 (the former R6)'
        return 0
    fi
    # SDD D23: the owner's running command for Apex-35B (the R5 shape, MTP head used). Measured by this
    # repo on 2026-11-09 with Apex-35B-V1.0-Q5_K_S.gguf, a file the wizard does not offer. No row env.
    if [ "$m" = apex35 ] && [ "$v" = full ] && [ "$c" = 262k ] && [ "$q" = q5k_m ]; then
        R_ID=R9; R_BIND=0.0.0.0
        R_FLAGS='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 24 -b 65536 --lm-prefill-ubatch 2048 --lm-prefill-segment 65536 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 3 -lv 4 --hybrid-moe-prefill-free-all --spec-type draft-mtp --spec-draft-p-min 0.1 --spec-draft-n-min 0 --spec-draft-n-max 3 --spec-mtp-defer -ctkd q8_0 -ctvd q8_0 --mtp-draft-vocab 32768 --lm-prefill-fuse --no-warmup'
        R_DEVICE=auto
        R_MEASURED='TG ~100-110 tok/s (600-token greedy reply), PP ~3,900 t/s on a 20.7k-token prompt (2026-11-09, RTX 5090 Laptop 24 GiB, Windows, on Apex-35B-V1.0-Q5_K_S.gguf)'
        R_NOTE='measured on Apex-35B-V1.0-Q5_K_S.gguf, NOT on the offered Q5_K_M (same architecture qwen35moe, 41 blocks with 1 MTP layer; the offered file carries the blk.40.nextn tensors, GGUF header read 2026-11-09). Windows only, one MoE family'
        return 0
    fi
    return 1
}

# recipe_for MODEL VRAM CTX [QUANT] -> recipe_exact, else the model's default-quant
# row (R_QUANT_FALLBACK=1; the caller warns). Sets R_QUANT to the quant of the row.
recipe_for() {
    local m="$1" v="$2" c="$3" q="${4:-}" dq
    R_QUANT_FALLBACK=0
    R_CLASS_FALLBACK=0
    if recipe_exact "$m" "$v" "$c" "$q"; then R_QUANT="$q"; return 0; fi
    dq=$(default_quant "$m")
    if [ -n "$dq" ] && [ "$dq" != "$q" ] && recipe_exact "$m" "$v" "$c" "$dq"; then
        R_QUANT="$dq"; R_QUANT_FALLBACK=1; return 0
    fi
    # SDD D24: a MoE model with no row of its own at 8g takes the shared 8g MoE profile
    if is_moe_model "$m" && moe_class_row "$v" "$c"; then
        R_CLASS_FALLBACK=1; return 0
    fi
    return 1
}

is_moe_model() {  # is_moe_model KEY -> 0 when any table entry of the model says MoE (SDD D24)
    local i
    for i in 0 1 2 3 4 5 6; do
        if [ "${M_KEYS[$i]}" = "$1" ]; then
            case "${M_KINDS[$i]}" in *MoE*) return 0 ;; esac
        fi
    done
    return 1
}

# moe_class_row VRAM CTX -> loads the shared 8g MoE profile (SDD D24): EVERY MoE model, whatever its name
# and whether or not it was benchmarked, takes the owner's 8 GiB GTX 1070 row (R7, qwen36 q4k_m) at 8g.
# Other VRAM targets keep their per-model rows. Sets R_QUANT to the row's own quant.
moe_class_row() {
    [ "$1" = 8g ] || return 1
    recipe_exact qwen36 8g "$2" q4k_m || return 1
    R_QUANT=q4k_m
    return 0
}

default_quant() {  # default_quant MODEL -> the model's default-quant (SDD D19)
    local i
    for i in 0 1 2 3 4 5 6; do
        if [ "${M_KEYS[$i]}" = "$1" ] && [ "${M_DEFAULTS[$i]}" = 1 ]; then echo "${M_QUANTS[$i]}"; return 0; fi
    done
    return 1
}

recipe_avail() {  # recipe_avail MODEL -> prints the available rows (all quants)
    local m="$1" v c q
    say "no measured recipe for $m / $VRAM / $CONTEXT / ${QUANT:-default quant} - available rows for $m:"
    for v in full 16g 8g; do
        for c in 262k 128k; do
            for q in q4k_m q5k_xl q8k_xl q5k_m; do
                if recipe_exact "$m" "$v" "$c" "$q"; then
                    printf '    %-3s %-5s %-6s %-7s %s\n' "$R_ID" "$v" "$c" "$q" "$R_MEASURED"
                fi
            done
        done
    done
    # SDD D24: every MoE model also has the shared 8g profile
    if is_moe_model "$m" && moe_class_row 8g 262k; then
        printf '    %-3s %-5s %-6s %-7s %s\n' "$R_ID" 8g 262k 'any MoE' "(shared 8g profile of every MoE model) $R_MEASURED"
    fi
}

# --- step 1: environment check (SDD 4.1) ------------------------------------
ENV_BAD=()
GPU_LINE=''

ver_ge() {  # ver_ge A B  (A and B like "4.2" or "2.43.0")
    local a1 a2 b1 b2
    a1=${1%%.*}; a2=${1#*.}; a2=${a2%%.*}
    b1=${2%%.*}; b2=${2#*.}; b2=${b2%%.*}
    [ "$a2" = "$1" ] && a2=0
    [ "$b2" = "$2" ] && b2=0
    if [ "$a1" -gt "$b1" ] 2>/dev/null; then return 0; fi
    [ "$a1" -eq "$b1" ] 2>/dev/null && [ "$a2" -ge "$b2" ] 2>/dev/null
}

check_env() {
    step 'Environment check'
    ENV_BAD=()

    # git >= 2.30
    if force_missing git; then
        ENV_BAD+=(git); say '    git          MISSING'
    else
        if git --version >/dev/null 2>&1; then
            local gv; gv=$(git --version 2>/dev/null | sed 's/^git version //')
            if ver_ge "$gv" 2.30; then
                say "    git          OK   ($gv)"
            else
                ENV_BAD+=(git); say "    git          TOO-OLD ($gv, need >= 2.30)"
            fi
        else
            ENV_BAD+=(git); say '    git          MISSING'
        fi
    fi

    # cmake >= 4.2
    if force_missing cmake; then
        ENV_BAD+=(cmake); say '    cmake        MISSING'
    else
        if command -v cmake >/dev/null 2>&1; then
            local cv; cv=$(cmake --version 2>/dev/null | head -1 | sed 's/^cmake version //')
            if ver_ge "$cv" 4.2; then
                say "    cmake        OK   ($cv)"
            else
                ENV_BAD+=(cmake); say "    cmake        TOO-OLD ($cv, need >= 4.2)"
            fi
        else
            ENV_BAD+=(cmake); say '    cmake        MISSING'
        fi
    fi

    # CUDA toolkit
    if force_missing cuda; then
        ENV_BAD+=(cuda); say '    cuda         MISSING'
    else
        local cv=''
        if command -v nvcc >/dev/null 2>&1; then
            cv=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)
            [ -n "$cv" ] && say "    cuda         OK   (nvcc $cv)"
        fi
        if [ -z "$cv" ]; then
            local d
            for d in /usr/local/cuda*; do
                [ -x "$d/bin/nvcc" ] || continue
                cv=$("$d/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)
                [ -n "$cv" ] && say "    cuda         OK   (nvcc $cv at $d)" && break
            done
        fi
        [ -z "$cv" ] && { ENV_BAD+=(cuda); say '    cuda         MISSING (nvcc not found, no /usr/local/cuda*)'; }
    fi

    # gcc >= 9 / g++
    if force_missing gcc; then
        ENV_BAD+=(gcc); say '    gcc          MISSING'
    else
        if command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
            local gvv; gvv=$(gcc --version 2>/dev/null | head -1 | awk '{print $NF}')
            if ver_ge "$gvv" 9.0; then
                say "    gcc          OK   (gcc $gvv, g++ present)"
            else
                ENV_BAD+=(gcc); say "    gcc          TOO-OLD (gcc $gvv, need >= 9)"
            fi
        else
            ENV_BAD+=(gcc); say '    gcc          MISSING'
        fi
    fi

    # nvidia-smi (GPU + driver): fatal if missing, never auto-installed
    if force_missing nvidia-smi; then
        ENV_BAD+=(nvidia-smi); say '    nvidia-smi   MISSING'
    else
        if command -v nvidia-smi >/dev/null 2>&1; then
            GPU_LINE=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1)
            if [ -n "$GPU_LINE" ]; then
                say "    nvidia-smi   OK   ($GPU_LINE)"
            else
                ENV_BAD+=(nvidia-smi); say '    nvidia-smi   PRESENT-BUT-ERROR (nvidia-smi failed)'
            fi
        else
            ENV_BAD+=(nvidia-smi); say '    nvidia-smi   MISSING'
        fi
    fi

    # RAM (warn only)
    local ram_gib=0 mt
    mt=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo 2>/dev/null)
    [ -n "$mt" ] && ram_gib="$mt"
    if [ "$ram_gib" -ge 48 ]; then
        say "    ram          OK   ($ram_gib GiB >= 48 GiB recommended for MoE host experts)"
    else
        say "    ram          LOW  ($ram_gib GiB, >= 48 GiB recommended for MoE host experts)"
    fi

    # print bad summary
    local b
    for b in "${ENV_BAD[@]:-}"; do
        [ -n "$b" ] || continue
        case "$b" in
            nvidia-smi)
                warn 'no NVIDIA GPU/driver found. Install the driver first: https://www.nvidia.com/Download/index.aspx'
                ;;
        esac
    done
}

# --- step 2: install missing tools (apt; SDD 4.2) ---------------------------
install_missing() {
    local todo=() b
    for b in "${ENV_BAD[@]:-}"; do
        [ -n "$b" ] || continue
        [ "$b" = nvidia-smi ] && continue   # driver: never auto-installed
        case "$b" in git|cmake|cuda|gcc) todo+=("$b") ;; esac
    done
    [ ${#todo[@]} -eq 0 ] && return 0
    step 'Install missing tools (apt)'
    local pkgs=()
    for b in "${todo[@]}"; do
        case "$b" in
            git)   pkgs+=(git) ;;
            cmake) pkgs+=(cmake) ;;
            gcc)   pkgs+=(build-essential) ;;
            cuda)  pkgs+=(nvidia-cuda-toolkit) ;;
        esac
    done
    local line='sudo apt install -y '"$(printf '%s ' "${pkgs[@]}")"
    say "$line"
    if [ "$DRY_RUN" = 1 ]; then say '(dry-run: nothing installed)'; return 0; fi
    if ! confirm 'Install now?' y; then
        warn 'manual equivalents:'
        say '  git:            sudo apt install git'
        say '  cmake:          sudo apt install cmake'
        say '  build-essential: sudo apt install build-essential'
        say '  cuda toolkit:   sudo apt install nvidia-cuda-toolkit   (or the NVIDIA CUDA repo: https://developer.nvidia.com/cuda-downloads)'
        say '  then re-run this wizard.'
        fail 'user declined install; missing tools remain' 2
    fi
    [ "$NI" = 1 ] && [ "$YES_INSTALL" != 1 ] && fail 'non-interactive install requires --yes-install' 2
    sudo apt-get update -y
    sudo apt install -y "${pkgs[@]}"
    hash -r
    check_env
    local still=0
    for b in "${ENV_BAD[@]:-}"; do
        [ -n "$b" ] || continue
        [ "$b" = nvidia-smi ] && continue
        warn "after install: $b still missing"
        still=1
    done
    [ "$still" = 1 ] && fail 'a required tool is still missing after the install attempt; see the table above' 2
    # distro CUDA toolkit version gate (SDD 4.2): the fork's Linux build was
    # measured on CUDA 12.8; distro toolkits older than 12 need the NVIDIA repo.
    if command -v nvcc >/dev/null 2>&1; then
        local cv; cv=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\).*/\1/p' | head -1)
        if [ -n "$cv" ] && [ "$cv" -lt 12 ]; then
            fail "distro CUDA toolkit is $cv (< 12). Add the NVIDIA CUDA repo: https://developer.nvidia.com/cuda-downloads (the fork's Linux build was measured on CUDA 12.8, see docs/guide/LINUX-PASCAL.md)" 2
        fi
    fi
    ok 'install complete; environment re-checked'
}

# --- step 3: repository (SDD 4.3, decision D8) ------------------------------
get_repo_root() {
    local src="${BASH_SOURCE[0]:-}"
    if [ -n "$src" ] && [ -f "$src" ]; then
        local d; d=$(cd "$(dirname "$src")" && pwd)
        [ -d "$d/third_party/llama.cpp" ] && { echo "$d"; return; }
        d=$(cd "$d" 2>/dev/null && pwd)
        while [ -n "$d" ] && [ "$d" != / ]; do
            [ -d "$d/third_party/llama.cpp" ] && { echo "$d"; return; }
            d=$(dirname "$d")
        done
    fi
    return 1
}

invoke_clone() {
    step 'Repository'
    local in_repo
    if [ -z "$CLONE_DIR" ]; then
        # no explicit target: running inside the repo means nothing to clone
        if in_repo=$(get_repo_root); then
            ok "already inside a checkout: $in_repo (clone skipped)"
            REPO="$in_repo"
            return 0
        fi
        CLONE_DIR="$HOME/poormansllama"
        say "no --clone-dir given; using default $CLONE_DIR"
    fi
    REPO="$CLONE_DIR"
    if [ -e "$REPO" ] && [ -n "$(ls -A "$REPO" 2>/dev/null)" ]; then
        # reuse a valid existing checkout so a re-run (different model, same
        # day or later) works; anything else is still refused. Valid = a git
        # repo whose origin is this wizard's repo URL (D58, ported from
        # setup.ps1: strip a trailing .git, lowercase, trim slashes).
        if [ -d "$REPO/.git" ]; then
            local origin onorm unorm
            origin=$(git -C "$REPO" remote get-url origin 2>/dev/null)
            onorm=$(printf '%s' "$origin" | sed -e 's/\.git$//' -e 's#/*$##' | tr '[:upper:]' '[:lower:]')
            unorm=$(printf '%s' "$REPO_URL" | sed -e 's/\.git$//' -e 's#/*$##' | tr '[:upper:]' '[:lower:]')
            if [ -n "$onorm" ] && [ "$onorm" = "$unorm" ]; then
                ok "reusing existing checkout: $REPO"
                return 0
            fi
        fi
        fail "clone target $REPO exists and is not empty (use --clone-dir to pick another)" 1
    fi
    local args=(clone)
    [ "$FULL_CLONE" = 0 ] && args+=(--depth 1)
    args+=(--branch "$BRANCH" "$REPO_URL" "$REPO")
    if [ "$DRY_RUN" = 1 ]; then say "(dry-run: git ${args[*]} )"; return 0; fi
    # No 'set -o pipefail' in this script, so 'git … | sed' reports sed's exit
    # status, not git's (measured 2026-11-04: a failed --branch clone printed
    # [ok] and the fallback below never ran). Capture the output, test $?
    # directly (decision D59).
    local clone_out rc
    clone_out=$(git "${args[@]}" 2>&1)
    rc=$?
    [ -n "$clone_out" ] && printf '%s\n' "$clone_out" | sed 's/^/  /'
    if [ "$rc" -ne 0 ]; then
        if [ "$BRANCH" != "main" ]; then
            # the requested branch is not published (e.g. in-flight work): fall
            # back to the remote default branch; the flag-capability check
            # below refuses to emit a row the resulting binary cannot run.
            warn "clone of branch '$BRANCH' failed (not on the remote?); falling back to the remote default branch"
            rm -rf "$REPO"
            args=(clone)
            [ "$FULL_CLONE" = 0 ] && args+=(--depth 1)
            args+=("$REPO_URL" "$REPO")
            clone_out=$(git "${args[@]}" 2>&1)
            rc=$?
            [ -n "$clone_out" ] && printf '%s\n' "$clone_out" | sed 's/^/  /'
            [ "$rc" -ne 0 ] && fail "git clone failed (branch: $BRANCH)" 1
        else
            fail "git clone failed" 1
        fi
    fi
    # An incomplete tree must not reach the build step (loud failure).
    [ -d "$REPO/third_party/llama.cpp" ] || fail "clone at $REPO is incomplete (third_party/llama.cpp missing)" 1
    ok "cloned -> $REPO (branch: $BRANCH or remote default)"
}

check_row_flags() {
    # $1 exe, $2 flags, $3 row id: every --flag of the row must appear in
    # 'llama-server --help', or the emitted script can never run
    local exe=$1 flags=$2 row=$3
    [ -x "$exe" ] || { say "(flag check skipped: $exe not built yet)"; return 0; }
    local help tok missing=()
    help=$("$exe" --help 2>&1)
    for tok in $flags; do
        case "$tok" in --*) ;; *) continue ;; esac
        case "$help" in *"$tok"*) ;; *) missing+=("$tok") ;; esac
    done
    if [ ${#missing[@]} -gt 0 ]; then
        fail "binary does not support row $row flags: ${missing[*]} — rebuild from a branch that contains them, or pick another row" 1
    fi
    ok "binary supports all $row row flags"
}

# --- step 4: build (SDD 4.4, decisions D9/D10) ------------------------------
# --- D22: row-flag completion -------------------------------------------------
# The wizard's one place that changes what a row emits, beyond the D11/D17
# substitutions (model path, bind, port, device index). Pure over the row:
# the same (flags, env, compute cap) always gives the same output, so the
# same-day byte-identical re-run (SDD AC-2) still holds.
detect_gpu_arch() {
    # D22: compute capability of the card the run will use (nvidia-smi's
    # first line, the D12/D17 convention) as 'M.m'. SETUP_WIZARD_GPU_CC
    # overrides for testing (SETUP_WIZARD_FORCE_MISSING is the precedent).
    # Returns 1 when unavailable; the caller then emits the row as measured.
    if [ -n "${SETUP_WIZARD_GPU_CC:-}" ]; then echo "$SETUP_WIZARD_GPU_CC"; return 0; fi
    local cc
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    [[ "$cc" =~ ^[0-9]+\.[0-9]+$ ]] || return 1
    echo "$cc"
}

complete_row_flags() {
    # D22-B: every profile emits --lm-prefill-prearm always. An existing
    # occurrence (R3's 'always', R8's 'auto') is REPLACED, otherwise the
    # flag is appended - two occurrences would be a parse hazard.
    # D22-A: on Pascal and lower (compute cap <= 6.1) the pass's flash
    # attention takes the int8 Q.K^T tile (the flag is literally
    # setenv(LLAMA_FATTN_TILE_Q8QK,1) in the server, help: 'Same as
    # LLAMA_FATTN_TILE_Q8QK=1', target 'a GPU without fast FP16, such as
    # Pascal'). If the row's env already sets that env, the env is the
    # emitted config - it stays, and nothing is appended; otherwise the
    # flag is appended. AN APPROXIMATION (int8 Q per 32 values): the
    # caller prints a D22 note when the flag was added.
    # $1 flags, $2 row env (KEY=V ...), $3 compute cap ('' = unavailable).
    # Echoes the completed flags; sets D22_ADDED_INT8=1 when the flag was
    # say goes to stderr: the caller runs this in a command substitution, so
    # the caller derives 'was the flag added' from the output itself.
    local f="$1" env="$2" cc="${3:-}" maj min
    case "$f" in
        *'--lm-prefill-prearm off'*)   f="${f/--lm-prefill-prearm off/--lm-prefill-prearm always}" ;;
        *'--lm-prefill-prearm always'*) f="${f/--lm-prefill-prearm always/--lm-prefill-prearm always}" ;;
        *'--lm-prefill-prearm auto'*)  f="${f/--lm-prefill-prearm auto/--lm-prefill-prearm always}" ;;
        *) f="$f --lm-prefill-prearm always" ;;
    esac
    if [[ "$cc" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
        maj="${BASH_REMATCH[1]}"; min="${BASH_REMATCH[2]}"
        if [ "$maj" -lt 6 ] || { [ "$maj" -eq 6 ] && [ "$min" -le 1 ]; }; then
            local has_env=0 tok
            for tok in $env; do
                [ "$tok" = 'LLAMA_FATTN_TILE_Q8QK=1' ] && has_env=1
            done
            if [ "$has_env" -eq 1 ]; then
                say "row env already sets LLAMA_FATTN_TILE_Q8QK=1 (compute capability $cc); the env is the emitted config, no flag added" >&2
            else
                # already present (idempotence: a re-completed output must not
                # gain a second copy)
                case " $f " in *' --lm-prefill-attn-int8 '*) ;; *) f="$f --lm-prefill-attn-int8" ;; esac
            fi
        fi
    fi
    # VOD-6: $4 = 1 appends the vision flags, exactly once; $MMPROJ is defined by the emitted script next to
    # $MODEL (a path with spaces needs no quoting here). The row's own flags never carry --mmproj
    # (vision_offered refuses a row that does).
    if [ "${4:-0}" = 1 ]; then
        case "$f" in *'--mmproj-on-demand'*) ;; *) f="$f"' --mmproj "$MMPROJ" --mmproj-on-demand' ;; esac
    fi
    echo "$f"
}

detect_cuda_arch() {
    # nvidia-smi compute_cap -> CMAKE_CUDA_ARCHITECTURES entry (6.1 -> 61)
    local cc
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    if [[ "$cc" =~ ^[0-9]+\.[0-9]+$ ]]; then
        echo "${cc%.*}${cc#*.}"
        return 0
    fi
    return 1
}

binary_fresh() {
    # $1 repo, $2 bin: true iff bin exists and no scanned source is newer
    [ -f "$2" ] || return 1
    local newer
    newer=$(find "$1/third_party/llama.cpp/src" "$1/third_party/llama.cpp/ggml" \
        "$1/third_party/llama.cpp/common" "$1/third_party/llama.cpp/tools/llama-server" \
        -type f \( -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.hpp' \) \
        -newer "$2" 2>/dev/null | head -1)
    [ -z "$newer" ]
}

invoke_build() {
    # SEQUENTIAL form (SDD D20): used for the --skip-model-without-model run
    # and for single-leg (build-only) phases. The parallel phase (step 5)
    # uses build_start + poll_download instead.
    step 'Build llama-server'
    local bin="$REPO/build/llama-cuda/bin/llama-server"
    if [ "$SKIP_BUILD" = 1 ]; then say '(skipped: --skip-build)'; BUILD_BIN="$bin"; return 0; fi
    if [ "$FULL_BUILD" != 1 ] && binary_fresh "$REPO" "$bin"; then
        ok "build skipped (binary newer than sources): $bin"
        BUILD_BIN="$bin"
        return 0
    fi
    local arch="$CUDA_ARCH"
    if [ -z "$arch" ]; then
        if ! arch=$(detect_cuda_arch); then
            fail 'could not detect the CUDA architecture from nvidia-smi; pass --cuda-arch NNN (e.g. 61 for Pascal, 89 for Ada; the LINUX-PASCAL build uses an explicit list)' 2
        fi
        say "detected CUDA arch: $arch"
    fi
    # one variable per statement: 'local a=$x b=$a' expands $a before it is set (set -u dies)
    local src="$REPO/third_party/llama.cpp"
    local bld="$REPO/build/llama-cuda"
    local log="$bld/setup-wizard-build.log"
    mkdir -p "$bld"
    local nproc; nproc=$(nproc 2>/dev/null || echo 4)
    if [ "$DRY_RUN" = 1 ]; then
        say "(dry-run: cmake -S $src -B $bld -DGGML_CUDA=ON -DLLAMA_LAYER_STREAMING=ON -DCMAKE_CUDA_ARCHITECTURES=$arch -DCMAKE_BUILD_TYPE=Release)"
        say "(dry-run: cmake --build $bld --target llama-server --parallel $nproc)"
        BUILD_BIN="$bin"
        return 0
    fi
    cmake -S "$src" -B "$bld" -DGGML_CUDA=ON -DLLAMA_LAYER_STREAMING=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$arch" -DCMAKE_BUILD_TYPE=Release \
        2>&1 | tee "$log"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        tail -50 "$log" | sed 's/^/    /'
        fail "cmake configure failed (full log: $log)" 4
    fi
    cmake --build "$bld" --target llama-server --parallel "$nproc" 2>&1 | tee -a "$log"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        tail -50 "$log" | sed 's/^/    /'
        fail "build failed (full log: $log)" 4
    fi
    [ -f "$bin" ] || fail "build finished but $bin was not produced" 4
    ok "built $bin"
    BUILD_BIN="$bin"
}

# --- step 5: model (SDD 4.5) -------------------------------------------------
pick_model() {
    if [ -n "$MODEL" ]; then
        if [ -n "$QUANT" ]; then mi "$MODEL" "$QUANT"; else mi "$MODEL"; fi
        return 0
    fi
    local i qtag opts=()
    for i in 0 1 2 3 4 5 6; do
        qtag="${M_QUANTS[$i]}"
        [ "${M_DEFAULTS[$i]}" = 1 ] || qtag="$qtag (alt)"
        opts+=("$(printf '%s: %s - %s, %s GiB [Q: %s]' "${M_KEYS[$i]}" "${M_KINDS[$i]}" "${M_FILES[$i]}" "${M_SIZES_GIB[$i]}" "$qtag")")
    done
    if [ "$NI" = 1 ]; then MI_IDX=1; return 0; fi   # default: qwen36 q4k_m (the vehicle model)
    local sel
    sel=$(ask_choice 'Which model (and quant)?' "$(printf '%s' "${opts[1]}")" "${opts[@]}")
    case "$sel" in
        *qwen38*Q5_K_XL*) MI_IDX=3 ;;
        *qwen36*Q5_K_XL*) MI_IDX=4 ;;
        *qwen36*Q8_K_XL*) MI_IDX=5 ;;
        *apex35*Q5_K_M*) MI_IDX=6 ;;
        *qwen38*) MI_IDX=0 ;;
        *qwen36*) MI_IDX=1 ;;
        *flint*)  MI_IDX=2 ;;
        *) fail "could not parse model choice '$sel'" 1 ;;
    esac
}

resolve_model_file() {
    # D20: settings phase - decides LOCAL_FILE (an existing local file) or
    # MODEL_FILE_TARGET (the table file to fetch in the parallel phase).
    # Nothing is downloaded here.
    local dir="$MODEL_DIR"
    [ -z "$dir" ] && dir="${M_DIRS[$MI_IDX]}"
    local file="$dir/${M_FILES[$MI_IDX]}"
    MODEL_FILE_TARGET=''
    if [ -n "$MODEL_FILE" ]; then
        [ -f "$MODEL_FILE" ] || fail "-model-file does not exist: $MODEL_FILE" 1
        LOCAL_FILE="$MODEL_FILE"
        ok "model: $LOCAL_FILE (--model-file)"
        return 0
    fi
    if [ -f "$file" ]; then
        LOCAL_FILE="$file"
        ok "model: $LOCAL_FILE"
        return 0
    fi
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir" || fail "cannot create model dir $dir" 1
    fi
    # offer other ggufs in the dir
    local others=() f
    for f in "$dir"/*.gguf; do
        [ -f "$f" ] && others+=("$f")
    done
    if [ ${#others[@]} -gt 0 ]; then
        say "no ${M_FILES[$MI_IDX]} in $dir, but found:"
        local i
        for i in "${!others[@]}"; do say "    $((i+1))) $(basename "${others[$i]}")"; done
        local a
        if [ "$NI" = 1 ]; then a='y'; else
            read -r -p '  use one of these (number), y to download the table file, or type a full path [y] ' a
            a="${a:-y}"
        fi
        if [[ "$a" =~ ^[0-9]+$ ]] && [ "$a" -ge 1 ] && [ "$a" -le ${#others[@]} ]; then
            LOCAL_FILE="${others[$((a-1))]}"
            return 0
        fi
        case "$a" in
            *.gguf) [ -f "$a" ] && { LOCAL_FILE="$a"; return 0; } ;;
        esac
        # fall through: download target
    fi
    LOCAL_FILE=''
    MODEL_FILE_TARGET="$file"
}

# --- parallel phase (SDD D20): build leg + download leg -------------------
# Both legs run in the background; poll_download() renders the combined
# status line (~1 s tick) and fails fast on the first hard failure. The
# download's resume/Range/verify logic is unchanged (curl -C -); the build
# keeps the documented cmake commands (D9 arch, D10 freshness checked by
# the caller before build_start is ever called).
build_start() {
    # one variable per statement: 'local a=$x b=$a' expands $a before it is
    # set and set -u dies (D44)
    local arch="$CUDA_ARCH"
    if [ -z "$arch" ]; then
        if ! arch=$(detect_cuda_arch); then
            fail 'could not detect the CUDA architecture from nvidia-smi; pass --cuda-arch NNN (e.g. 61 for Pascal, 89 for Ada; the LINUX-PASCAL build uses an explicit list)' 2
        fi
        say "detected CUDA arch: $arch"
    fi
    local src="$REPO/third_party/llama.cpp"
    local bld="$REPO/build/llama-cuda"
    BUILD_CFG_LOG="$bld/setup-wizard-cmake-configure.log"
    BUILD_BLD_LOG="$bld/setup-wizard-cmake-build.log"
    BUILD_RC1="$bld/setup-wizard-rc1.tmp"
    BUILD_RC2="$bld/setup-wizard-rc2.tmp"
    rm -f "$BUILD_RC1" "$BUILD_RC2"
    (
        cmake -S "$src" -B "$bld" -DGGML_CUDA=ON -DLLAMA_LAYER_STREAMING=ON \
            -DCMAKE_CUDA_ARCHITECTURES="$arch" -DCMAKE_BUILD_TYPE=Release \
            > "$BUILD_CFG_LOG" 2>&1
        echo "$?" > "$BUILD_RC1"
        cmake --build "$bld" --target llama-server --parallel "$(nproc 2>/dev/null || echo 4)" \
            > "$BUILD_BLD_LOG" 2>&1
        echo "$?" > "$BUILD_RC2"
    ) &
    BUILD_PID=$!
    ok "build started (configure log: $BUILD_CFG_LOG, build log: $BUILD_BLD_LOG)"
}

build_wait() {
    # blocking collector: waits for the leg, checks both stage rcs, prints
    # the log tail and fails 4 on any stage failure (same tails as
    # invoke_build). Sets BUILD_BIN on success.
    local bld="$REPO/build/llama-cuda"
    local bin="$bld/bin/llama-server"
    wait "$BUILD_PID" 2>/dev/null
    [ -f "$BUILD_RC1" ] || fail 'cmake configure did not report an exit code (log: '$BUILD_CFG_LOG')' 4
    local rc1; rc1=$(head -1 "$BUILD_RC1")
    if [ "$rc1" != 0 ]; then
        tail -50 "$BUILD_CFG_LOG" | sed 's/^/    /'
        fail "cmake configure failed (exit $rc1; full log: $BUILD_CFG_LOG)" 4
    fi
    [ -f "$BUILD_RC2" ] || fail 'cmake build did not report an exit code (log: '$BUILD_BLD_LOG')' 4
    local rc2; rc2=$(head -1 "$BUILD_RC2")
    if [ "$rc2" != 0 ]; then
        tail -50 "$BUILD_BLD_LOG" | sed 's/^/    /'
        fail "build failed (exit $rc2; full log: $BUILD_BLD_LOG)" 4
    fi
    [ -f "$bin" ] || fail 'build finished but '$bin' was not produced' 4
    ok "built $bin"
    BUILD_BIN="$bin"
}

build_kill() {
    # fail-fast / Ctrl-C: take the whole cmake tree (make/ninja descendants)
    if [ -n "${BUILD_PID:-}" ]; then
        pkill -TERM -P "$BUILD_PID" 2>/dev/null
        kill -TERM "$BUILD_PID" 2>/dev/null
        sleep 1
        pkill -KILL -P "$BUILD_PID" 2>/dev/null
        kill -KILL "$BUILD_PID" 2>/dev/null
    fi
}

download_start() {
    local dir="$1"
    DL_FILE="$dir/${M_FILES[$MI_IDX]}"
    DL_PART="$DL_FILE.part"
    DL_URL="https://huggingface.co/${M_REPOS[$MI_IDX]}/resolve/main/${M_FILES[$MI_IDX]}"
    DL_MAXBYTES=0
    [ "$MAX_DL_MIB" -gt 0 ] && DL_MAXBYTES=$(( MAX_DL_MIB * 1048576 ))
    if [ "$SKIP_DOWNLOAD" = 1 ]; then fail '--skip-download set but no local model found (expected '$DL_FILE')' 5; fi
    command -v curl >/dev/null 2>&1 || fail 'curl is required for the download (install it: sudo apt install curl; manual: '$DL_URL')' 5
    say "model: downloading ${M_FILES[$MI_IDX]} (${M_SIZES_GIB[$MI_IDX]} GiB) from Hugging Face"
    say "$DL_URL"
    if [ -s "$DL_PART" ]; then
        say "resuming: $(( $(stat -c %s "$DL_PART" 2>/dev/null || echo 0) / 1048576 )) MiB already in .part"
    fi
    if [ "$DL_MAXBYTES" -gt 0 ]; then say "(test hook: --max-download-mib $MAX_DL_MIB, stopping at $MAX_DL_MIB MiB)"; fi
    local auth=()
    if [ -n "$HF_TOKEN" ]; then auth=(-H "Authorization: Bearer $HF_TOKEN"); say '(using HF token from --hf-token/HF_TOKEN; never printed)'; fi
    say 'downloading ... (Ctrl-C aborts; partial data is kept for resume)'
    # curl's stderr goes to a log: the combined status line is the console
    # presentation, and the progress meter must not interleave it (measured
    # 2026-11-06, d20-parallel: the meter interleaved with the status line).
    DL_ERRLOG="${DL_PART}.curl-err.log"
    curl -fL -C - --retry 3 --retry-delay 5 ${auth[@]+"${auth[@]}"} -o "$DL_PART" "$DL_URL" 2> "$DL_ERRLOG" &
    DL_CURL_PID=$!
    DL_RC=0
}

poll_download() {
    # combined status line (~1 s tick) + fail-fast. Returns 0 when the
    # transfer completed, 2 when the --max-download-mib test hook tripped
    # (LOCAL_FILE is the .part, D33 semantics), or exits the wizard on a
    # hard failure of either leg.
    local t0 got btxt rc1f rc2f
    rc1f="${BUILD_RC1:-}"; rc2f="${BUILD_RC2:-}"
    # rc1f/rc2f are empty on the download-only path (build_start never ran);
    # '${...:-}' keeps 'set -u' happy (measured 2026-11-06: a bare $BUILD_RC2
    # died the wizard mid-download and orphaned the curl).
    t0=$(date +%s)
    while :; do
        if [ -z "$rc1f" ] && [ -z "$rc2f" ]; then
            # download-only path: the build was fresh or -SkipBuild
            btxt='not running (fresh or -SkipBuild)'
        elif [ -n "$rc2f" ] && [ -f "$rc2f" ]; then
            btxt='done'
        elif [ -n "$rc1f" ] && [ -f "$rc1f" ]; then
            btxt="building (cmake --build) ... $(( ( $(date +%s) - t0 ) / 60 )) min"
        else
            btxt="configuring ... $(( ( $(date +%s) - t0 ) / 60 )) min"
        fi
        got=$(stat -c %s "$DL_PART" 2>/dev/null || echo 0)
        printf '\r  build: %s   |   model: %s MiB downloaded      ' "$btxt" "$(( got / 1048576 ))"
        # fail-fast: a cmake stage exited nonzero
        if [ -n "$rc1f" ] && [ -f "$rc1f" ] && [ "$(head -1 "$rc1f")" != 0 ]; then
            printf '\n'; kill "$DL_CURL_PID" 2>/dev/null; wait "$DL_CURL_PID" 2>/dev/null
            build_kill; build_wait
        fi
        if [ -n "$rc2f" ] && [ -f "$rc2f" ] && [ "$(head -1 "$rc2f")" != 0 ]; then
            printf '\n'; kill "$DL_CURL_PID" 2>/dev/null; wait "$DL_CURL_PID" 2>/dev/null
            tail -50 "$BUILD_BLD_LOG" | sed 's/^/    /'
            fail "build failed (exit $(head -1 "$rc2f"); full log: $BUILD_BLD_LOG); the download was stopped (re-run resumes the .part)" 4
        fi
        # the test hook cap
        if [ "$DL_MAXBYTES" -gt 0 ] && [ "$got" -ge "$DL_MAXBYTES" ]; then
            kill "$DL_CURL_PID" 2>/dev/null
            wait "$DL_CURL_PID" 2>/dev/null
            got=$(stat -c %s "$DL_PART" 2>/dev/null || echo 0)
            if [ "$got" -gt "$DL_MAXBYTES" ]; then truncate -s "$DL_MAXBYTES" "$DL_PART"; got=$DL_MAXBYTES; fi
            printf '\n'
            warn "test hook stopped transfer at $(( got / 1048576 )) MiB (expected full ${M_SIZES_GIB[$MI_IDX]} GiB)"
            say 'leaving .part in place to verify resume; NOT renaming to the final name.'
            LOCAL_FILE="$DL_PART"
            return 2
        fi
        # curl finished?
        if ! kill -0 "$DL_CURL_PID" 2>/dev/null; then
            wait "$DL_CURL_PID"
            DL_RC=$?
            break
        fi
        sleep 1
    done
    printf '\n'
    if [ "$DL_RC" -ne 0 ]; then
        [ -f "$DL_ERRLOG" ] && tail -10 "$DL_ERRLOG" | sed 's/^/    /'
        fail "download failed (curl exit $DL_RC; manual: $DL_URL; log: $DL_ERRLOG); re-run resumes the .part" 5
    fi
    return 0
}

download_finalize() {
    # integrity + rename (unchanged from the original download tail)
    local file="$DL_FILE" part="$DL_PART"
    local got; got=$(stat -c %s "$part" 2>/dev/null || echo 0)
    [ "$got" -gt 0 ] || fail 'download produced an empty file' 5
    # a file of identical size is verified byte-identical before rename
    if [ -f "$file" ]; then
        local a b
        a=$(sha256sum "$file" | awk '{print $1}')
        b=$(sha256sum "$part" | awk '{print $1}')
        if [ "$a" = "$b" ]; then rm -f "$file"; else fail 'existing file has a different sha-256; manual check needed' 5; fi
    fi
    mv "$part" "$file"
    ok "model ready: $file"
    LOCAL_FILE="$file"
}

download_mmproj() {
    # VOD-6: fetch the vision projector with the model (same repo, curl -C - resume, token as for the
    # model), synchronously: it is 0.84 GiB, the combined status line is for the model. Nothing to do
    # without vision or when the file is there already.
    [ "$VISION" = 1 ] && [ -n "$MMPROJ_PATH" ] || return 0
    if [ -f "$MMPROJ_PATH" ]; then ok "projector: $MMPROJ_PATH"; return 0; fi
    local mm_repo="${M_MMPROJ_REPOS[$MI_IDX]:-${M_REPOS[$MI_IDX]}}"
    local url="https://huggingface.co/${mm_repo}/resolve/main/${M_MMPROJ[$MI_IDX]}"
    if [ "$DRY_RUN" = 1 ]; then
        say "(dry-run: not downloading the projector ${M_MMPROJ[$MI_IDX]} from ${mm_repo})"
        return 0
    fi
    [ "$SKIP_DOWNLOAD" = 1 ] && fail '--skip-download set but no local projector found (expected '"$MMPROJ_PATH"')' 5
    command -v curl >/dev/null 2>&1 || fail "curl is required for the download (install it: sudo apt install curl; manual: $url)" 5
    mkdir -p "$(dirname "$MMPROJ_PATH")" || fail "cannot create $(dirname "$MMPROJ_PATH")" 1
    say "projector: downloading ${M_MMPROJ[$MI_IDX]} ($(( ${M_MMPROJ_BYTES[$MI_IDX]} / 1048576 )) MiB) from Hugging Face"
    local auth=()
    if [ -n "$HF_TOKEN" ]; then auth=(-H "Authorization: Bearer $HF_TOKEN"); fi
    local part="$MMPROJ_PATH.part"
    curl -fL -C - --retry 3 --retry-delay 5 ${auth[@]+"${auth[@]}"} -o "$part" "$url" || fail "projector download failed (manual: $url); re-run resumes the .part" 5
    [ -s "$part" ] || fail 'projector download produced an empty file' 5
    mv "$part" "$MMPROJ_PATH"
    ok "projector ready: $MMPROJ_PATH"
}

wait_build() {
    # collect the build leg after the download finished (a download failure
    # has already exited 5 with the build killed)
    [ -n "${BUILD_PID:-}" ] || return 0
    build_wait
}

check_disk() {
    # SDD 4.1: disk >= model size + 15 GiB, confirm to proceed (default no)
    local dir="${1:-$LOCAL_FILE}"
    local need_gib=$(( ${M_SIZES_GIB_INT[$MI_IDX]} + 15 ))
    local avail_k
    avail_k=$(df --output=avail -k "$dir" 2>/dev/null | tail -1 | tr -d ' ')
    [ -z "$avail_k" ] && return 0
    local avail_gib=$(( avail_k / 1048576 ))
    if [ "$avail_gib" -lt "$need_gib" ]; then
        warn "disk has $avail_gib GiB free, the model plus build wants ~$need_gib GiB"
        if ! confirm 'proceed anyway?' n; then
            fail 'aborted: not enough disk space' 6
        fi
    fi
}

# --- steps 6-8: recipe, emission, start (SDD 4.6/4.7) ------------------------
pick_recipe() {
    # D20: recipe selection is part of the settings phase (no step number).
    say 'recipe (measured rows only):'
    # SDD D19: the rows were measured on each model's Default quant; picking
    # the alt build runs the row's flags against a file nobody measured.
    if [ -z "$VRAM" ]; then
        VRAM=$(ask_choice 'VRAM target?' 'full' 'full' '16g' '8g')
    fi
    if [ -z "$CONTEXT" ]; then
        CONTEXT=$(ask_choice 'Context size?' '262k' '262k' '128k')
    fi
    if ! recipe_for "${M_KEYS[$MI_IDX]}" "$VRAM" "$CONTEXT" "${M_QUANTS[$MI_IDX]}"; then
        recipe_avail "${M_KEYS[$MI_IDX]}"
        fail 'no measured recipe for the chosen model/quant/VRAM/context (rows above are the available combinations; the wizard only emits measured rows)' 3
    fi
    # SDD D21/D19: a fallback row (the model's default quant) was measured on another file
    if [ "${R_CLASS_FALLBACK:-0}" = 1 ]; then
        # SDD D24: the shared 8g MoE profile, measured on another model
        warn "row $R_ID is the shared 8g profile of every MoE model (measured on qwen36 / $R_QUANT); applied to ${M_KEYS[$MI_IDX]} / ${M_QUANTS[$MI_IDX]} (${M_FILES[$MI_IDX]}) - its speed/VRAM numbers are for the measured model"
    elif [ "$R_QUANT" != "${M_QUANTS[$MI_IDX]}" ]; then
        warn "row $R_ID was measured on quant $R_QUANT; you chose ${M_QUANTS[$MI_IDX]} (${M_FILES[$MI_IDX]}) - its speed/VRAM numbers are for $R_QUANT"
    fi
    say "recipe $R_ID: $R_MEASURED"
    [ -n "$R_NOTE" ] && say "note: $R_NOTE"
        # SDD D25 (2026-11-09): the owner's own run of this model through the shared profile
        if [ "${M_KEYS[$MI_IDX]}" = apex35 ] && [ "${M_QUANTS[$MI_IDX]}" = q5k_m ] && [ "$VRAM" = 8g ]; then
            say "measured on ${M_FILES[$MI_IDX]} with this row (owner, 8 GiB card, Linux): prompt 51,885 tokens at 537.42 t/s (96.5 s), 1,400 generated tokens at 30.88 t/s (32.39 ms per token)"
        fi
    # D12: warning only, threshold by VRAM target. D17: device-pinned rows
    # check the card they will actually use, not nvidia-smi's first line.
    local gpu_mib need=0 card='a 24 GiB card' gpu_line="$GPU_LINE"
    if [ -n "$R_DEVICE" ]; then
        local dev; dev=$(resolve_gpu_index "$VRAM")
        local mibs; mibs=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null)
        local sel; sel=$(echo "$mibs" | sed -n "$((dev + 1))p")
        [ -n "$sel" ] && gpu_line="$sel"
    fi
    gpu_mib=$(echo "$gpu_line" | sed -n 's/^[^,]*,\s*\([0-9,]*\)\s*MiB.*/\1/p' | tr -d ',')
    case "$VRAM" in
        full) need=22528 ;;
        16g)  need=10240 ;;
        8g)   need=8192 ;;
    esac
    if [ "$R_ID" = R7 ]; then card='an 8 GiB card (GTX 1070)'; fi
    if [ -n "$gpu_mib" ] && [ "$gpu_mib" -gt 0 ] && [ "$gpu_mib" -lt "$need" ]; then
        warn "GPU has $gpu_mib MiB; row $R_ID was measured on $card (SDD D12: warning only, the server's memory guard fails loudly)"
    fi
}

# SDD D17: CUDA index for device-pinned rows (R5/R7). --device N (DEVICE)
# wins; otherwise auto: full -> largest card, 16g/8g -> smallest card at or
# above the row minimum (10240/8192 MiB), else the smallest card (the D12
# warning then fires). Pure: echoes the index (the --device option wins).
# NOTE: no global cache - callers invoke it via $(...) subshells, where
# global assignments would die with the subshell (measured 2026-11-05).
resolve_gpu_index() {
    local vram="$1"
    if [ -n "$DEVICE" ]; then echo "$DEVICE"; return 0; fi
    local need=0
    case "$vram" in 16g) need=10240 ;; 8g) need=8192 ;; esac
    local rows
    rows=$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader 2>/dev/null)
    if [ -z "$rows" ]; then
        say 'nvidia-smi unavailable; pinning --device CUDA0 (override with --device N)' >&2
        echo 0; return 0
    fi
    local count=0 largest='' largest_mib=0 smallest='' smallest_mib=0 fit='' fit_mib=0
    local idx mib
    while IFS=',' read -r idx mib; do
        idx=${idx//[!0-9]/}; mib=${mib//[!0-9]/}
        [ -z "$idx" ] && continue
        count=$((count + 1))
        if [ -z "$largest" ] || [ "$mib" -gt "$largest_mib" ] || { [ "$mib" -eq "$largest_mib" ] && [ "$idx" -lt "$largest" ]; }; then largest=$idx; largest_mib=$mib; fi
        if [ -z "$smallest" ] || [ "$mib" -lt "$smallest_mib" ] || { [ "$mib" -eq "$smallest_mib" ] && [ "$idx" -lt "$smallest" ]; }; then smallest=$idx; smallest_mib=$mib; fi
        if [ "$need" -gt 0 ] && [ "$mib" -ge "$need" ] && { [ -z "$fit" ] || [ "$mib" -lt "$fit_mib" ] || { [ "$mib" -eq "$fit_mib" ] && [ "$idx" -lt "$fit" ]; }; }; then fit=$idx; fit_mib=$mib; fi
    done <<< "$rows"
    local pick=0 why=''
    if [ "$count" -gt 1 ]; then
        if [ "$vram" = full ]; then pick=$largest; why="largest card, ${largest_mib} MiB"
        elif [ -n "$fit" ]; then pick=$fit; why="smallest card >= ${need} MiB, ${fit_mib} MiB"
        else pick=$smallest; why="smallest card, ${smallest_mib} MiB (below the row minimum)"; fi
        # >&2: the index goes to stdout for command substitution (D17)
        say "multi-GPU system: pinning --device CUDA$pick ($why; override with --device N)" >&2
    else
        pick=$largest
    fi
    echo "$pick"
}

git_short() {
    local h
    h=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null | head -1)
    [ -n "$h" ] || h='unknown'
    echo "$h"
}

emit_run_script() {
    step 'Emit run-server.sh'
    local bind="$BIND"
    [ -z "$bind" ] && bind="$R_BIND"
    [ -z "$PORT" ] && PORT=8090   # D22: the settings phase normally fills PORT; this keeps the emitter self-contained
    local out="$REPO/run-server.sh" last="$REPO/run-server.last.sh"
    local log="$REPO/build/llama-cuda/server-$R_ID.log"
    if [ -f "$out" ]; then mv "$out" "$last"; say "previous run-server.sh saved as run-server.last.sh"; fi
    local secnote=''
    if [ "$bind" = '0.0.0.0' ]; then
        secnote="# NOTE: this recipe binds 0.0.0.0 (measured that way). If you do not need the server on
# the network, change --host 0.0.0.0 to --host 127.0.0.1 (see docs/guide/USAGE.md section 5)."
    fi
    local note=''
    [ -n "$R_NOTE" ] && note="# NOTE: $R_NOTE"
    local gs; gs=$(git_short)
    local date; date=$(date +%F)
    {
        echo '# run-server.sh - generated by setup.sh on '$date
        echo "# Recipe $R_ID (measured: $R_MEASURED)"
        echo "# Model: $LOCAL_FILE"
        echo "# Wizard: $gs  |  regenerate: ./setup.sh"
        [ -n "$note" ] && echo "$note"
        [ -n "$secnote" ] && echo "$secnote"
        echo
        echo 'set -u'
        echo 'set -o pipefail'
        echo "EXE=\"$BUILD_BIN\""
        echo "MODEL=\"$LOCAL_FILE\""
        if [ "$VISION" = 1 ]; then
            echo "MMPROJ=\"$MMPROJ_PATH\""
            echo '# vision (not part of the measured row): --mmproj-on-demand keeps the projector (~0.84 GiB) in host RAM and on the GPU only while a request'\''s images are encoded (docs/plans/vision-on-demand-results.md). For a text-only server remove `--mmproj "$MMPROJ" --mmproj-on-demand` below.'
            case "${M_KEYS[$MI_IDX]}" in qwen36|apex35) ;; *) echo '# vision on this model is NOT measured: --mmproj-on-demand was measured on Qwen3.6-35B-A3B only (apex35 shares its architecture). The first image request on this model is the test; to go back to text only, remove the two vision flags below.' ;; esac
        fi
        echo "PORT=$PORT"
        echo "LOG=\"$log\""
        # SDD D28: a default cap on the length of one response. -n is the limit for a request that names none of its own
        # (a request that sends max_tokens overrides it); a recipe that already sets -n keeps its own.
        local has_predict=0
        has_predict_flag "${R_FLAGS_OUT:-$R_FLAGS}" && has_predict=1
        if [ "$has_predict" = 0 ]; then
            echo '# max new tokens per response (-n) for a request that sets no max_tokens of its own: 16000. Raise it here if you need longer answers.'
            echo 'MAX_TOKENS=16000'
        fi
        if [ -n "$R_ENV" ]; then
            echo "# env (row $R_ID, verbatim): $R_ENV"
            echo "export $R_ENV"
        fi
        echo 'mkdir -p "$(dirname "$LOG")"'
        local flags="${R_FLAGS_OUT:-$R_FLAGS}"
        [ -n "$R_DEVICE" ] && flags="$flags --device CUDA$(resolve_gpu_index "$VRAM")"
        [ "$D22_ADDED_INT8" = 1 ] && echo '# D22: --lm-prefill-attn-int8 was ADDED for this card (compute capability <= 6.1, Pascal or lower): the pass flash attention takes the int8 Q.K^T tile, an approximation of the exact kernel (see the help text of --lm-prefill-attn-int8). Remove it for exact arithmetic.'
        local n_arg=''
        if [ "$has_predict" = 0 ]; then
            echo 'echo "max new tokens per response: $MAX_TOKENS (-n). Edit MAX_TOKENS at the top of this script if you need more (a request that sets max_tokens itself overrides it)."'
            n_arg='-n "$MAX_TOKENS" '
        fi
        echo "\"\$EXE\" -m \"\$MODEL\" --host $bind --port \"\$PORT\" $n_arg$flags 2>&1 | tee \"\$LOG\""
        echo
        echo '# stop: Ctrl-C in this console.  Health: curl http://127.0.0.1:'"$PORT"'/health'
        echo '# Pong test (USAGE.md section 5):'
        echo '# curl http://127.0.0.1:'"$PORT"'/v1/chat/completions -H "Content-Type: application/json" \'
        echo "#   -d '{\"model\":\"qwen3.6-35b-a3b\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly one word: pong\"}],\"max_tokens\":64,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}'"
    } > /tmp/.run-server.sh.out.$$
    if [ "$DRY_RUN" = 1 ]; then
        say '(dry-run: would write '$out')'
        sed 's/^/    /' /tmp/.run-server.sh.out.$$
        rm -f /tmp/.run-server.sh.out.$$
        return 0
    fi
    mv /tmp/.run-server.sh.out.$$ "$out"
    chmod +x "$out"
    ok "wrote $out"
}

start_server_now() {
    local script="$REPO/run-server.sh"
    if [ "$NO_START" = 1 ]; then say '(not starting: --no-start)'; return 0; fi
    if [ "$DRY_RUN" = 1 ]; then say '(dry-run: not starting)'; return 0; fi
    if confirm 'Start the server now?' n; then
        say 'starting server (Ctrl-C stops it here and in the server)...'
        bash "$script"
        local rc=$?
        [ "$rc" -ne 0 ] && warn "server exited with code $rc (log tail below)"
        local log="$REPO/build/llama-cuda/server-$R_ID.log"
        [ -f "$log" ] && tail -10 "$log" | sed 's/^/    /'
        return 0
    fi
    ok 'done. Run it any time with: bash ./run-server.sh'
}

# SDD D28: does a flag string already set the per-response token limit (-n / --n-predict / --predict)? Pure: returns 0 / 1.
has_predict_flag() {
    case " $1 " in *' -n '*|*' --n-predict '*|*' --predict '*|*' -n='*|*' --n-predict='*|*' --predict='*) return 0 ;; esac
    return 1
}

# --- main --------------------------------------------------------------------
echo
echo '  poormansllama setup wizard (Linux)'
echo

# D20: Ctrl-C in the parallel phase must stop both legs; the .part stays for
# resume. (The terminal already signals the whole process group; this is the
# safety net for the build subshell's children.)
on_interrupt() {
    [ -n "${BUILD_PID:-}" ] && { kill "$BUILD_PID" 2>/dev/null; pkill -P "$BUILD_PID" 2>/dev/null; }
    [ -n "${DL_CURL_PID:-}" ] && kill "$DL_CURL_PID" 2>/dev/null
    printf '\n'
    fail '[aborted by user]' 6
}
trap on_interrupt INT TERM

# EXIT net: if the wizard dies MID-DOWNLOAD for any reason (an internal error,
# a refusal, Ctrl-C handled elsewhere), the background curl must not be left
# orphaned (measured 2026-11-06: a set -u death left curl running for 97 s).
cleanup_download() {
    if [ -n "${DL_CURL_PID:-}" ] && kill -0 "$DL_CURL_PID" 2>/dev/null; then
        kill "$DL_CURL_PID" 2>/dev/null
    fi
}
trap cleanup_download EXIT

check_env
if [ ${#ENV_BAD[@]} -gt 0 ]; then
    install_missing
    check_env
fi
for b in "${ENV_BAD[@]:-}"; do
    [ -n "$b" ] || continue
    if [ "$b" = nvidia-smi ]; then
        warn 'no NVIDIA GPU/driver found. Install the driver first: https://www.nvidia.com/Download/index.aspx'
        fail 'NVIDIA driver/GPU required (not auto-installed)' 2
    fi
done

invoke_clone

if [ -n "$MODEL_FILE" ] && [ -z "$MODEL" ] && { [ "$SKIP_MODEL" != 1 ] || [ -n "$MODEL" ]; }; then
    : # MODEL_FILE without MODEL is refused below unless the no-ask shape holds
fi
if [ "$SKIP_MODEL" = 1 ] && [ -z "$MODEL" ]; then
    # AC-5 shape: env + clone + build only, nothing to ask. Sequential build:
    # there is no download leg to run in parallel with.
    invoke_build
    ok "done: environment + repository + build (model/recipe steps skipped: --skip-model without --model)"
    ok "next: re-run with --model qwen38|qwen36|flint|apex35 to pick a recipe and emit run-server.sh"
    exit 0
fi
if [ -n "$MODEL_FILE" ] && [ -z "$MODEL" ]; then
    fail '--model-file requires --model (qwen38|qwen36|flint|apex35) for recipe selection' 1
fi

# --- settings (SDD D20): every question BEFORE the heavy work --------------
step 'Settings (asked before the build and the download)'
pick_model
if [ "$SKIP_MODEL" = 1 ]; then
    LOCAL_FILE="${MODEL_DIR:-${M_DIRS[$MI_IDX]}}/${M_FILES[$MI_IDX]}"
    warn "--skip-model: assuming the model is at $LOCAL_FILE (not checked)"
else
    resolve_model_file
    # disk check needs the model size; it runs here, before the download
    check_disk "${MODEL_DIR:-${M_DIRS[$MI_IDX]}}"
fi
pick_recipe

# D22-C: the server port, asked with the other settings (before the heavy
# work). Default 8090; an explicit --port skips the question.
if [ -z "$PORT" ]; then
    if [ "$NI" = 1 ]; then
        PORT=8090
    else
        local_port=''
        while :; do
            printf '  Server port? ' >&2
            read -r local_port || local_port=''
            local_port="${local_port:-8090}"
            if [[ "$local_port" =~ ^[0-9]+$ ]] && [ "$local_port" -ge 1 ] && [ "$local_port" -le 65535 ]; then
                PORT="$local_port"; break
            fi
            printf '  (port must be a number 1-65535; got %s)\n' "$local_port" >&2
        done
    fi
fi

# VOD-6: vision support. The projector is fetched from the SAME Hugging Face repo as the model file and
# the emitted command gets --mmproj "$MMPROJ" --mmproj-on-demand (the projector then sits in host RAM
# and on the GPU only while a request's images are encoded). Default yes; --no-vision suppresses the
# question; --mmproj-file uses a local projector.
VISION=0
MMPROJ_PATH=''
if [ -n "$MMPROJ_FILE" ]; then
    [ "$NO_VISION" = 1 ] && fail '--mmproj-file and --no-vision contradict each other' 1
    vision_offered || fail "--mmproj-file: vision is not offered for ${M_KEYS[$MI_IDX]} / recipe $R_ID (no projector is published for this model, or the recipe already carries --mmproj)" 1
    [ -f "$MMPROJ_FILE" ] || fail "--mmproj-file does not exist: $MMPROJ_FILE" 1
    VISION=1
    MMPROJ_PATH="$MMPROJ_FILE"
    ok "vision: $MMPROJ_FILE (--mmproj-file)"
elif vision_offered && [ "$NO_VISION" != 1 ]; then
    if [ "$SKIP_MODEL" = 1 ]; then
        say 'vision: skipped with --skip-model (no download; pass --mmproj-file to use a local projector)'
    elif confirm "Vision support? Also fetches the projector ${M_MMPROJ[$MI_IDX]} ($(( ${M_MMPROJ_BYTES[$MI_IDX]} / 1048576 )) MiB) from ${M_MMPROJ_REPOS[$MI_IDX]:-${M_REPOS[$MI_IDX]}}$([ -n "${M_MMPROJ_REPOS[$MI_IDX]}" ] && echo " (a separate repo: the model's own repo publishes none)" || echo " (the model's repo)"); it is parked in host RAM between image requests" y; then
        VISION=1
        mm_dir="${MODEL_DIR:-${M_DIRS[$MI_IDX]}}"
        [ -n "$MODEL_FILE" ] && mm_dir="$(dirname "$MODEL_FILE")"
        MMPROJ_PATH="$mm_dir/${M_MMPROJ[$MI_IDX]}"
        if [ -f "$MMPROJ_PATH" ]; then ok "projector: $MMPROJ_PATH"; else say "projector: $MMPROJ_PATH (downloaded in the parallel phase)"; fi
    fi
elif [ -n "${M_MMPROJ[$MI_IDX]}" ] && ! vision_offered; then
    say 'vision: not offered for this recipe (it already carries --mmproj)'
elif [ -z "${M_MMPROJ[$MI_IDX]}" ] && [ "$NO_VISION" != 1 ] && [ -z "$MMPROJ_FILE" ]; then
    say "vision: not available for ${M_KEYS[$MI_IDX]} (no vision projector is published for it; pass --mmproj-file to use your own)"
fi

# D22-A/B: complete the row's flags (pure; see complete_row_flags). The
# compute capability is for the card the run will use; on failure the row
# is emitted as measured (the int8 tile is a performance approximation, not
# a correctness requirement).
GPU_CC=''
if GPU_CC=$(detect_gpu_arch); then
    say "card compute capability: $GPU_CC"
else
    say 'compute capability unavailable (nvidia-smi); the row flags stay as measured (no D22 additions)'
fi
R_FLAGS_OUT=$(complete_row_flags "$R_FLAGS" "$R_ENV" "$GPU_CC" "$VISION")
# the note is about the flag being ADDED (the subshell above cannot set
# globals): present in the output and absent from the measured row
D22_ADDED_INT8=0
case "$R_FLAGS_OUT" in *'--lm-prefill-attn-int8'*)
    case "$R_FLAGS" in *'--lm-prefill-attn-int8'*) ;; *) D22_ADDED_INT8=1 ;; esac ;;
esac

# --- parallel phase (SDD D20): build leg and download leg ------------------
BUILD_BIN="$REPO/build/llama-cuda/bin/llama-server"
build_needed=1
if [ "$SKIP_BUILD" = 1 ]; then
    build_needed=0
    say '(skipped: --skip-build)'
elif [ "$FULL_BUILD" != 1 ] && binary_fresh "$REPO" "$BUILD_BIN"; then
    build_needed=0
    ok "build skipped (binary newer than sources): $BUILD_BIN"
fi

if [ "$DRY_RUN" = 1 ]; then
    # dry-run: no processes. Same command text as the sequential build.
    if [ "$build_needed" = 1 ]; then
        if [ -z "$CUDA_ARCH" ]; then
            if ! CUDA_ARCH=$(detect_cuda_arch); then CUDA_ARCH='? (nvidia-smi unavailable)'; fi
            say "detected CUDA arch: $CUDA_ARCH"
        fi
        say "(dry-run: cmake -S $REPO/third_party/llama.cpp -B $REPO/build/llama-cuda -DGGML_CUDA=ON -DLLAMA_LAYER_STREAMING=ON -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH -DCMAKE_BUILD_TYPE=Release)"
        say "(dry-run: cmake --build $REPO/build/llama-cuda --target llama-server --parallel $(nproc 2>/dev/null || echo 4))"
    fi
    if [ -n "$MODEL_FILE_TARGET" ]; then
        say '(dry-run: not downloading)'
        # the emitted script must name the file the real run would produce
        LOCAL_FILE="$MODEL_FILE_TARGET"
    fi
    download_mmproj          # VOD-6: in a dry run it only announces the projector download
elif [ "$build_needed" = 1 ] && [ -n "$MODEL_FILE_TARGET" ]; then
    step 'Build + download (in parallel; Ctrl-C keeps the .part for resume)'
    build_start
    download_start "${MODEL_DIR:-${M_DIRS[$MI_IDX]}}"
    poll_download
    prc=$?
    if [ "$prc" = 0 ]; then
        download_finalize
        download_mmproj      # VOD-6: the projector follows the model while the build leg keeps running
    fi
    # prc = 2: test-hook cap tripped, LOCAL_FILE is the .part (D33 semantics)
    wait_build
elif [ "$build_needed" = 1 ]; then
    # no download leg: build sequentially in the foreground (live output)
    invoke_build
    download_mmproj          # VOD-6: the projector, when vision is on and the file is not there yet
else
    # build fresh/skipped; download only (no combined line needed)
    if [ -n "$MODEL_FILE_TARGET" ]; then
        step 'Download model (Hugging Face; Ctrl-C keeps the .part for resume)'
        local_dir="${MODEL_DIR:-${M_DIRS[$MI_IDX]}}"
        download_start "$local_dir"
        poll_download
        prc=$?
        if [ "$prc" = 0 ]; then download_finalize; fi
    fi
    download_mmproj          # VOD-6
fi

check_row_flags "$BUILD_BIN" "${R_FLAGS_OUT:-$R_FLAGS}" "$R_ID"
emit_run_script
start_server_now

ok 'wizard finished'
exit 0
