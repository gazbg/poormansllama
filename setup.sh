#!/usr/bin/env bash
# ============================================================================
# setup.sh - poormansllama setup wizard (Linux)
#
# Purpose: check the environment (git, cmake, CUDA toolkit, gcc, NVIDIA GPU),
# auto-install missing toolchain pieces via apt (y/n confirmed), clone the
# repo if not already inside a checkout, build llama-server with the fork's
# required options (GGML_CUDA=ON, LLAMA_LAYER_STREAMING=ON), ask a few
# questions (model, VRAM target, context), download the selected GGUF from
# Hugging Face if missing (resume-capable), pick one of the seven MEASURED
# recipe rows (R1-R7, unmeasured combinations are refused), and emit a
# ready-to-run run-server.sh in the repo root. Optionally starts the server.
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
PORT=8033
BIND=''
DEVICE=''     # CUDA index for device-pinned rows R5/R7; empty = auto (SDD D17)
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

usage() {
    cat <<'EOF'
usage: setup.sh [options]
  --model qwen38|qwen36|flint   model to use (asked if omitted)
  --vram full|16g|8g            VRAM target (asked if omitted)
  --context 262k|128k           context size (asked if omitted)
  --port N                      server port (default 8033)
  --host ADDR                   bind address (default: the recipe row's own)
  --device N                    CUDA index for device-pinned rows R5/R7 (default: auto)
  --model-file PATH             explicit local GGUF (still needs --model)
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
        --vram) VRAM="$2"; shift 2 ;;
        --context) CONTEXT="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --host) BIND="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --model-file) MODEL_FILE="$2"; shift 2 ;;
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

# ask_choice "Question" KEY1 KEY2 ...
# -> echoes the chosen key (default: first key)
# IMPORTANT: menu prints go to stderr (>&2) so $() command substitution
# captures only the final selection echo to stdout. If we used stdout for
# the menu, the user would never see it.
ask_choice() {
    local q="$1"; shift
    local keys=("$@")
    local def="${keys[0]}"
    if [ "${NI:-0}" = "1" ]; then echo "$def"; return; fi
    for i in "${!keys[@]}"; do
        printf '    %d) %s\n' "$((i+1))" "${keys[$i]}" >&2
    done
    while :; do
        printf '  %s (1-%d, default 1) ' "$q" "${#keys[@]}" >&2
        read -r a || a=""
        a="${a:-1}"
        # Normalise: if user typed a number, convert to key
        if [[ "$a" =~ ^[0-9]+$ ]]; then
            if [ "$a" -ge 1 ] 2>/dev/null && [ "$a" -le "${#keys[@]}" ] 2>/dev/null; then
                echo "${keys[$((a-1))]}"; return
            fi
            a=1
        fi
        # Check if user typed one of the keys directly
        for k in "${keys[@]}"; do
            if [ "$a" = "$k" ]; then echo "$a"; return; fi
        done
        # Default: return the first key
        echo "${keys[0]}"; return
    done
}

# --- model table (SDD 4.5; links verified 2026-11-04) -----------------------
MODEL_KEYS=(qwen38 qwen36 flint)
MODEL_REPOS=('unsloth/Qwen3.8-27B-GGUF' 'unsloth/Qwen3.6-35B-A3B-GGUF' 'unsloth/flint-GGUF')
MODEL_FILES=('Qwen3.8-27B-UD-Q4_K_M.gguf' 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf' 'flint-UD-Q5_K_XL.gguf')
MODEL_SIZES_GIB=(15.4 20.6 27.9)
MODEL_SIZES_GIB_INT=(16 21 28)
MODEL_KINDS=('dense hybrid' 'MoE' 'SWA MoE')
MODEL_DIRS=("$HOME/models/qwen-3.8" "$HOME/models/qwen-3.6" "$HOME/models/flint")

mi() {  # mi KEY -> sets MI_IDX or exits 1
    case "$1" in
        qwen38) MI_IDX=0 ;;
        qwen36) MI_IDX=1 ;;
        flint)  MI_IDX=2 ;;
        1|2|3)  MI_IDX=$(( $1 - 1 )) ;;
        *) fail "unknown model '$1' (choose qwen38, qwen36, flint, or 1-3)" 1 ;;
    esac
}

# --- recipe table (SDD 4.6; rows verbatim from the measured commands) -------
# recipe_for MODEL VRAM CTX -> sets R_ID, R_FLAGS, R_BIND, R_MEASURED, R_NOTE,
# R_ENV; returns 1 if the combination is not measured.
recipe_for() {
    R_ID=''; R_FLAGS=''; R_BIND=''; R_MEASURED=''; R_NOTE=''; R_ENV=''; R_DEVICE=''
    local m="$1" v="$2" c="$3"
    if [ "$m" = qwen38 ] && [ "$v" = full ] && [ "$c" = 262k ]; then
        R_ID=R1; R_BIND=127.0.0.1
        R_FLAGS='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 -t 16 --no-warmup --lm-prefill-host-layers 0-37 --lm-decode-stream dynamic --lm-decode-slots 2 --lm-prefill --spec-type draft-mtp --spec-draft-p-min 0.6 --spec-draft-n-min 0 --spec-draft-n-max 3 --spec-mtp-defer -ctkd q8_0 -ctvd q8_0 -lv 4'
        R_MEASURED='PP 1093.25 t/s, TG 14.95 t/s (owner 2026-10-29, RTX 5090 24 GiB)'
        return 0
    fi
    if [ "$m" = qwen38 ] && [ "$v" = 16g ] && [ "$c" = 262k ]; then
        R_ID=R2; R_BIND=127.0.0.1
        R_FLAGS='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 -t 16 --no-warmup --lm-prefill-host-layers 0-63 --lm-decode-stream dynamic --lm-decode-slots 2 --lm-prefill --spec-type draft-mtp --spec-draft-p-min 0.6 --spec-draft-n-min 0 --spec-draft-n-max 3 --spec-mtp-defer -ctkd q8_0 -ctvd q8_0 -lv 4'
        R_MEASURED='PP 1007.87 t/s, TG 9.96 t/s (owner 2026-10-29, RTX 5090 24 GiB)'
        return 0
    fi
    if [ "$m" = qwen38 ] && [ "$v" = 8g ] && [ "$c" = 128k ]; then
        R_ID=R3; R_BIND=127.0.0.1
        R_FLAGS='-c 128000 -ctk q4_0 -ctv q4_0 -fa on -np 1 -t 16 --no-warmup --lm-prefill-host-layers 0-63 --lm-decode-stream dynamic --lm-decode-slots 2 --lm-prefill --lm-prefill-prearm always -lv 4'
        R_MEASURED='PP 36.95 t/s, TG 9.92 t/s (owner 2026-10-29, RTX 5090 24 GiB)'
        return 0
    fi
    if [ "$m" = flint ] && [ "$v" = full ] && [ "$c" = 262k ]; then
        R_ID=R4; R_BIND=0.0.0.0
        R_FLAGS='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 24 -b 262000 -ub 2048 --lm-prefill-ubatch 2048 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 3 -lv 4 --lm-prefill-copy-piece 64'
        R_MEASURED='PP 2422.94 t/s, TG 53.80 t/s (owner 2026-11-02, RTX 5090 24 GiB)'
        return 0
    fi
    if [ "$m" = qwen36 ] && [ "$v" = full ] && [ "$c" = 262k ]; then
        R_ID=R5; R_BIND=0.0.0.0
        R_FLAGS='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 24 -b 262000 -ub 2048 --lm-prefill-ubatch 2048 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 3 -lv 4 --hybrid-moe-prefill-free-all'
        R_DEVICE=auto
        R_MEASURED='owner working recipe 2026-11-02 (RTX 5090 24 GiB)'
        return 0
    fi
    if [ "$m" = qwen36 ] && [ "$v" = 16g ] && [ "$c" = 262k ]; then
        R_ID=R6; R_BIND=127.0.0.1
        R_FLAGS='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe --lm-prefill --lm-prefill-canonical --lm-prefill-fuse --lm-prefill-prearm auto --lm-prefill-ubatch 2048 --lm-prefill-segment 32768 -b 32768 --hybrid-moe-prefill-ubatch 0 -fitt 16000 --ctx-checkpoints 32'
        R_MEASURED='G42 canonical: TG 3.0-3.7x stock, PP 3.65-4.11x stock (docs/guide/USAGE.md, build s53)'
        R_NOTE='measured on Qwen3.6-35B-A3B-MTP-UD-Q4_K_S; the wizard offers the closest HF file, Q4_K_M (SDD D13)'
        return 0
    fi
    if [ "$m" = qwen36 ] && [ "$v" = 8g ] && [ "$c" = 262k ]; then
        R_ID=R7; R_BIND=0.0.0.0
        R_FLAGS='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 32 -b 32768 -ub 2048 --lm-prefill-ubatch 2048 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 2 -lv 4 --hybrid-moe-prefill-free-all --lm-prefill-segment 32768 -sm none -mg 0'
        R_DEVICE=auto
        R_MEASURED='owner working recipe 2026-11-05 (8 GiB target, binds 0.0.0.0)'
        R_ENV='LLAMA_FATTN_TILE_Q8QK=1'
        R_NOTE='the emitted script sets env LLAMA_FATTN_TILE_Q8QK=1 (int8 attention; part of the 8g row); the old LINUX-PASCAL.md fuse/prearm recipe is historical'
        return 0
    fi
    return 1
}

recipe_avail() {  # recipe_avail MODEL -> prints the available rows
    local m="$1"
    say "no measured recipe for $m / $VRAM / $CONTEXT - available rows for $m:"
    local v c
    for v in full 16g 8g; do
        for c in 262k 128k; do
            if recipe_for "$m" "$v" "$c"; then
                printf '    %-3s %-5s %-6s %s\n' "$R_ID" "$v" "$c" "$R_MEASURED"
            fi
        done
    done
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
    step '1. Environment check'
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
    step '2. Install missing tools (apt)'
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
    step '3. Repository'
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
    step '4. Build llama-server'
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
        mi "$MODEL"
        return 0
    fi
    local i opts=()
    for i in 0 1 2; do
        opts+=("$(printf '%d) %s: %s - %s, %s GiB' "$((i+1))" "${MODEL_KEYS[$i]}" "${MODEL_KINDS[$i]}" "${MODEL_FILES[$i]}" "${MODEL_SIZES_GIB[$i]}")")
    done
    if [ "$NI" = 1 ]; then MI_IDX=1; return 0; fi   # default: qwen36 (the vehicle model)
    local sel
    sel=$(ask_choice 'Which model?' "${opts[@]}")
    case "$sel" in
        *qwen38*) MI_IDX=0 ;;
        *qwen36*) MI_IDX=1 ;;
        *flint*)  MI_IDX=2 ;;
        *) fail "could not parse model choice '$sel'" 1 ;;
    esac
}

resolve_model_file() {
    # sets LOCAL_FILE; downloads when missing
    local dir="$MODEL_DIR"
    [ -z "$dir" ] && dir="${MODEL_DIRS[$MI_IDX]}"
    local file="$dir/${MODEL_FILES[$MI_IDX]}"
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
        say "no ${MODEL_FILES[$MI_IDX]} in $dir, but found:"
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
        # fall through: download
    fi
    download_model "$dir"
}

download_model() {
    local dir="$1"
    local file="$dir/${MODEL_FILES[$MI_IDX]}"
    local part="$file.part"
    local url="https://huggingface.co/${MODEL_REPOS[$MI_IDX]}/resolve/main/${MODEL_FILES[$MI_IDX]}"
    step "5. Download model (${MODEL_SIZES_GIB[$MI_IDX]} GiB) from Hugging Face"
    say "$url"
    if [ "$SKIP_DOWNLOAD" = 1 ]; then fail '--skip-download set but no local model found (expected $file)' 5; fi
    if [ "$DRY_RUN" = 1 ]; then say '(dry-run: not downloading)'; LOCAL_FILE="$file"; return 0; fi
    command -v curl >/dev/null 2>&1 || fail 'curl is required for the download (install it: sudo apt install curl; manual: $url)' 5
    local auth=()
    if [ -n "$HF_TOKEN" ]; then auth=(-H "Authorization: Bearer $HF_TOKEN"); say '(using HF token from --hf-token/HF_TOKEN; never printed)'; fi
    if [ "$MAX_DL_MIB" -gt 0 ]; then say "(test hook: --max-download-mib $MAX_DL_MIB, stopping at $MAX_DL_MIB MiB)"; fi
    say 'downloading ... (Ctrl-C aborts; partial data is kept for resume)'
    local maxbytes=0
    [ "$MAX_DL_MIB" -gt 0 ] && maxbytes=$(( MAX_DL_MIB * 1048576 ))
    if [ -s "$part" ]; then
        say "resuming: $(( $(stat -c %s "$part" 2>/dev/null || echo 0) / 1048576 )) MiB already in .part"
    fi
    local rc=0 got
    if [ "$maxbytes" -gt 0 ]; then
        # test hook: run curl in the background and stop it once the transfer
        # reaches the cap, simulating an interrupted download.
        curl -fsSL -C - --retry 3 --retry-delay 5 ${auth[@]+"${auth[@]}"} -o "$part" "$url" &
        local curlpid=$!
        while kill -0 "$curlpid" 2>/dev/null; do
            got=$(stat -c %s "$part" 2>/dev/null || echo 0)
            [ "$got" -ge "$maxbytes" ] && break
            sleep 0.2
        done
        kill "$curlpid" 2>/dev/null
        wait "$curlpid" 2>/dev/null
        got=$(stat -c %s "$part" 2>/dev/null || echo 0)
        [ "$got" -gt "$maxbytes" ] && truncate -s "$maxbytes" "$part" && got="$maxbytes"
        [ "$got" -gt 0 ] || fail "test hook download failed before $MAX_DL_MIB MiB (manual: $url)" 5
        warn "test hook stopped transfer at $(( got / 1048576 )) MiB (expected full ${MODEL_SIZES_GIB[$MI_IDX]} GiB)"
        say 'leaving .part in place to verify resume; NOT renaming to the final name.'
        LOCAL_FILE="$part"
        return 0
    fi
    curl -fL -C - --retry 3 --retry-delay 5 ${auth[@]+"${auth[@]}"} -o "$part" "$url" || rc=$?
    if [ "$rc" -ne 0 ]; then
        fail "download failed (curl exit $rc; manual: $url)" 5
    fi
    got=$(stat -c %s "$part" 2>/dev/null || echo 0)
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

check_disk() {
    # SDD 4.1: disk >= model size + 15 GiB, confirm to proceed (default no)
    local dir="${1:-$LOCAL_FILE}"
    local need_gib=$(( ${MODEL_SIZES_GIB_INT[$MI_IDX]} + 15 ))
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
    step '6. Recipe (measured only)'
    if [ -z "$VRAM" ]; then
        VRAM=$(ask_choice 'VRAM target?' 'full' '16g' '8g')
    fi
    if [ -z "$CONTEXT" ]; then
        CONTEXT=$(ask_choice 'Context size?' '262k' '128k')
    fi
    if ! recipe_for "${MODEL_KEYS[$MI_IDX]}" "$VRAM" "$CONTEXT"; then
        recipe_avail "${MODEL_KEYS[$MI_IDX]}"
        fail 'no measured recipe for the chosen model/VRAM/context (rows above are the available combinations; the wizard only emits measured rows)' 3
    fi
    say "recipe $R_ID: $R_MEASURED"
    [ -n "$R_NOTE" ] && say "note: $R_NOTE"
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
# warning then fires). Result cached in GPU_DEV.
resolve_gpu_index() {
    local vram="$1"
    if [ -n "$DEVICE" ]; then GPU_DEV="$DEVICE"; echo "$GPU_DEV"; return 0; fi
    if [ -n "$GPU_DEV" ]; then echo "$GPU_DEV"; return 0; fi
    local need=0
    case "$vram" in 16g) need=10240 ;; 8g) need=8192 ;; esac
    local rows
    rows=$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader 2>/dev/null)
    if [ -z "$rows" ]; then
        say 'nvidia-smi unavailable; pinning --device CUDA0 (override with --device N)' >&2
        GPU_DEV=0; echo 0; return 0
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
    GPU_DEV=$pick
    echo "$pick"
}

git_short() {
    local h
    h=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null | head -1)
    [ -n "$h" ] || h='unknown'
    echo "$h"
}

emit_run_script() {
    step '7. Emit run-server.sh'
    local bind="$BIND"
    [ -z "$bind" ] && bind="$R_BIND"
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
        echo "PORT=$PORT"
        echo "LOG=\"$log\""
        if [ -n "$R_ENV" ]; then
            echo "# env (row $R_ID, verbatim): $R_ENV"
            echo "export $R_ENV"
        fi
        echo 'mkdir -p "$(dirname "$LOG")"'
        local flags="$R_FLAGS"
        [ -n "$R_DEVICE" ] && flags="$flags --device CUDA$(resolve_gpu_index "$VRAM")"
        echo "\"\$EXE\" -m \"\$MODEL\" --host $bind --port \"\$PORT\" $flags 2>&1 | tee \"\$LOG\""
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

# --- main --------------------------------------------------------------------
echo
echo '  poormansllama setup wizard (Linux)'
echo

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
invoke_build

if [ -n "$MODEL_FILE" ] && [ -z "$MODEL" ]; then
    fail '--model-file requires --model (qwen38|qwen36|flint) for recipe selection' 1
fi
if [ "$SKIP_MODEL" = 1 ] && [ -z "$MODEL" ]; then
    # env + clone + build only, nothing to ask
    ok "done: environment + repository + build (model/recipe steps skipped: --skip-model without --model)"
    ok "next: re-run with --model qwen38|qwen36|flint to pick a recipe and emit run-server.sh"
    exit 0
fi
pick_model
if [ "$SKIP_MODEL" = 1 ]; then
    LOCAL_FILE="${MODEL_DIR:-${MODEL_DIRS[$MI_IDX]}}/${MODEL_FILES[$MI_IDX]}"
    warn "--skip-model: assuming the model is at $LOCAL_FILE (not checked)"
else
    resolve_model_file
    check_disk
fi
pick_recipe
check_row_flags "$BUILD_BIN" "$R_FLAGS" "$R_ID"
emit_run_script
start_server_now

ok 'wizard finished'
exit 0
