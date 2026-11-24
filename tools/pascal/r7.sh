# usage: r7.sh LABEL NP "extra flags" [par_bench args...]
L=$1; NP=$2; X=$3; shift 3
R7="-n 16000 -cram 4096 --ctx-checkpoints 16 -c 262144 -ctk q8_0 -ctv q8_0 -fa on -np $NP --hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 24 -b 32768 -ub 512 --lm-prefill-ubatch 2048 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 2 --hybrid-moe-prefill-free-all --lm-prefill-segment 16384 --lm-prefill-attn-int8 --mmproj $HOME/models/qwen-3.6/mmproj-BF16.gguf --mmproj-on-demand $X"
flock /tmp/gpu-2x1070.lock python3 -I tools/pascal/par_bench.py --label $L --flags "$R7" --env LLAMA_MOE_HYBRID_LEAN_STALL_US=10000 --env LLAMA_FATTN_GQA8_MAXROWS=4 "$@"
