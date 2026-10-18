#!/bin/bash
# G42 stability and memory, build s53 (np 1): soaks (fuse; fuse + prearm auto; canonical + fuse + prearm always), growth, needles, fault injection again, memcheck of a fused pass with snapshots and of the armed / disarmed passes.
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
until [ -f build/llama-cuda/k5500-chain.done ]; do sleep 20; done
sleep 25
rm -f build/llama-cuda/k5600-chain.done
F=tools/server-fixtures
P=build/llama-cuda/gplr07-qa-prompts
B=s53
ABG="$F/agentic-A.json,$F/agentic-B.json,$F/agentic-G.json"
L() { echo "--exe=build/llama-cuda/bin/$B/llama-server.exe --ub 0 --batch 0 --no-hybrid --no-override --env LLAMA_HYBRID_FREE_WATCH_MS=20 --env LLAMA_LM_LOG=1 --arg=--hybrid-moe --arg=--lm-prefill --arg=-fitt --arg=16000"; }
SK="--fixtures $ABG --requests 160 --workers 2 --max-ctx 40000 --rss-allow-mib 1024"
bash tools/run-server-soak.sh k5601-soak-def-fuse "$SK" -- $(L) --arg=--lm-prefill-fuse --arg=--cache-ram --arg=0 > build/llama-cuda/k5601.out 2>&1
bash tools/run-server-soak.sh k5602-soak-def-fuse-auto "$SK" -- $(L) --arg=--lm-prefill-fuse --arg=--lm-prefill-prearm --arg=auto --arg=--cache-ram --arg=0 > build/llama-cuda/k5602.out 2>&1
bash tools/run-server-soak.sh k5603-soak-canon-fuse-always "$SK" -- $(L) --arg=--lm-prefill-canonical --arg=--lm-prefill-fuse --arg=--lm-prefill-prearm --arg=always --arg=--cache-ram --arg=0 > build/llama-cuda/k5603.out 2>&1
echo done > build/llama-cuda/k5600-part1.done
bash tools/run-server-mt.sh k5604-growth-fuse-auto run "--fixture $F/agentic-G.json --scenarios growth --reprocess-allow 2112" -- $(L) --arg=--lm-prefill-fuse --arg=--lm-prefill-prearm --arg=auto > build/llama-cuda/k5604.out 2>&1
bash tools/run-server-mt.sh k5605-growth-canon-fuse-always run "--fixture $F/agentic-G.json --scenarios growth --reprocess-allow 2112" -- $(L) --arg=--lm-prefill-canonical --arg=--lm-prefill-fuse --arg=--lm-prefill-prearm --arg=always > build/llama-cuda/k5605.out 2>&1
bash tools/run-server-needle.sh k5606-needle-fuse-always "--prompt $P/anchor-24000.txt --prompt $P/needle-50000.txt --prompt $P/needle-100000.txt" -- $(L) --arg=--lm-prefill-fuse --arg=--lm-prefill-prearm --arg=always > build/llama-cuda/k5606.out 2>&1
run() { bash tools/run-g42-arm.sh "$@" > /dev/null 2>&1; }
run k5611-fi-run1-fuse  s53 agentic --ho "--max-turns 5" --env LLAMA_LM_FAIL_RUN=1 --env LLAMA_LM_FAIL_AT=0 --arg --lm-prefill-fuse
run k5612-fi-run2-fuse  s53 agentic --ho "--max-turns 5" --env LLAMA_LM_FAIL_RUN=2 --env LLAMA_LM_FAIL_AT=3 --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg always
run k5613-fi-acq-fuse   s53 agentic --ho "--max-turns 5" --env LLAMA_LM_FAIL_ACQUIRE=1 --arg --lm-prefill-fuse --arg --lm-prefill-prearm --arg always
echo done > build/llama-cuda/k5600-part2.done
CS="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/compute-sanitizer/compute-sanitizer.exe"
WD=C:/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
export SERVER_GRACEFUL_STOP=1 SERVER_GRACEFUL_WAIT=1800 SERVER_STOP_MODE=child
mkdir -p build/llama-cuda/k5621-memcheck-fuse-always
bash tools/run-server-mt.sh k5621-memcheck-fuse-always run "--fixture $F/agentic-A.json --scenarios agentic --max-turns 7 --reprocess-allow 2112" -- $(L) --arg=--lm-prefill-fuse --arg=--lm-prefill-prearm --arg=always --arg=--cache-ram --arg=0 "--wrap=$CS" --wrap=--tool --wrap=memcheck --wrap=--print-limit --wrap=500 --wrap=--log-file --wrap=$WD/build/llama-cuda/k5621-memcheck-fuse-always/sanitizer.txt > build/llama-cuda/k5621.out 2>&1
echo done > build/llama-cuda/k5600-chain.done
