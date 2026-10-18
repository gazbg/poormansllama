#!/bin/bash
# G30 / M3 chain on build g48: B0 against B0 + MTP on the best prefill of each band (stock path, CPU/GPU interleave, layer-major, segmented), code and prose per length.
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
bash docs/plans/tg-beat-stock-specs/run-session-queue.sh g48 g94-lc4k g94-sc24 g94-sp24 g94-sc500 g94-sp500 g94-ic1k g94-ip1k g94-ic2k g94-ip2k g94-lp4k g94-lc24k g94-lp24k g94-sgc50k g94-sgp50k
echo finished > build/llama-cuda/g94-chain.finished
