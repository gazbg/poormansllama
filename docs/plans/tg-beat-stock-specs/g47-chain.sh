#!/bin/bash
# G27 / G28 / G29 chain on build g47 (g45 + the sampler patch + the segmented-prefill patch + G29 guard + G28b scheduler pre-reserve; all opt-in or identity-preserving). g91-ident already ran (PASS).
# Order: default-off identity first, then the prefill sessions, the G29 confirmation, then the sampled and TG sessions.
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
bash docs/plans/tg-beat-stock-specs/run-session-queue.sh g47 g90-ident g91-pp-50k g91-pp-100k g91-pp-200k g91-pp-250k g92-bm1024-200k g90-sampled-B g90-sampled-A g91-tg-50k g91-tg-200k
echo finished > build/llama-cuda/g47-chain.finished
