# G42 session chains (exact command lines of the measured runs)

These are the shell chains that produced the G42 evidence (`docs/plans/llama-server-integration.md`
section G42, `docs/plans/llama-server-integration-results.md` Round 7). They were run from the repo
root under the GPU mutex (`tools/run-g42-arm.sh`, `tools/run-server-mt.sh`, ... call
`tools/with-gpu-lock.ps1`). Results of a run land in `build/llama-cuda/<tag>/` (ignored by git).

| chain | build | what |
|---|---|---|
| k5100 | s48 | first fused run (zero think time) |
| k5200 | s47/s50 | speed arms, fixture A, 3 s think time, A B B A |
| k5310 | s50 | fault injection with --lm-prefill-fuse |
| k5330 | s47/s51 | small-turn fixture S, fuse-min sweep |
| k5350 | s47/s51 | identity, the five hit-versus-fresh checks (canonical + fuse), replicated pairs, default mode |
| k5390 | s47/s51 | checkpoint positions (trace log) and canonical checkpoint hashes |
| k5450 | s52 | restore-time diagnostic (undrained idle arm) |
| k5500 | s47/s53 | FINAL binary: speed arms A / S / canonical, identity, first token |
| k5600 | s53 | FINAL binary: soaks, growth, needles, fault injection, memcheck |
| k5800 | s53 | closing verification (canonical + fuse + prearm always) |

k5400 and k5700 were cancelled before they ran (replaced by k5600 and k5800); k6001 (canonical + fuse,
512-token chunks, stock decode, swapped suite) was a single command run by hand:
`tools/run-server-mt.sh k6001-c512-fuse-lmstock-swap run "<swapped suite options, --reprocess-allow 600>" -- <stock-decode server options> --arg=--lm-prefill --arg=--lm-prefill-canonical --arg=--lm-prefill-fuse --arg=--lm-prefill-ubatch --arg=512`.
