#!/bin/bash
# G42 correctness, build s51 (np 1): GA-1 identity (smoke A B B A, s47 against s51 without the new flags), GA-5 the five hit-versus-fresh checks in six sessions with --lm-prefill-canonical --lm-prefill-fuse
# (stock decode, frozen cache, dynamic cache x original / swapped fixtures), the replicated pairs, and the default mode with and without --lm-prefill-fuse.
cd /c/Development/AI/qwen38GZI/.claude/worktrees/attribution-handoff-doc-8ed1d1
until [ -f build/llama-cuda/k5330-chain.done ]; do sleep 20; done
sleep 25
rm -f build/llama-cuda/k5350-chain.done
F=tools/server-fixtures
B=s51
eval "$(grep '^PL=' build/llama-cuda/k4200-chain.sh)"
SMK=tools/server-scenarios/smoke.json
LMS="--ub 0 --batch 0 --no-hybrid --no-override --env LLAMA_HYBRID_FREE_WATCH_MS=20 --env LLAMA_LM_LOG=1 --arg=--hybrid-moe --arg=--lm-prefill --arg=-fitt --arg=16000"
# 1. identity
bash tools/run-server-probe.sh k5351-id-A1 $SMK --exe=build/llama-cuda/bin/s47/llama-server.exe $LMS > build/llama-cuda/k5351.out 2>&1
bash tools/run-server-probe.sh k5352-id-B1 $SMK --exe=build/llama-cuda/bin/$B/llama-server.exe $LMS > build/llama-cuda/k5352.out 2>&1
bash tools/run-server-probe.sh k5353-id-B2 $SMK --exe=build/llama-cuda/bin/$B/llama-server.exe $LMS > build/llama-cuda/k5353.out 2>&1
bash tools/run-server-probe.sh k5354-id-A2 $SMK --exe=build/llama-cuda/bin/s47/llama-server.exe $LMS > build/llama-cuda/k5354.out 2>&1
python tools/server_smoke_cmp.py k5351-id-A1 k5352-id-B1 k5353-id-B2 k5354-id-A2 > build/llama-cuda/k5350-identity.txt 2>&1
# 2. the five checks, canonical + fuse
SC="prefix,agentic,branch,alternate,sampling,equiv,xcheck,noise,stream"
HO="--fixture $F/agentic-A.json --fixture2 $F/agentic-B.json --scenarios $SC --reprocess-allow 2112"
HOS="--fixture $F/agentic-B.json --fixture2 $F/agentic-A.json --scenarios $SC --reprocess-allow 2112"
N37="--no-hybrid --no-override --arg=-ngl --arg=99 --arg=-ncmoe --arg=37 --arg=-fit --arg=off"
CF="--arg=--lm-prefill-canonical --arg=--lm-prefill-fuse"
X="--exe=build/llama-cuda/bin/$B/llama-server.exe --batch 0 --env LLAMA_LM_LOG=1 $N37 --arg=--lm-prefill $CF"
LMH="--exe=build/llama-cuda/bin/$B/llama-server.exe --ub 0 --batch 0 --no-hybrid --no-override --env LLAMA_HYBRID_FREE_WATCH_MS=20 --env LLAMA_LM_LOG=1 --arg=--hybrid-moe --arg=--lm-prefill --arg=-fitt --arg=16000 $CF"
bash tools/run-server-mt.sh k5361-cf-lmstock-mt-swap run "$HOS" -- $X > build/llama-cuda/k5361.out 2>&1
bash tools/run-server-mt.sh k5362-cf-lmstock-mt run "$HO" -- $X > build/llama-cuda/k5362.out 2>&1
bash tools/run-server-mt.sh k5363-cf-frozen-mt-swap run "$HOS" -- $LMH --arg=--hybrid-moe-frozen > build/llama-cuda/k5363.out 2>&1
bash tools/run-server-mt.sh k5364-cf-frozen-mt run "$HO" -- $LMH --arg=--hybrid-moe-frozen > build/llama-cuda/k5364.out 2>&1
bash tools/run-server-mt.sh k5365-cf-dyn-mt-swap run "$HOS" -- $LMH > build/llama-cuda/k5365.out 2>&1
bash tools/run-server-mt.sh k5366-cf-dyn-mt run "$HO" -- $LMH > build/llama-cuda/k5366.out 2>&1
echo done > build/llama-cuda/k5350-part1.done
# 3. replicated pairs, canonical + fuse
bash tools/run-server-hitfresh.sh k5371-hf-cf-lmstock "$PL" -- $X > build/llama-cuda/k5371.out 2>&1
bash tools/run-server-hitfresh.sh k5372-hf-cf-frozen "$PL" -- $LMH --arg=--hybrid-moe-frozen > build/llama-cuda/k5372.out 2>&1
bash tools/run-server-hitfresh.sh k5373-hf-cf-dyn "$PL" -- $LMH > build/llama-cuda/k5373.out 2>&1
echo done > build/llama-cuda/k5350-part2.done
# 4. default mode: the dynamic suites and the replicated pairs, s47 against s51 + fuse
D47="--exe=build/llama-cuda/bin/s47/llama-server.exe --ub 0 --batch 0 --no-hybrid --no-override --env LLAMA_HYBRID_FREE_WATCH_MS=20 --env LLAMA_LM_LOG=1 --arg=--hybrid-moe --arg=--lm-prefill --arg=-fitt --arg=16000"
D51="--exe=build/llama-cuda/bin/$B/llama-server.exe --ub 0 --batch 0 --no-hybrid --no-override --env LLAMA_HYBRID_FREE_WATCH_MS=20 --env LLAMA_LM_LOG=1 --arg=--hybrid-moe --arg=--lm-prefill --arg=-fitt --arg=16000"
HOD="--fixture $F/agentic-A.json --fixture2 $F/agentic-B.json --scenarios $SC"
HOSD="--fixture $F/agentic-B.json --fixture2 $F/agentic-A.json --scenarios $SC"
bash tools/run-server-mt.sh k5381-def47-dyn-mt run "$HOD" -- $D47 > build/llama-cuda/k5381.out 2>&1
bash tools/run-server-mt.sh k5382-def47-dyn-mt-swap run "$HOSD" -- $D47 > build/llama-cuda/k5382.out 2>&1
bash tools/run-server-mt.sh k5383-def51fuse-dyn-mt run "$HOD" -- $D51 --arg=--lm-prefill-fuse > build/llama-cuda/k5383.out 2>&1
bash tools/run-server-mt.sh k5384-def51fuse-dyn-mt-swap run "$HOSD" -- $D51 --arg=--lm-prefill-fuse > build/llama-cuda/k5384.out 2>&1
bash tools/run-server-hitfresh.sh k5385-hf-def47-dyn "$PL" -- $D47 > build/llama-cuda/k5385.out 2>&1
bash tools/run-server-hitfresh.sh k5386-hf-def51fuse-dyn "$PL" -- $D51 --arg=--lm-prefill-fuse > build/llama-cuda/k5386.out 2>&1
echo done > build/llama-cuda/k5350-chain.done
