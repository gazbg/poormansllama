#!/usr/bin/env bash
# DQ-6: the stock path through the old (git HEAD) and the new Pascal MMQ tile configuration, bit for bit
cd "$(dirname "$0")/../.." || exit 1; mkdir -p build/dense
CFG=third_party/llama.cpp/ggml/src/ggml-cuda/mmq-config-pascal.cuh
mkdir -p build/dense/lib-new build/dense/lib-old
cp -a build/llama-cuda/bin/libggml-cuda.so* build/dense/lib-new/
cp $CFG build/dense/mmq-config-pascal.new.cuh
git show HEAD:third_party/llama.cpp/ggml/src/ggml-cuda/mmq-config-pascal.cuh > $CFG
cmake --build build/llama-cuda --target ggml-cuda --parallel 8 > build/dense/dq6-build-old.log 2>&1
cp -a build/llama-cuda/bin/libggml-cuda.so* build/dense/lib-old/
cp build/dense/mmq-config-pascal.new.cuh $CFG
cmake --build build/llama-cuda --target ggml-cuda --parallel 8 > build/dense/dq6-build-new.log 2>&1
cmp build/llama-cuda/bin/libggml-cuda.so.0.20.2 build/dense/lib-new/libggml-cuda.so.0.20.2 && echo "new library rebuilt identical to the saved one" || echo "NOTE: rebuilt library differs byte-wise from the saved one (not reproducible builds)"
ls -la build/dense/lib-old build/dense/lib-new | head -12
FL="-c 6144 -ctk q8_0 -ctv q8_0 -ub 512 --ctx-checkpoints 0 --lm-prefill-host-layers 38"
echo "== NEW tile configuration, ordinary path"
NPRED=16 tools/dense/collect.sh mmqnew "$FL" 
echo "== OLD tile configuration (git HEAD), ordinary path"
LD_LIBRARY_PATH=$PWD/build/dense/lib-old NPRED=16 tools/dense/collect.sh mmqold "$FL"
python3 -I - <<'PY'
import json
a=json.load(open('build/dense/eq-mmqnew.json')); b=json.load(open('build/dense/eq-mmqold.json'))
ok=0
for x,y in zip(a,b):
    ids_x=[t['id'] for t in x['tokens']]; ids_y=[t['id'] for t in y['tokens']]
    lp_x=[t['lp'] for t in x['tokens']]; lp_y=[t['lp'] for t in y['tokens']]
    same = ids_x==ids_y and lp_x==lp_y
    ok += same
    print(f"prompt {x['i']} ({x['prompt_n']} tok, pp {x['pp_tps']:.0f} vs {y['pp_tps']:.0f} t/s): ids {'identical' if ids_x==ids_y else 'DIFFER'}, logprobs {'bit-identical' if lp_x==lp_y else 'differ'}")
print(f"{ok}/{len(a)} prompts bit-identical (new tile vs git HEAD tile)")
PY
echo DQ6 DONE
