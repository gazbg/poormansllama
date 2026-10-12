# Session specs of the TG-beat-stock goal (G5-G7 and the final-build G3/G4 sessions)

Inputs for `python tools/run-tg-arms.py --spec <file>`; run each inside one GPU-mutex hold with the build under test named LAST on the wrapped command line, e.g.

    powershell -NoProfile -File tools/with-gpu-lock.ps1 -VerdictTag g12-g5 -Command "python tools\run-tg-arms.py --spec docs\plans\tg-beat-stock-specs\g12-g5.json --attest build\llama-cuda\bin\g12\llama-layer-streaming.exe"

The specs name `build/llama-cuda/bin/<tag>` side-copies (rebuild and side-copy first) and the model / prompt paths of the dev machine. `g11-queue.sh` is the order the final-build sessions ran in.
Meaning of each spec: see `docs/plans/tg-beat-stock-8gb-262k.md` (G4-G7 contracts and results). Note: `run-tg-arms.py` writes the verdict next to the session directory under `build/llama-cuda/`, which is not versioned.

Generators of the later sessions (run `python <generator>.py`; each writes `build/llama-cuda/specs/<session>.json` and a copy here): `mk_qa.py` (needle QA of the final recipe), `mk_deep.py` (deep-fill QA / speed arms), `mk_final.py` (keep-free comparison with stock), `mk_g43.py` .. `mk_g45.py` (GQA8 and prefill-recipe sessions), `mk_g46.py` (short-prompt sweep, stock against layer-major), `mk_g47.py` (the final configuration on a dedicated GPU: QA at six lengths and the headline against stock), `mk_g48.py` (diagnostics: plain batch-major at 200k / 250k, f16 KV, MTP re-measure), `mk_g49.py` (stock prefill with the op offload off and with ubatch 2048), `mk_g52.py` / `mk_g53.py` (slot-window sweeps of the layer-major prefill, cold and warm), `mk_g50.py` (prepared for the stage C build: the CPU leg of the layer-major prefill, `--pp-cpu-layers`; `mk_g51.py` for its QA to be written). Queues: `run-session-queue.sh <build> <session> ...`.
