# Session specs of the TG-beat-stock goal (G5-G7 and the final-build G3/G4 sessions)

Inputs for `python tools/run-tg-arms.py --spec <file>`; run each inside one GPU-mutex hold with the build under test named LAST on the wrapped command line, e.g.

    powershell -NoProfile -File tools/with-gpu-lock.ps1 -VerdictTag g12-g5 -Command "python tools\run-tg-arms.py --spec docs\plans\tg-beat-stock-specs\g12-g5.json --attest build\llama-cuda\bin\g12\llama-layer-streaming.exe"

The specs name `build/llama-cuda/bin/<tag>` side-copies (rebuild and side-copy first) and the model / prompt paths of the dev machine. `g11-queue.sh` is the order the final-build sessions ran in.
Meaning of each spec: see `docs/plans/tg-beat-stock-8gb-262k.md` (G4-G7 contracts and results). Note: `run-tg-arms.py` writes the verdict next to the session directory under `build/llama-cuda/`, which is not versioned.
