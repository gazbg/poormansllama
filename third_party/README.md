# third_party

## llama.cpp (vendored)

`third_party/llama.cpp/` is a plain copy of the llama.cpp fork used by this
project. It is not a submodule and has no nested `.git`, so the repo can be
cloned and built on its own.

| Item | Value |
|---|---|
| Upstream base | ggml-org/llama.cpp `04b569142da23d91beca090a99098d592d3f3c80` = tag `b10488` (2026-08-18), found 2026-10-30 by file-hash matching with `tools/find-upstream-base.py`: 3,403 of 3,488 vendored files of commit `6dae153` are byte-identical to it. Machine-readable copy: `third_party/UPSTREAM_BASE` |
| Synced to | **b12004** (`3c809f44383bcebc318b8cddd8c6d11aaa861e9b`, 2026-10-30): the upstream sync of 2026-10-30 is complete (waypoints b10729, b10991, b11311, b11675, b11850, b12004); the tree is that upstream tag plus the fork delta. The rows below describe the OLD state before the sync. Progress, decisions, every conflict resolution and the owner summary are in `docs/plans/upstream-sync-plan.md` |
| Fork snapshot | `36f0ae09028b524b30d37807de330d2e2f7409e9`, a commit of the old nested checkout (commit `6dae153` here). This row used to be called the upstream base; the id is NOT on ggml-org (a fetch by id is refused). It is upstream b10488 plus the August-September fork work, with no per-commit history |
| Fork tip | `f0a247e6991dfc0d5257d576683666771eea0035`, branch `research/layer-residency` in the old nested checkout (commit `cfa2fc4` here) |
| Fork delta | 26 commits, replayed on top of the fork snapshot with `git am` under `third_party/llama.cpp/`. Against the true upstream base the whole fork is 148 files, +58,337 / -314 (100 added, 48 modified; measured 2026-10-30 at HEAD `0a7a0f4`) |
| Tree check | `git rev-parse HEAD:third_party/llama.cpp` was `34c4eee80a925ad0b82db6750332347a5ad1bb9c` (= `f0a247e69^{tree}`) before the sync. After it the tree is `1cdf9871302ceba73058af8e10e6c791d342dd35`, byte-identical to the scratch-repo branch `sync/w6` (upstream b12004 + the fork delta; the intermediate trees are in the commit messages) |
| Extra fork commit | `optimizaionV1` commit `f732b73c7` is kept as `archive/llama-optimizaionV1` (commit `b1963ad`) |

Changes to the vendored tree belong in this repo's commits. Upstream pull
requests and pushes are not part of this layout.
