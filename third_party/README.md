# third_party

## llama.cpp (vendored)

`third_party/llama.cpp/` is a plain copy of the llama.cpp fork used by this
project. It is not a submodule and has no nested `.git`, so the repo can be
cloned and built on its own.

| Item | Value |
|---|---|
| Upstream base | ggml-org/llama.cpp `36f0ae09028b524b30d37807de330d2e2f7409e9` (commit `6dae153` here) |
| Fork tip | `f0a247e6991dfc0d5257d576683666771eea0035`, branch `research/layer-residency` in the old nested checkout (commit `cfa2fc4` here) |
| Fork delta | 26 commits, replayed on top of the base with `git am` under `third_party/llama.cpp/` |
| Tree check | `git rev-parse HEAD:third_party/llama.cpp` equals `f0a247e69^{tree}` (`34c4eee80a925ad0b82db6750332347a5ad1bb9c`) |
| Extra fork commit | `optimizaionV1` commit `f732b73c7` is kept as `archive/llama-optimizaionV1` (commit `b1963ad`) |

Changes to the vendored tree belong in this repo's commits. Upstream pull
requests and pushes are not part of this layout.
