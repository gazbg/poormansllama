# History: where the old work lives

This repository was built by consolidating several folders and branches from the
experiment period (August to October 2026). This file says where each part went.
The experiment outcomes themselves are in [LEDGER.md](LEDGER.md).

## 1. Sources and where they are now

| Source | What it was | Where it is now | Status |
|---|---|---|---|
| Main experiment repo (`qwen38GZI`, branch `research/layer-residency`, 233 commits) | Main line of the work | This repository, history preserved. Placeholder identity rewritten to GAZ. Machine name and username replaced by `<host>` and `<user>` in tracked history. 266 commits on `main` | The old folder was removed. The repo now lives at `C:\Development\AI\qwen38GZI` |
| Nested llama.cpp fork (`third_party/llama.cpp`, upstream base `36f0ae09`, fork tip `f0a247e69`, 26 fork commits) | The fork delta | `third_party/llama.cpp/` as plain files, tree `34c4eee8`. The 26 commits replayed with their messages. Patch copies in `archive/source-repos/llama-fork-patches/` | The old nested `.git` was removed. Mapping of old and new ids in [third_party/README.md](../../third_party/README.md) |
| `qwen38-aer-worktree` (about 50 GB) | Activation-ranker and hybrid-decode work, branch `freebuff/activation-trace-worktree` | Branch `archive/aer-activation-trace` (30 commits). Nested fork bundle `archive/source-repos/aer-llama-nested-all-refs.bundle`. Index of its 15 fork commits in [history/nested-aer.md](history/nested-aer.md). Text evidence in `archive/old-build-evidence/qwen38-aer-build-text/` | Folder removed. About 45 GB of run data and telemetry not kept |
| `qwen38-optv1-worktree` (about 0.7 GB) | Optimization v1 branch | Branch `archive/optimizaionV1` (2 commits). Nested commit as `archive/llama-optimizaionV1` (1 commit). Text evidence in `archive/old-build-evidence/optv1-build-text/` | Folder removed |
| `qwen38GZI-releases` (60 MB) | Prebuilt `s53` binaries and launcher | `releases/g42-s53/` (git-ignored) | Folder removed. Copy kept |
| `.freebuff/worktrees/*` in the main repo | Tool-managed worktrees, work in progress | Branches `archive/activation-trace-slice` (3), `archive/hybrid-mode-probe` (3), `archive/mtp-usage-predictor` (25), `archive/plan-idea` (1). Untracked files of the hybrid probe in `archive/wip-hybrid-mode-probe/`. Nested clones as bundles (5 and 1 fork commits, see [history/nested-freebuff-5e3cb7b5.md](history/nested-freebuff-5e3cb7b5.md) and [history/nested-freebuff-b78d8fc7.md](history/nested-freebuff-b78d8fc7.md)) | Folders removed. Their build logs were not backed up |
| `.claude/worktrees/*` in the main repo | Agent worktrees | Three worktrees with no unique commits. Their history is in the outer bundle | Folders removed. Their build logs were not backed up |
| Old build folder of the main repo | Run telemetry (`.jsonl`), logs, binaries | Text and telemetry copied to `archive/old-build-evidence/qwen38GZI-build/` (3.0 GB, 8,319 files). Binaries not kept (regenerable from the commits) | Folder removed |
| Old `out/` folder | Logs | `archive/old-build-evidence/qwen38GZI-out/` (79 files) | Kept |
| Root logs of the main repo | Logs and one stray object file | `archive/old-build-evidence/qwen38GZI-root-ignored/` | Kept |

## 2. Backups in `archive/`

`archive/` is git-ignored and is not published. Do not commit it.

| File | Contents | Verified | Contains the old strings |
|---|---|---|---|
| `source-repos/qwen38GZI-all-refs.bundle` (16 MB) | Deleted in this session, at your request. It held all refs of the main repo, the 23 agent checkpoint refs, and the pre-scrub history | Verified before deletion | Yes: 20 commits mentioned the machine name or username |
| `source-repos/aer-llama-nested-all-refs.bundle` (390 MB) | Nested fork of the AER work, all refs and tags | `git bundle verify` okay | No hits in its 15 fork commits. The rest is public upstream history |
| `source-repos/freebuff-5e3cb7b5-llama-nested-all-refs.bundle` (390 MB) | Nested fork, MTP predictor work | `git bundle verify` okay | No hits in its 5 fork commits |
| `source-repos/freebuff-b78d8fc7-llama-nested-all-refs.bundle` (390 MB) | Nested fork, activation trace work | `git bundle verify` okay | No hits in its 1 fork commit |
| `source-repos/llama-fork-patches/` (27 patches) | Fork delta as patches, with `a/third_party/llama.cpp/` prefixes | Applied in order with `git am` to the base | No hits in a text search |
| `source-repos/nested-refs.txt`, `qwen38GZI-refs.txt` | Ref lists of the old repos | Text | No |
| `source-repos/pre-identity-rewrite-commits.txt`, `pre-scrub-commits.txt` | Commit lists before the two history rewrites | Text | No hits |
| `old-build-evidence/` | Text and telemetry from the old build folders | Copied with `robocopy`, counts checked | Yes: the username appears in logs (170 files) |
| `wip-hybrid-mode-probe/` | Untracked files of one worktree | Copied | No hits in a text search |

Three bundles were deleted at your request in this session:
`pre-identity-rewrite.bundle`, `pre-scrub.bundle` and `qwen38GZI-all-refs.bundle`.
The first two held the history before the identity and scrub rewrites. The third
held the outer repo's full history, including the 23 agent checkpoint refs.

The old outer commit ids that the older records cite (for example `cb20819` and
`d676202`) no longer resolve anywhere on this machine. The text files
`pre-identity-rewrite-commits.txt` and `pre-scrub-commits.txt` still record each
old id with its subject, so the mapping can be read, but the commits cannot be
checked out.

### Restoring from a bundle

The nested fork bundles remain. Each restores a retired llama.cpp fork with its
upstream history:

```bash
git clone archive/source-repos/aer-llama-nested-all-refs.bundle restored-aer-fork
```

## 3. Corrections to earlier reports

- Earlier reports said about 850 llama.cpp commits lived only in the nested clones.
  That number counted commits reachable from about 7,100 upstream tags. Those
  commits are public ggml-org history. The fork commits on branches are 15 (AER),
  5 (MTP predictor) and 1 (activation trace). All are indexed above.

## 4. What is not in this repository

The consolidation covered the folders in the experiment chain above. Other folders
in `C:\Development\AI` were not consolidated and are not in this repository's
ledger. They include llama-related work such as `llamacppMAin`,
`llamacppPPTGOptim`, `llamaBonsai`, `llamacppCUDA`, `llamacppVulkan`,
`llamacppCache`, `llamacpp-master`, `moe-inference-engine`, the `Benchmarks` and
`evidence` folders, and unrelated tooling. Adding any of them to the ledger is a
separate step.

## 5. Where the rest of the history is

- Dated entries: [history/currentstate-entries.md](history/currentstate-entries.md),
  generated from `docs/currentstate.md` and its shards.
- Plans and specifications: [history/plans.md](history/plans.md).
- Commits on `main`: `git log --oneline`. Commits on the archive branches:
  [history/outer-archive-branches.md](history/outer-archive-branches.md).
- Commits of each retired nested fork: `history/nested-*.md`.
