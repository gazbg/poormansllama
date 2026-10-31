#!/usr/bin/env bash
# Put the tree of a scratch-repo branch into this worktree's index and working tree as third_party/llama.cpp.
#   usage: tools/build/sync-export-wp.sh <scratch repo> <branch>        e.g. tools/build/sync-export-wp.sh /c/Development/AI/llama-sync sync/w3
# Every step is checked; nothing destructive runs behind a pipeline (a failed fetch once emptied the directory).
set -euo pipefail
SCRATCH=$1
BRANCH=$2
cd "$(dirname "$0")/../.."
NAME=$(echo "$BRANCH" | tr '/' '-')
T=$(git -C "$SCRATCH" rev-parse "$BRANCH^{tree}")
# a parentless commit of the tree: the blobless scratch clone cannot serve its own history to another repository
R=$(echo "export of $BRANCH $(git -C "$SCRATCH" rev-parse "$BRANCH")" | git -C "$SCRATCH" commit-tree "$T")
git -C "$SCRATCH" update-ref "refs/export/$NAME" "$R"
git fetch -q "$SCRATCH" "refs/export/$NAME:refs/sync/$NAME"
test "$(git rev-parse "refs/sync/$NAME^{tree}")" = "$T" || { echo "fetched tree differs"; exit 1; }
git rm -r -q -f third_party/llama.cpp
git read-tree --prefix=third_party/llama.cpp/ -u "refs/sync/$NAME"
GOT=$(git write-tree --prefix=third_party/llama.cpp/)
test "$GOT" = "$T" || { echo "index subtree $GOT != $T"; exit 1; }
echo "third_party/llama.cpp == $BRANCH ($T), $(git ls-files third_party/llama.cpp | wc -l) files"
