"""Split docs/plans/generic-prefill-layer-residency-plan.md (2026-09-20).

Follows the currentstate split convention (AGENT_ENVIRONMENT.md, "Where current
state lives"): task blocks move VERBATIM, the split is recorded mechanically in
archive-manifest.json, and a from-disk reconstruction check proves byte-exact
reassembly.

Representation notes (both bit the first implementation, measured):
- Work in LF (the git blob form; core.autocrlf makes the worktree CRLF). The
  manifest records BOTH hashes; verify compares the blob hash.
- A block is tiled by STRING POSITION (regex ^### to the next ^###), so every
  chunk carries its own trailing blank separator line. Files are built by plain
  concatenation with NO re-joining and NO file terminator: joining extracted
  line-lists collapses a trailing blank line (join(['x','']) == 'x\\n'), which
  silently dropped exactly one byte at the seam after the last live block.
Run:
    python tools/split-gplr-plan.py            (dry-run prints the plan)
    python tools/split-gplr-plan.py --apply    (writes the files)
    python tools/split-gplr-plan.py --verify   (re-proves byte-exact reassembly)
"""
import hashlib
import io
import json
import re
import sys

LIVE = "docs/plans/generic-prefill-layer-residency-plan.md"
ARCHIVE = "docs/plans/generic-prefill-layer-residency-archive.md"
MANIFEST = "docs/plans/archive-manifest.json"
ARCHIVE_HEADER_LINES = 6  # title + blank + 4, stripped on reconstruction
ORIGINAL_WORKTREE_SHA256 = "f76c319ac3edbfc64116d2885512a8c0002b96df3695434f35b62769cac54966"


def load(p):
    with io.open(p, "r", encoding="utf-8", newline="") as f:
        return f.read()


def save(p, s):
    with io.open(p, "w", encoding="utf-8", newline="") as f:
        f.write(s)


def sha256(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def tile(text):
    """Return ([(start, end_exclusive), ...], title_of) for each '### ' block.
    Blocks are cut at the next heading's position, so each chunk ends with the
    blank separator line that preceded the next heading in the original."""
    starts = [m.start() for m in re.finditer(r"(?m)^### ", text)]
    blocks = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        blocks.append((s, e))
    return blocks


def is_live_block(text, a, b):
    head = text[a:min(a + 700, b)]
    return ("Status: in-progress" in head) or ("Status: draft" in head)


def split_into_chunks(body, markers):
    """Cut `body` into len(markers) chunks; chunk k starts at markers[k], in order."""
    pos = body.find(markers[0])
    assert pos == 0, "first marker not at position 0"
    starts = []
    for m in markers:
        idx = body.find(m, pos)
        assert idx >= 0, "marker out of order: " + m[:60]
        starts.append(idx)
        pos = idx
    return [body[starts[k]:(starts[k + 1] if k + 1 < len(starts) else len(body))]
            for k in range(len(starts))]


def apply_split():
    t_crlf = load(LIVE)
    if not t_crlf.startswith("### "):
        print("ABORT: expected the file to start with a '### ' task heading")
        sys.exit(1)
    t = t_crlf.replace("\r\n", "\n")  # work in LF (git blob form)

    blob_sha = sha256(t)
    if sha256(t_crlf) != ORIGINAL_WORKTREE_SHA256:
        print("ABORT: worktree sha256 does not match the recorded pre-split original")
        sys.exit(1)
    if t_crlf != t.replace("\n", "\r\n"):
        print("ABORT: worktree is not a pure LF->CRLF conversion of the blob form")
        sys.exit(1)

    blocks = tile(t)
    print("blocks:", len(blocks))
    for a, b in blocks:
        line = t.count("\n", 0, a) + 1
        print(f"  line {line:>4}.. {t[a:a+95].strip()}")

    live_idx = [i for i, (a, b) in enumerate(blocks) if is_live_block(t, a, b)]
    print("live:", [t[blocks[i][0]:blocks[i][0] + 70].strip() for i in live_idx])
    if len(live_idx) != 2:
        print(f"ABORT: expected exactly 2 live blocks (T-GPLR-37 in-progress, T-GPLR-39 draft), found {len(live_idx)}")
        sys.exit(1)

    archive_chunks, archive_map, live_map = [], [], []
    for i, (a, b) in enumerate(blocks):
        rec = {
            "title": t[a:t.find("\n", a)].rstrip("\r"),
            "orig_line_start": t.count("\n", 0, a) + 1,
            "orig_line_end": t.count("\n", 0, b),
        }
        if i in live_idx:
            live_map.append(rec)
        else:
            archive_chunks.append(t[a:b])
            archive_map.append(rec)

    hdr = (
        "# Generic prefill layer residency — CLOSED/COMPLETED task archive\n"
        "\n"
        "Verbatim archive shard split from docs/plans/generic-prefill-layer-residency-plan.md\n"
        "on 2026-09-20. This header is 6 lines (title + blank + 4); strip it when\n"
        "reconstructing or diffing against the original bytes.\n"
        "\n"
    )
    arch_text = hdr + "".join(archive_chunks)
    save(ARCHIVE, arch_text)

    live_header = (
        "# Generic prefill layer residency — executable backlog (live tasks only)\n"
        "\n"
        "SPLIT 2026-09-20: the 18 closed/completed task blocks moved VERBATIM to\n"
        "`docs/plans/generic-prefill-layer-residency-archive.md` (mechanical record:\n"
        "docs/plans/archive-manifest.json, including the byte-exact reconstruction check).\n"
        "This file now holds ONLY live work (in-progress or draft): search, read and edit it\n"
        "as before. Index of archived blocks (original order preserved in the archive):\n"
    )
    for rec in archive_map:
        live_header += f"- {rec['title']}\n"
    live_header += (
        "\n"
        "Rules (same convention as the currentstate split, AGENT_ENVIRONMENT.md):\n"
        "- Closed tasks are archived, never edited in place; a new closure appends its RESULT\n"
        "  in the live file first, and the block moves to the archive on the NEXT split.\n"
        "- The archive's 6-line header is not original bytes; strip it when reconstructing.\n"
        "- Re-prove the split any time: python tools/split-gplr-plan.py --verify\n"
    )
    live_text = live_header + "".join(t[blocks[i][0]:blocks[i][1]] for i in live_idx)
    # Drop ONE file-final blank line (block 39's verbatim separator before what
    # used to be block 31): content-identical, but git diff --check would flag
    # "new blank line at EOF" forever. verify restores it when rebuilding.
    live_terminator_stripped = live_text.endswith("\n\n")
    if live_terminator_stripped:
        live_text = live_text[:-1]
    save(LIVE, live_text)

    man = {
        "split_date": "2026-09-20",
        "source": LIVE,
        "original_worktree_sha256": ORIGINAL_WORKTREE_SHA256,
        "original_worktree_bytes": len(t_crlf.encode("utf-8")),
        "blob_sha256_lf": blob_sha,
        "blob_bytes_lf": len(t.encode("utf-8")),
        "line_ending_note": "worktree CRLF == git blob LF with \\n -> \\r\\n (core.autocrlf); split done in LF, verify compares the blob hash",
        "tiling_note": "blocks cut at the next '^### ' position, so each chunk carries its own trailing blank separator; files built by plain concatenation",
        "archive": ARCHIVE,
        "archive_header_lines": ARCHIVE_HEADER_LINES,
        "live_file": LIVE,
        "blocks_total": len(blocks),
        "blocks_archived": len(archive_map),
        "blocks_live": len(live_map),
        "live_blocks": live_map,
        "archived_blocks": archive_map,
        "live_file_terminator_stripped": live_terminator_stripped,
        "verify_command": "python tools/split-gplr-plan.py --verify",
    }
    save(MANIFEST, json.dumps(man, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {ARCHIVE} ({len(arch_text.encode('utf-8'))} B)")
    print(f"     {LIVE} ({len(live_text.encode('utf-8'))} B)")
    print("original worktree sha256", ORIGINAL_WORKTREE_SHA256)
    print("blob (LF) sha256        ", blob_sha)


def verify_split():
    man = json.loads(load(MANIFEST))
    arch_now = load(ARCHIVE).replace("\r\n", "\n")
    live_now = load(LIVE).replace("\r\n", "\n")

    arch_recs = man["archived_blocks"]
    live_recs = man["live_blocks"]

    arch_lines = arch_now.split("\n")
    assert arch_lines[0].startswith("# "), "archive header missing"
    arch_body = "\n".join(arch_lines[man["archive_header_lines"]:])
    arch_chunks = split_into_chunks(arch_body, [r["title"] for r in arch_recs])

    live_lines = live_now.split("\n")
    first_head = next(i for i, ln in enumerate(live_lines) if ln.startswith("### "))
    live_body = "\n".join(live_lines[first_head:])
    live_chunks = split_into_chunks(live_body, [r["title"] for r in live_recs])

    all_recs = ([(r["orig_line_start"], "a", i) for i, r in enumerate(arch_recs)]
                + [(r["orig_line_start"], "l", i) for i, r in enumerate(live_recs)])
    all_recs.sort()
    final_live_i = len(live_recs) - 1  # the file-final live chunk
    parts = []
    for _, side, i in all_recs:
        parts.append(arch_chunks[i] if side == "a" else live_chunks[i])
        if side == "l" and i == final_live_i and man.get("live_file_terminator_stripped"):
            parts.append("\n")  # restore the stripped file-final blank line
    rebuilt = "".join(parts)

    got = sha256(rebuilt)
    want = man["blob_sha256_lf"]
    ok = got == want
    print(f"reconstructed (LF): {len(rebuilt.encode('utf-8'))} B  sha256 {got}")
    print(f"manifest blob (LF): {man['blob_bytes_lf']} B  sha256 {want}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main():
    if "--verify" in sys.argv:
        verify_split()
    elif "--apply" in sys.argv:
        apply_split()
    else:
        dry = load(LIVE).replace("\r\n", "\n")
        blocks = tile(dry)
        print("blocks:", len(blocks))
        for a, b in blocks:
            live = "LIVE " if is_live_block(dry, a, b) else "arch "
            print(f"  {live} line {dry.count(chr(10), 0, a)+1:>4}.. {dry[a:a+95].strip()}")
        print("\n(dry-run; pass --apply to write, --verify to re-prove)")


main()
