"""Re-split docs/plans/generic-prefill-layer-residency-plan.md (2026-09-21).

The 2026-09-20 splitter (tools/split-gplr-plan.py) is one-shot: it proved the
first split against a recorded original hash, so --verify necessarily FAILs
once the live file legitimately grows with new closures (T-GPLR-50/51/52 land,
T-GPLR-39 was closed in place). This tool performs the NEXT split: it moves
every non-live block (Status neither in-progress nor draft) from the live file
to the archive VERBATIM, rebuilds the live header's archived-block index, and
records a reconstruction invariant of its own:

    python tools/resplit-gplr-plan.py            (dry-run prints the plan)
    python tools/resplit-gplr-plan.py --apply    (moves the blocks)
    python tools/resplit-gplr-plan.py --verify   (re-proves byte-exact state)

Verify invariant: (archive body minus its 6-line header) + (live body minus its
header) tiles, by the recorded per-block positions, back into the pre-resplit
live file byte-exactly (LF form). The 2026-09-20 manifest
(docs/plans/archive-manifest.json) is left untouched as the historical record
of the first split; this tool writes resplit-manifest-2026-09-21.json.

Representation rules carried over from the first split (measured there):
- Work in LF (git blob form); compare hashes in LF.
- Blocks tile at string positions (^### to next ^###) so each chunk carries its
  own trailing blank separator; files are built by plain concatenation.
"""
import hashlib
import io
import json
import re
import sys

LIVE = "docs/plans/generic-prefill-layer-residency-plan.md"
ARCHIVE = "docs/plans/generic-prefill-layer-residency-archive.md"
MANIFEST = "docs/plans/resplit-manifest-2026-09-21.json"
ARCHIVE_HEADER_LINES = 6  # unchanged from the 2026-09-20 split


def load(p):
    with io.open(p, "r", encoding="utf-8", newline="") as f:
        return f.read()


def save(p, s):
    with io.open(p, "w", encoding="utf-8", newline="") as f:
        f.write(s)


def sha256(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def tile(text):
    starts = [m.start() for m in re.finditer(r"(?m)^### ", text)]
    blocks = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        blocks.append((s, e))
    return blocks


def is_live_block(text, a, b):
    head = text[a:min(a + 700, b)]
    return ("Status: in-progress" in head) or ("Status: draft" in head)


def body_after_header(text, header_lines):
    return "\n".join(text.replace("\r\n", "\n").split("\n")[header_lines:])


def main_apply():
    live_crlf = load(LIVE)
    if live_crlf.startswith("### "):
        print("ABORT: live file has no header to rebuild (already bare blocks?)")
        sys.exit(1)
    live = live_crlf.replace("\r\n", "\n")
    arch = load(ARCHIVE).replace("\r\n", "\n")

    pre_sha = sha256(live)
    print("pre-resplit live (LF) sha256:", pre_sha)

    blocks = tile(live)
    print("blocks:", len(blocks))
    recs = []
    for a, b in blocks:
        title = live[a:live.find("\n", a)].rstrip("\r")
        recs.append({
            "title": title,
            "orig_line_start": live.count("\n", 0, a) + 1,
            "orig_line_end": live.count("\n", 0, b),
            "live": is_live_block(live, a, b),
        })
    for r in recs:
        print("  %s %s" % ("LIVE " if r["live"] else "ARCH", r["title"][:80]))

    moving = [r for r in recs if not r["live"]]
    if not moving:
        print("nothing to move; aborting without changes")
        sys.exit(1)

    # archive gets the moving chunks appended verbatim, in live-file order
    moving_chunks = []
    for (a, b), r in zip(blocks, recs):
        if not r["live"]:
            moving_chunks.append(live[a:b])
            r["chunk"] = live[a:b]
    new_arch_body = body_after_header(arch, ARCHIVE_HEADER_LINES) + "".join(moving_chunks)
    new_arch = arch.split("\n")[:ARCHIVE_HEADER_LINES]
    new_arch = "\n".join(new_arch) + "\n" + new_arch_body
    if not new_arch.endswith("\n"):
        new_arch += "\n"

    # live header rebuilt: same preamble updated, index = ALL archived titles
    idx = [ln for ln in live.split("\n")[:40] if ln.startswith("- ### ")]
    moved_titles = [r["title"] for r in moving]
    new_idx = idx + moved_titles
    header = (
        "# Generic prefill layer residency — executable backlog (live tasks only)\n"
        "\n"
        "SPLIT 2026-09-20: the 18 closed/completed task blocks moved VERBATIM to\n"
        "`docs/plans/generic-prefill-layer-residency-archive.md` (mechanical record:\n"
        "docs/plans/archive-manifest.json, including the byte-exact reconstruction check).\n"
        "RESPLIT 2026-09-21: the blocks closed since then (T-GPLR-39, T-GPLR-50,\n"
        "T-GPLR-51) moved to the archive the same way; mechanical record:\n"
        "docs/plans/resplit-manifest-2026-09-21.json\n"
        "(python tools/resplit-gplr-plan.py --verify re-proves the CURRENT state;\n"
        "the 2026-09-20 tool's --verify is historical and no longer applies).\n"
        "This file holds ONLY live work (in-progress or draft). Index of archived\n"
        "blocks (original order preserved in the archive):\n"
    )
    for t in new_idx:
        header += t + "\n"
    header += (
        "\n"
        "Rules (same convention as the currentstate split, AGENT_ENVIRONMENT.md):\n"
        "- Closed tasks are archived, never edited in place; a new closure appends its RESULT\n"
        "  in the live file first, and the block moves to the archive on the NEXT split\n"
        "  (python tools/resplit-gplr-plan.py --apply, which also refreshes the index).\n"
        "- The archive's 6-line header is not original bytes; strip it when reconstructing.\n"
    )
    live_body = "".join(live[a:b] for (a, b), r in zip(blocks, recs) if r["live"])
    new_live = header + live_body
    if not new_live.endswith("\n"):
        new_live += "\n"

    man = {
        "resplit_date": "2026-09-21",
        "tool": "tools/resplit-gplr-plan.py",
        "pre_resplit_live_sha256_lf": pre_sha,
        "pre_resplit_live_bytes": len(live.encode("utf-8")),
        "archive_header_lines": ARCHIVE_HEADER_LINES,
        "live_header_rule": "live body = everything from the first '### ' heading",
        "blocks": [
            {
                "title": r["title"],
                "orig_line_start": r["orig_line_start"],
                "orig_line_end": r["orig_line_end"],
                "dest": "live" if r["live"] else "archive",
            }
            for r in recs
        ],
        "moved_titles": moved_titles,
        "verify_command": "python tools/resplit-gplr-plan.py --verify",
    }
    print("\napplying: %d blocks move, %d stay live" % (len(moving), len(recs) - len(moving)))
    save(ARCHIVE, new_arch.replace("\n", "\r\n"))
    save(LIVE, new_live.replace("\n", "\r\n"))
    save(MANIFEST, json.dumps(man, indent=2, sort_keys=True) + "\n")
    print("wrote", ARCHIVE, "(", len(new_arch.encode('utf-8')), "B )")
    print("wrote", LIVE, "(", len(new_live.encode('utf-8')), "B )")
    print("wrote", MANIFEST)


def main_verify():
    man = json.loads(load(MANIFEST))
    live = load(LIVE).replace("\r\n", "\n")
    arch = load(ARCHIVE).replace("\r\n", "\n")

    arch_body = "\n".join(arch.split("\n")[man["archive_header_lines"]:])
    first_head = next(i for i, ln in enumerate(live.split("\n")) if ln.startswith("### "))
    live_body = "\n".join(live.split("\n")[first_head:])

    # pull each block's chunk out of its destination file by title position
    def chunks_by_title(body, titles):
        starts = [m.start() for m in re.finditer(r"(?m)^### ", body)]
        out = {}
        for i, s in enumerate(starts):
            e = starts[i + 1] if i + 1 < len(starts) else len(body)
            title = body[s:body.find("\n", s)].rstrip("\r")
            if title in titles:
                assert title not in out, "duplicate title: " + title
                out[title] = body[s:e]
        return out

    wanted = {b["title"]: b for b in man["blocks"]}
    got = chunks_by_title(arch_body, wanted)
    got.update(chunks_by_title(live_body, wanted))
    missing = [t for t in wanted if t not in got]
    assert not missing, "blocks missing from both files: %r" % missing

    # reassemble pre-resplit live by original line order
    ordered = sorted(wanted.values(), key=lambda b: b["orig_line_start"])
    rebuilt = "".join(got[b["title"]] for b in ordered)
    got_sha = sha256(rebuilt)
    ok = got_sha == man["pre_resplit_live_sha256_lf"]
    print("reconstructed (LF): %d B  sha256 %s" % (len(rebuilt.encode("utf-8")), got_sha))
    print("manifest pre-split : %d B  sha256 %s" % (
        man["pre_resplit_live_bytes"], man["pre_resplit_live_sha256_lf"]))
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if "--verify" in sys.argv:
        main_verify()
    elif "--apply" in sys.argv:
        main_apply()
    else:
        main_apply.__globals__["_dry"] = True
        # dry-run: run the classify/print part only
        live = load(LIVE).replace("\r\n", "\n")
        blocks = tile(live)
        print("blocks:", len(blocks))
        for a, b in blocks:
            title = live[a:live.find("\n", a)].rstrip("\r")
            print("  %s %s" % ("LIVE " if is_live_block(live, a, b) else "ARCH", title[:85]))
        print("\n(dry run; --apply to move ARCH blocks, --verify to re-prove)")
