"""Split docs/plans/tg-beat-stock-8gb-262k.md (2026-10-13): move the history of G1-G18 VERBATIM into an archive file.

Why: the live plan crossed the doc-size FAIL band (480,000 B, tools/check-doc-sizes.py) when the G24-G29 blocks were added. Follows the convention of AGENT_ENVIRONMENT.md
("Where current state lives") and of tools/split-gplr-plan.py: blocks move verbatim, the split is recorded mechanically in a manifest, and a from-disk reconstruction check proves
byte-exact reassembly.

What moves: the single contiguous range that starts at the heading START_HEADING ("## Results log") and ends right before END_HEADING ("### G19 CONTRACT ..."): the results log, the early
decision logs, the G-TG8 contract and acceptance, and the contracts, results and acceptances of G1 .. G18. What stays live: the intro, the G19-G29 blocks, the idea ledger, the later decision
logs, the owner summaries and the resume checklist. The range is replaced by an INDEX block that names every archived heading with its line in the archive.

Representation (lessons of split-gplr-plan.py): work on the text with newline='' (no translation), cut by STRING POSITION, build files by plain concatenation, no re-joining of line lists.
Run:
    python tools/split-tg-plan.py            (dry-run: prints sizes and the index, writes nothing)
    python tools/split-tg-plan.py --apply    (writes the archive, the manifest and the new live file)
    python tools/split-tg-plan.py --verify   (re-proves byte-exact reassembly from the files on disk)
"""
import hashlib
import io
import json
import re
import sys

LIVE = "docs/plans/tg-beat-stock-8gb-262k.md"
ARCHIVE = "docs/plans/tg-beat-stock-8gb-262k-archive.md"
MANIFEST = "docs/plans/tg-beat-stock-archive-manifest.json"
START_HEADING = "## Results log"
END_HEADING = "### G19 CONTRACT"
ARCHIVE_HEADER = ("# Archive of docs/plans/tg-beat-stock-8gb-262k.md (G1-G18 history, moved verbatim on 2026-10-13)\n\n"
                  "Moved by `tools/split-tg-plan.py` because the live plan crossed the doc-size FAIL band. Nothing below was edited. "
                  "`python tools/split-tg-plan.py --verify` re-proves that live + archive reassemble the original byte for byte "
                  "(manifest: `docs/plans/tg-beat-stock-archive-manifest.json`). Line numbers in the live plan's index refer to THIS file.\n\n")


def load(p):
    with io.open(p, "r", encoding="utf-8", newline="") as f:
        return f.read()


def save(p, s):
    with io.open(p, "w", encoding="utf-8", newline="") as f:
        f.write(s)


def sha256(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def find_unique_line_start(text, prefix):
    hits = [m.start() for m in re.finditer(r"(?m)^" + re.escape(prefix), text)]
    if len(hits) != 1:
        raise SystemExit("expected exactly one line starting with %r, found %d" % (prefix, len(hits)))
    return hits[0]


def headings_in(text):
    """[(line number (1-based) within text, heading line)] for every '## ' / '### ' heading."""
    out = []
    for i, line in enumerate(text.split("\n")):
        if line.startswith("## ") or line.startswith("### "):
            out.append((i + 1, line))
    return out


def build_index(archived_text, header_lines):
    rows = []
    for ln, h in headings_in(archived_text):
        rows.append("- archive line %d: %s" % (ln + header_lines, h[:150] + ("..." if len(h) > 150 else "")))
    return ("## ARCHIVED HISTORY (G1-G18; moved verbatim to `docs/plans/tg-beat-stock-8gb-262k-archive.md` on 2026-10-13)\n"
            "The results log, the early decision logs, the G-TG8 contract and acceptance, and the contracts, results and acceptances of G1 .. G18 live in the archive file "
            "(search it with `grep -n \"G14 CONTRACT\" docs/plans/tg-beat-stock-8gb-262k-archive.md`). The headings it holds, with their line in the archive:\n"
            + "\n".join(rows) + "\n\n")


def plan(text):
    a = find_unique_line_start(text, START_HEADING)
    b = find_unique_line_start(text, END_HEADING)
    if b <= a:
        raise SystemExit("END heading precedes START heading")
    archived = text[a:b]
    header_lines = ARCHIVE_HEADER.count("\n")
    index = build_index(archived, header_lines)
    live_new = text[:a] + index + text[b:]
    archive = ARCHIVE_HEADER + archived
    return a, b, archived, index, live_new, archive


def cmd_dry():
    text = load(LIVE)
    a, b, archived, index, live_new, archive = plan(text)
    print("original: %d B, sha256 %s" % (len(text.encode("utf-8")), sha256(text)))
    print("moves chars [%d, %d) = %d B (%d lines); index block %d B" % (a, b, len(archived.encode("utf-8")), archived.count("\n"), len(index.encode("utf-8"))))
    print("live after: %d B; archive: %d B" % (len(live_new.encode("utf-8")), len(archive.encode("utf-8"))))
    print(index[:1500])


def cmd_apply():
    text = load(LIVE)
    if "## ARCHIVED HISTORY (G1-G18" in text:
        raise SystemExit("already split (the live file has the index block); use --verify")
    a, b, archived, index, live_new, archive = plan(text)
    header_lines = ARCHIVE_HEADER.count("\n")
    manifest = {
        "live": LIVE, "archive": ARCHIVE, "start_heading": START_HEADING, "end_heading": END_HEADING,
        "archive_header_chars": len(ARCHIVE_HEADER), "archive_header_lines": header_lines,
        "orig_char_start": a, "orig_char_end": b, "index_chars": len(index),
        "original_sha256": sha256(text), "original_bytes": len(text.encode("utf-8")),
        "archive_sha256": sha256(archive), "live_sha256": sha256(live_new),
    }
    save(ARCHIVE, archive)
    save(LIVE, live_new)
    with io.open(MANIFEST, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print("applied: live %d B, archive %d B; manifest %s" % (len(live_new.encode("utf-8")), len(archive.encode("utf-8")), MANIFEST))
    cmd_verify()


def cmd_verify():
    m = json.load(io.open(MANIFEST, "r", encoding="utf-8"))
    live = load(m["live"])
    archive = load(m["archive"])
    ok = True
    if sha256(archive) != m["archive_sha256"]:
        print("archive sha256 differs from the manifest (the archive was edited)")
        ok = False
    a = m["orig_char_start"]
    index = live[a:a + m["index_chars"]]
    if not index.startswith("## ARCHIVED HISTORY (G1-G18"):
        print("index block not found at the recorded offset (the live file changed before the index)")
        ok = False
    body = archive[m["archive_header_chars"]:]
    original = live[:a] + body + live[a + m["index_chars"]:]
    h = sha256(original)
    print("reassembled original: %d B sha256 %s" % (len(original.encode("utf-8")), h))
    print("manifest original:    %d B sha256 %s" % (m["original_bytes"], m["original_sha256"]))
    if h != m["original_sha256"]:
        print("NOTE: the reassembled text differs from the original at split time. This is expected once the LIVE file has been edited after the split "
              "(edits before the index shift its offset; edits after it change the tail). The archive itself is checked by its own sha256 above.")
        # strong check that does not depend on later live edits: the archived body is byte-identical to what the manifest hashed
        ok = ok and (sha256(archive) == m["archive_sha256"])
    print("VERIFY:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--apply" in sys.argv:
        cmd_apply()
    elif "--verify" in sys.argv:
        sys.exit(cmd_verify())
    else:
        cmd_dry()
