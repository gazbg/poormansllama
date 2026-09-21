"""Re-split docs/plans/generic-prefill-layer-residency-plan.md (2026-09-21).

The 2026-09-20 splitter (tools/split-gplr-plan.py) is one-shot: its --verify
hashes the original pre-split file, so it necessarily fails once the live
file grows with new closures. This tool performs the NEXT and future splits.

    python tools/resplit-gplr-plan.py            (dry-run: classify blocks)
    python tools/resplit-gplr-plan.py --apply    (move closed blocks to the archive)
    python tools/resplit-gplr-plan.py --verify   (re-prove the current state)

Verify invariant (final form; corrected twice by measurement):
- The ARCHIVE body is the frozen record: byte-exact vs archive_body_sha256_lf
  (recorded at apply time, LF form).
- LIVE blocks legitimately grow as work continues, so they get presence and
  no-shrink checks only; their apply-time spans are historical records.
- The apply-time EOF terminator strip (git diff --check convention) removes
  one newline from the ARCHIVE-FINAL chunk, not the orig-order-last chunk
  (measured); verify restores it before span checks.

Representation rules carried over from the first split (measured there):
- Work in LF (git blob form); compare hashes in LF.
- Blocks tile at string positions (^### to next ^###) so each chunk carries
  its own trailing blank separator; files are built by plain concatenation.
- Index lines in the live header MUST carry the "- " prefix: bare "### "
  lines in the header parse as phantom blocks (measured).
"""
import hashlib
import io
import json
import re
import sys

LIVE = "docs/plans/generic-prefill-layer-residency-plan.md"
ARCHIVE = "docs/plans/generic-prefill-layer-residency-archive.md"
MANIFEST = "docs/plans/resplit-manifest-2026-09-21.json"
ARCHIVE_HEADER_LINES = 6


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


def classify():
    live = load(LIVE).replace("\r\n", "\n")
    recs = []
    for a, b in tile(live):
        recs.append({
            "title": live[a:live.find("\n", a)].rstrip("\r"),
            "orig_line_start": live.count("\n", 0, a) + 1,
            "orig_line_end": live.count("\n", 0, b),
            "live": is_live_block(live, a, b),
        })
    return live, recs


def chunk_bounds(live, rec):
    """String bounds of a block: from its title line through its trailing
    blank separator (i.e. up to the next '\\n### ' at column 0), matching
    the tiling rule."""
    a = live.find(rec["title"])
    assert a >= 0, "title not found: " + rec["title"][:60]
    nxt = live.find("\n### ", a + 1)
    b = nxt + 1 if nxt >= 0 else len(live)
    return a, b


def main_apply():
    live_crlf = load(LIVE)
    if not live_crlf.lstrip().startswith("# "):
        print("ABORT: live file has no header to rebuild")
        sys.exit(1)
    live, recs = classify()
    pre_sha = sha256(live)
    print("pre-resplit live (LF) sha256:", pre_sha)
    for r in recs:
        print("  %s %s" % ("LIVE" if r["live"] else "ARCH", r["title"][:80]))

    moving = [r for r in recs if not r["live"]]
    if not moving:
        print("nothing to move; aborting without changes")
        sys.exit(1)

    moving_chunks = [live[slice(*chunk_bounds(live, r))] for r in moving]

    arch = load(ARCHIVE).replace("\r\n", "\n")
    arch_body_old = "\n".join(arch.split("\n")[ARCHIVE_HEADER_LINES:])
    new_arch_body = arch_body_old + "".join(moving_chunks)
    hdr = "\n".join(arch.split("\n")[:ARCHIVE_HEADER_LINES])
    new_arch = hdr + "\n" + new_arch_body

    # live header: keep preamble, refresh index with moved titles (prefixed)
    header = live[:live.find("### ")]
    rules_start = header.find("Rules (same convention")
    preamble, rules = header[:rules_start], header[rules_start:]
    if "RESPLIT 2026-09-21" not in preamble:
        preamble = preamble.replace(
            "This file holds ONLY live work",
            "RESPLIT 2026-09-21: blocks closed since the first split (T-GPLR-39,\n"
            "T-GPLR-50, T-GPLR-51) moved to the archive via tools/resplit-gplr-plan.py\n"
            "(manifest: docs/plans/resplit-manifest-2026-09-21.json; --verify re-proves\n"
            "the CURRENT state; the 2026-09-20 tool's --verify is historical).\n"
            "This file holds ONLY live work")
    new_header = preamble
    seen = set()
    for ln in [x for x in header.split("\n") if x.startswith("- ### ")] + \
              ["- " + r["title"] for r in moving]:
        if ln not in seen:
            seen.add(ln)
            new_header += ln + "\n"
    new_header += "\n" + rules

    live_parts = [live[slice(*chunk_bounds(live, r))] for r in recs if r["live"]]
    new_live = new_header + "".join(live_parts)
    terminator_stripped = False
    if new_live.endswith("\n\n"):
        new_live = new_live[:-1]
        terminator_stripped = True

    man = {
        "resplit_date": "2026-09-21",
        "tool": "tools/resplit-gplr-plan.py",
        "pre_resplit_live_sha256_lf": pre_sha,
        "archive_body_sha256_lf": sha256(new_arch_body),
        "archive_header_lines": ARCHIVE_HEADER_LINES,
        "blocks": [
            {"title": r["title"], "orig_line_start": r["orig_line_start"],
             "orig_line_end": r["orig_line_end"],
             "dest": "live" if r["live"] else "archive"}
            for r in recs
        ],
        "moved_titles": [r["title"] for r in moving],
        "verify_command": "python tools/resplit-gplr-plan.py --verify",
    }
    print("\napplying: %d blocks move, %d stay live" % (len(moving), len(recs) - len(moving)))
    save(ARCHIVE, new_arch.replace("\n", "\r\n"))
    save(LIVE, new_live.replace("\n", "\r\n"))
    save(MANIFEST, json.dumps(man, indent=2, sort_keys=True) + "\n")
    print("wrote", ARCHIVE, "(%d B)" % len(new_arch.encode("utf-8")))
    print("wrote", LIVE, "(%d B)" % len(new_live.encode("utf-8")))
    print("wrote", MANIFEST)


def main_verify():
    man = json.loads(load(MANIFEST))
    live = load(LIVE).replace("\r\n", "\n")
    arch = load(ARCHIVE).replace("\r\n", "\n")

    arch_body = "\n".join(arch.split("\n")[man["archive_header_lines"]:])
    first_head = next(i for i, ln in enumerate(live.split("\n")) if ln.startswith("### "))
    live_body = "\n".join(live.split("\n")[first_head:])

    def chunks_by_title(body, titles):
        out = {}
        for i, s in enumerate([m.start() for m in re.finditer(r"(?m)^### ", body)]):
            e = [m.start() for m in re.finditer(r"(?m)^### ", body)]
            e = e[e.index(s) + 1] if e.index(s) + 1 < len(e) else len(body)
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

    # Binding invariant: the archive body is the frozen record.
    expected_arch = man.get("archive_body_sha256_lf")
    if expected_arch:
        got_arch = sha256(arch_body)
        if got_arch != expected_arch:
            print("archive body drift: %s vs expected %s" % (got_arch, expected_arch))
            print("RESULT: FAIL")
            sys.exit(1)
        print("archive body: byte-exact vs apply-time record (%d B)" % len(arch_body.encode("utf-8")))
    else:
        print("(manifest predates the archive-body invariant; span checks only)")

    # apply-time EOF strip removes one newline from the ARCHIVE-FINAL chunk
    # (measured: not from the orig-order-last chunk); restore before spans
    if man.get("archive_terminator_stripped") or not arch.endswith("\n\n"):
        arch_titles = []
        for m in re.finditer(r"(?m)^### ", arch_body):
            s = m.start()
            t = arch_body[s:arch_body.find("\n", s)].rstrip("\r")
            if t in wanted:
                arch_titles.append(t)
        if arch_titles and got[arch_titles[-1]].endswith("\n"):
            got[arch_titles[-1]] += "\n"

    ordered = sorted(wanted.values(), key=lambda b: b["orig_line_start"])
    for b in ordered:
        chunk = got[b["title"]]
        span = b["orig_line_end"] - b["orig_line_start"] + 1
        lines = chunk.count("\n") + (0 if chunk.endswith("\n") else 1)
        if b["dest"] == "archive":
            assert lines == span, "ARCHIVED block span drift (frozen record): %s (%d vs %d)" % (
                b["title"][:50], lines, span)
        else:
            if lines < span:
                raise AssertionError("live block SHRANK below apply-time span: %s" % b["title"][:50])
            print("live block %s: %d lines (apply-time %d - live growth allowed)" % (
                b["title"][4:14], lines, span))
    print("RESULT: PASS")
    sys.exit(0)


if __name__ == "__main__":
    if "--verify" in sys.argv:
        main_verify()
    elif "--apply" in sys.argv:
        main_apply()
    else:
        live = load(LIVE).replace("\r\n", "\n")
        blocks = tile(live)
        print("blocks:", len(blocks))
        for a, b in blocks:
            title = live[a:live.find("\n", a)].rstrip("\r")
            print("  %s %s" % ("LIVE" if is_live_block(live, a, b) else "ARCH", title[:85]))
        print("\n(dry run; --apply to move ARCH blocks, --verify to re-prove)")
