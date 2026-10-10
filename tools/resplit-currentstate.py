#!/usr/bin/env python3
"""Re-split docs/currentstate.md into live and verbatim historical files.

The initial 2026-09-18 split is recorded in
 docs/currentstate/archive-manifest.json. This one-shot 2026-09-24 resplit
moves every dated entry before 2026-09-23 into per-day shards, keeps the
2026-09-23/24 entries and the non-entry risk tail live, and derives the archive
index from the shard files.

    python tools/resplit-currentstate.py             # dry run and size report
    python tools/resplit-currentstate.py --apply     # write the split atomically
    python tools/resplit-currentstate.py --verify    # prove the applied split
    python tools/resplit-currentstate.py --repair-from-git --git-ref HEAD
                                                    # rebuild only from the recorded source hash

Representation rules:
- Work in LF and compare hashes in LF, independent of core.autocrlf.
- Entry text is copied byte-for-byte; only trailing blank separators are
  normalized to one newline. The verifier uses the recorded body length and
  hash, so separator spacing cannot disguise content drift.
- The live archive index is derived from the shard files, never trusted from
  the previous live index.
- Normal apply refuses to overwrite an existing target or manifest. The repair
  path requires the committed source blob to match the failed manifest hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "docs" / "currentstate.md"
SHARD_DIR = ROOT / "docs" / "currentstate"
MANIFEST = SHARD_DIR / "resplit-manifest-2026-10-10.json"
RESPLIT_DATE = "2026-10-10"
# 2026-10-10 bump: the 2026-09-25 cutoff grew back to 497,896 B (LF; 504,027 B on
# disk), again over the band. Measured day sizes in that window: 2026-10-02 alone
# is 263,582 B over 38 entries, so the cut is now taken at 2026-10-03, keeping
# 2026-10-03..2026-10-10 live (117,329 B of entries).
# 2026-10-10 cleanup (owner-directed, BEFORE the cut): two stray entries had been
# appended BELOW the risk tail by the earlier damage/restore -- a date heading whose
# separator is an em dash ("Nested engine rebuild and host-suite provenance
# recheck"), which DATE_HEADING_RE cannot match, and a colon-form "T-GPLR-07 CUDA
# SEMANTIC-QA SMOKE TIER PASS (50/512)". The colon-form one IS parsed by
# parse_entries even though it sits in the verbatim tail, so it would have collided
# with the existing 2026-09-24.md shard; its live copy was dropped (byte-identical
# copies survive in that shard) and the em-dash block was moved VERBATIM into it.
# T-GPLR-77: the 2026-09-24 cutoff let the live file grow back over the FAIL band
# (480,985 B vs the 480,000 B cap) because every session since then has written
# to 2026-09-25..2026-10-02. The cutoff is a POLICY choice, not a derived value,
# so it is bumped deliberately here rather than silently by a dry run.
LIVE_CUTOFF = date(2026, 10, 3)
HEADER_LINES = 8
TAIL_MARKER = "## Known gaps / risks (current)"
INDEX_HEADING = "## Index of archived entries"
INDEX_RULES_HEADING = "## Index rules"
DATE_HEADING_RE = re.compile(r"(?m)^## (\d{4}-\d{2}-\d{2})(?:\s+\([^)]*\))?:")
ANY_HEADING_RE = re.compile(r"(?m)^## ")


def load(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def load_lf(path: Path) -> str:
    return load(path).replace("\r\n", "\n")


def save_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def byte_count(text: str) -> int:
    return len(text.encode("utf-8"))


def parse_entries(text: str) -> list[dict]:
    starts = [match.start() for match in DATE_HEADING_RE.finditer(text)]
    entries: list[dict] = []
    for pos, start in enumerate(starts):
        next_heading = ANY_HEADING_RE.search(text, start + 1)
        end = next_heading.start() if next_heading is not None else len(text)
        match = DATE_HEADING_RE.match(text, start)
        assert match is not None
        chunk = text[start:end]
        # Entry text is exact; normalize only trailing blank separators to one
        # newline so body hashes do not depend on the following heading's spacing.
        body = chunk.rstrip("\n") + "\n"
        entries.append(
            {
                "date": match.group(1),
                "title": text[start : text.find("\n", start)].rstrip("\r"),
                "body": body,
                "body_sha256_lf": sha256(body),
                "body_chars_lf": len(body),
                "orig_line_start": text.count("\n", 0, start) + 1,
                "orig_line_end": text.count("\n", 0, end),
            }
        )
    if not entries:
        raise ValueError("no dated entries found")
    return entries


def render_entries(entries: list[dict]) -> str:
    out: list[str] = []
    for entry in entries:
        body = entry["body"]
        out.append(body if body.endswith("\n") else body + "\n")
        out.append("\n")
    return "".join(out)


def shard_header(covers: str) -> str:
    return (
        f"# currentstate.md archive shard -- {covers}\n"
        "\n"
        f"Verbatim archive of the {RESPLIT_DATE} resplit of `docs/currentstate.md`.\n"
        "Entries keep their original text and order. The live current state and\n"
        "derived index remain in `docs/currentstate.md`. Split ranges, entry\n"
        "hashes and the apply-time source hash are recorded in\n"
        f"`docs/currentstate/{MANIFEST.name}`.\n"
    )


def existing_shards() -> list[Path]:
    return sorted(SHARD_DIR.glob("*.md"), key=lambda path: path.name)


def shard_entries(text: str) -> list[dict]:
    out: list[dict] = []
    for match in ANY_HEADING_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        title = text[match.start() : text.find("\n", match.start())].rstrip("\r")
        date_match = re.match(r"## (\d{4}-\d{2}-\d{2})", title)
        out.append({"line": line, "title": title, "date": date_match.group(1) if date_match else None})
    return out


def date_from_name(path: Path) -> str:
    match = re.match(r"(\d{4}-\d{2}-\d{2})", path.name)
    return match.group(1) if match else "historical"


def build_index(
    shards: list[Path], planned_texts: dict[str, str] | None = None
) -> tuple[list[str], str, str]:
    rows: list[tuple[str, str, str]] = []
    shard_dates: list[tuple[str, Path]] = []
    planned_texts = planned_texts or {}
    for path in shards:
        rel = path.relative_to(ROOT).as_posix()
        text = planned_texts[rel] if rel in planned_texts else load_lf(path)
        entries = shard_entries(text)
        if not entries:
            continue
        effective_dates = [entry["date"] for entry in entries if entry["date"]]
        shard_date = max(effective_dates) if effective_dates else date_from_name(path)
        shard_dates.append((shard_date, path))
        for entry in entries:
            entry_date = entry["date"] or date_from_name(path)
            title = entry["title"]
            if len(title) > 120:
                title = title[:117].rstrip() + "..."
            rows.append((entry_date, f"- {entry_date} -- `{rel}:{entry['line']}` -- {title}", shard_date))
    rows.sort(key=lambda row: (row[0], row[2]), reverse=True)
    shard_dates.sort(key=lambda row: row[0], reverse=True)
    dates = [row[0] for row in rows]
    return (
        [row[1] for row in rows],
        min(dates) if dates else "none",
        max(dates) if dates else "none",
    )


def build_live(
    source: str,
    live_entries: list[dict],
    all_shards: list[Path],
    planned_texts: dict[str, str] | None = None,
) -> str:
    index_rows, old_date, new_date = build_index(all_shards, planned_texts)
    live_dates = [entry["date"] for entry in live_entries]
    live_from = min(live_dates) if live_dates else "none"
    live_to = max(live_dates) if live_dates else "none"
    shard_names = [path.relative_to(ROOT).as_posix() for path in all_shards]
    header = f"""# Current State

## How this file is organised (resplit {RESPLIT_DATE})

This is the canonical current-state record. The initial split on 2026-09-18
moved old entries into searchable shards after `docs/currentstate.md` passed
lean-ctx's per-file search cap. The {RESPLIT_DATE} resplit applies the same
rule again: recent dated entries stay live, older dated entries move verbatim
into per-day shards, and this file keeps a derived index plus the non-entry
risk tail.

- **Recent entries** ({live_from} .. {live_to}) stay in this file, below.
- **Everything older** ({old_date} .. {new_date}) lives in the indexed shards:
  {', '.join(f'`{name}`' for name in shard_names)}.
- **Current state that is not an entry** -- "Known gaps / risks" and
  "Blocked / escalated" -- stays at the bottom of this file.
- New entries are added at the top of the "Recent entries" block. Existing
  entry text is never edited in a shard; an intentional correction must be
  explicit and must update the split manifest.
- The split is mechanically checked with
  `python tools/resplit-currentstate.py --verify`. The apply-time source hash,
  entry hashes, shard bodies and dispositions are recorded in
  `docs/currentstate/{MANIFEST.name}`. Prefer `rg` across
  `docs/currentstate/` when the date is unknown.

## Recent entries ({live_from} .. {live_to})

"""
    index = (
        f"{INDEX_HEADING} ({old_date} .. {new_date})\n\n"
        "Format: `<date> -- <shard>:<line> -- <entry>`, newest first. The rows\n"
        "are derived from the shard files; do not edit them by hand.\n\n"
        + "\n".join(index_rows)
        + "\n\n"
        + INDEX_RULES_HEADING
        + "\n\n"
        "Shard entry text is frozen. New entries go in the live Recent entries\n"
        "block; the next resplit moves old entries and regenerates this index.\n\n"
    )
    tail_pos = source.find(TAIL_MARKER)
    if tail_pos < 0:
        raise ValueError(f"missing tail marker: {TAIL_MARKER}")
    tail = source[tail_pos:]
    return header + render_entries(live_entries) + index + tail


def preflight(
    source_override: str | None = None,
) -> tuple[str, list[dict], list[dict], list[Path], dict[str, str], str]:
    source = source_override if source_override is not None else load_lf(LIVE)
    entries = parse_entries(source)
    live_entries = [entry for entry in entries if date.fromisoformat(entry["date"]) >= LIVE_CUTOFF]
    archived = [entry for entry in entries if date.fromisoformat(entry["date"]) < LIVE_CUTOFF]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in archived:
        grouped[entry["date"]].append(entry)
    shard_texts: dict[str, str] = {}
    new_paths: list[Path] = []
    for day in sorted(grouped):
        path = SHARD_DIR / f"{day}.md"
        new_paths.append(path)
        body = render_entries(grouped[day])
        if body.endswith("\n\n"):
            body = body[:-1]
        shard_texts[path.relative_to(ROOT).as_posix()] = shard_header(day) + "\n" + body
    all_shards = existing_shards() + sorted(new_paths, key=lambda item: item.name, reverse=True)
    # Deduplicate by path in case a historical file is unexpectedly in both lists.
    unique_shards = sorted(set(all_shards), key=lambda path: path.name)
    live_text = build_live(source, live_entries, unique_shards, shard_texts)
    return source, live_entries, archived, unique_shards, shard_texts, live_text


def main_dry_run() -> None:
    source, live_entries, archived, shards, shard_texts, live_text = preflight()
    print(f"source: {LIVE.relative_to(ROOT).as_posix()} ({byte_count(source)} B, LF)")
    print(f"  live dated entries: {len(live_entries)}")
    print(f"  archived dated entries: {len(archived)}")
    print(f"  projected live file: {byte_count(live_text)} B")
    for path in shards:
        rel = path.relative_to(ROOT).as_posix()
        if rel in shard_texts:
            print(f"  new shard {rel}: {byte_count(shard_texts[rel])} B")
    print("dry run only; no files written")


def main_apply(
    source_override: str | None = None, overwrite_resplit: bool = False
) -> None:
    if MANIFEST.exists() and not overwrite_resplit:
        raise SystemExit(f"ABORT: manifest already exists: {MANIFEST}")
    source, live_entries, archived, shards, shard_texts, live_text = preflight(source_override)
    new_rel = set(shard_texts)
    for path in shards:
        if (
            path.relative_to(ROOT).as_posix() in new_rel
            and path.exists()
            and not overwrite_resplit
        ):
            raise SystemExit(f"ABORT: target shard already exists: {path}")
    # T-GPLR-77 (DATA LOSS GUARD). --overwrite may supersede a manifest, but it
    # must NEVER let a run clobber a shard that a PREVIOUS run of this tool
    # generated. Measured 2026-10-02: applying twice in a row re-read the
    # already-shrunk live file, found one leftover entry from the cut day, and
    # rewrote docs/currentstate/2026-09-24.md from 17 entries (57,602 B) to 1
    # entry (4,443 B) -- destroying 16 entries of historical record, which
    # --verify still reported as PASS because the manifest described the
    # damaged state it had just written. The guard refuses to overwrite any
    # shard the previous manifest lists under new_shards. Recovering a lost
    # shard requires an explicit, separate restore -- not a re-apply.
    if overwrite_resplit and MANIFEST.exists():
        try:
            prior = json.loads(load(MANIFEST))
        except Exception:
            prior = {}
        prior_generated = {
            shard.get("file") for shard in prior.get("new_shards", []) if shard.get("file")
        }
        collisions = sorted(prior_generated & new_rel)
        if collisions:
            raise SystemExit(
                "ABORT: refusing to overwrite shard(s) a previous resplit generated: "
                + ", ".join(collisions)
                + " Re-applying from an already-split file would rewrite them from"
                + " the shrunken live block and LOSE entries (measured: 16 lost, while"
                + " --verify still passed). Restore from git or a backup, or raise"
                + " LIVE_CUTOFF so the cut is taken from a source that still holds them."
            )
    tail_pos = source.find(TAIL_MARKER)
    assert tail_pos >= 0
    old_header = source.find("## How this file is organised")
    assert old_header >= 0
    pre_sha = sha256(source)
    preexisting = [
        {
            "file": path.relative_to(ROOT).as_posix(),
            "bytes_lf": byte_count(load_lf(path)),
            "sha256_lf": sha256(load_lf(path)),
        }
        for path in shards
        if path.relative_to(ROOT).as_posix() not in new_rel
    ]
    entry_records: list[dict] = []
    new_shards: list[dict] = []
    for entry in archived:
        day = entry["date"]
        rel = f"docs/currentstate/{day}.md"
        full = shard_texts[rel]
        body = "\n".join(full.split("\n")[HEADER_LINES:])
        restored = body + "\n" if full.endswith("\n") and not body.endswith("\n\n") else body
        # The apply-time EOF strip removed one of two final newlines. Restore it
        # before deriving the frozen body hash.
        if not restored.endswith("\n\n"):
            restored += "\n"
        pos = restored.find(entry["body"])
        if pos < 0:
            raise RuntimeError(f"entry missing from planned shard: {entry['title'][:80]}")
        line = HEADER_LINES + restored.count("\n", 0, pos) + 1
        record = {
            "date": day,
            "title": entry["title"],
            "orig_line_start": entry["orig_line_start"],
            "orig_line_end": entry["orig_line_end"],
            "body_sha256_lf": entry["body_sha256_lf"],
            "body_chars_lf": entry["body_chars_lf"],
            "dest": "shard",
            "shard": rel,
            "shard_line": line,
        }
        entry_records.append(record)
    for rel, text in shard_texts.items():
        body = "\n".join(text.split("\n")[HEADER_LINES:])
        restored = body + "\n" if not body.endswith("\n\n") else body
        new_shards.append(
            {
                "file": rel,
                "covers": Path(rel).stem,
                "bytes_lf": byte_count(text),
                "body_sha256_lf": sha256(restored),
                "body_bytes_lf": byte_count(restored),
                "eof_newline_stripped": True,
                "entries": sum(1 for item in entry_records if item["shard"] == rel),
            }
        )
    for entry in live_entries:
        entry_records.append(
            {
                "date": entry["date"],
                "title": entry["title"],
                "orig_line_start": entry["orig_line_start"],
                "orig_line_end": entry["orig_line_end"],
                "body_sha256_lf": entry["body_sha256_lf"],
                "body_chars_lf": entry["body_chars_lf"],
                "dest": "live",
            }
        )
    entry_records.sort(key=lambda item: item["orig_line_start"])
    manifest = {
        "resplit_date": RESPLIT_DATE,
        "tool": "tools/resplit-currentstate.py",
        "source": LIVE.relative_to(ROOT).as_posix(),
        "source_bytes_lf": byte_count(source),
        "source_sha256_lf": pre_sha,
        "source_lines": source.count("\n"),
        "live_cutoff": LIVE_CUTOFF.isoformat(),
        "policy": "dated entry text is copied verbatim; only trailing blank separators are normalized; the generated organization/index block is replaced; the non-entry risk tail stays live",
        "replaced_generated_region": {
            "start_line": source.count("\n", 0, old_header) + 1,
            "end_line": source.count("\n", 0, tail_pos),
            "disposition": "replaced by the derived live organization, recent-entry and archive-index blocks",
        },
        "entries": entry_records,
        "new_shards": new_shards,
        "preexisting_shards": preexisting,
        "live": {
            # Derived, not hardcoded: the window moves every time LIVE_CUTOFF is
            # bumped, and a pinned literal made --verify fail on the next re-cut.
            "recent_heading": (
                "## Recent entries ("
                + min(entry["date"] for entry in live_entries)
                + " .. "
                + max(entry["date"] for entry in live_entries)
                + ")"
            ),
            "apply_time_entry_count": len(live_entries),
            "apply_time_bytes_lf": byte_count(live_text),
            "tail_marker": TAIL_MARKER,
        },
        "verification": {
            "command": "python tools/resplit-currentstate.py --verify",
            "checks": [
                "apply-time source hash recorded",
                "preexisting shard files byte-exact in LF",
                "new shard bodies and entry bodies byte-exact in LF",
                "apply-time live entry bodies present and ordered",
                "live archive index exactly derived from all shard files",
                "live organization, recent block and risk tail present",
            ],
        },
    }
    # In-memory proof before the first write.
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    for rel, text in shard_texts.items():
        if not text.startswith(shard_header(Path(rel).stem)):
            raise RuntimeError(f"bad planned header: {rel}")
    _live_dates = [entry["date"] for entry in live_entries]
    _expected_heading = (
        "## Recent entries ("
        + (min(_live_dates) if _live_dates else "none")
        + " .. "
        + (max(_live_dates) if _live_dates else "none")
        + ")"
    )
    if live_text.count(_expected_heading) != 1:
        raise RuntimeError(f"planned live recent heading missing: {_expected_heading}")
    if TAIL_MARKER not in live_text:
        raise RuntimeError("planned live tail missing")
    for rel, text in shard_texts.items():
        save_atomic(ROOT / rel, text)
    save_atomic(LIVE, live_text)
    save_atomic(MANIFEST, manifest_text)
    print(f"pre-resplit source LF sha256: {pre_sha}")
    for shard in new_shards:
        print(f"wrote {shard['file']} ({shard['bytes_lf']} B, {shard['entries']} entries)")
    print(f"wrote {LIVE.relative_to(ROOT).as_posix()} ({byte_count(live_text)} B, {len(live_entries)} live entries)")
    print(f"wrote {MANIFEST.relative_to(ROOT).as_posix()}")
    print("run --verify")


def shard_body(text: str) -> str:
    body = "\n".join(text.split("\n")[HEADER_LINES:])
    return body


def fail(message: str) -> None:
    print(f"RESULT: FAIL: {message}")
    raise SystemExit(1)


def main_verify() -> None:
    if not MANIFEST.exists():
        print(f"RESULT: CANNOT-EVALUATE: missing {MANIFEST.relative_to(ROOT).as_posix()}")
        raise SystemExit(2)
    manifest = json.loads(load(MANIFEST))
    live = load_lf(LIVE)
    for shard in manifest["preexisting_shards"]:
        path = ROOT / shard["file"]
        if not path.exists():
            fail(f"missing preexisting shard {shard['file']}")
        text = load_lf(path)
        if byte_count(text) != shard["bytes_lf"] or sha256(text) != shard["sha256_lf"]:
            fail(f"preexisting shard drift: {shard['file']}")
    shard_bodies: dict[str, str] = {}
    for shard in manifest["new_shards"]:
        path = ROOT / shard["file"]
        if not path.exists():
            fail(f"missing new shard {shard['file']}")
        text = load_lf(path)
        body = shard_body(text)
        if shard["eof_newline_stripped"] and not body.endswith("\n\n"):
            body += "\n"
        if byte_count(body) != shard["body_bytes_lf"] or sha256(body) != shard["body_sha256_lf"]:
            fail(f"new shard body drift: {shard['file']}")
        shard_bodies[shard["file"]] = body
    for entry in manifest["entries"]:
        body = shard_bodies[entry["shard"]] if entry["dest"] == "shard" else live
        matches = [match.start() for match in re.finditer(re.escape(entry["title"]), body)]
        if not matches:
            fail(f"entry title missing: {entry['title'][:100]}")
        expected_hash = entry["body_sha256_lf"]
        expected_chars = entry["body_chars_lf"]
        if not any(
            sha256(body[start : start + expected_chars]) == expected_hash
            for start in matches
        ):
            fail(f"entry body drift: {entry['title'][:100]}")
    live_dates = [entry["date"] for entry in manifest["entries"] if entry["dest"] == "live"]
    if not live_dates:
        fail("manifest has no live entries")
    if manifest["live"]["recent_heading"] not in live:
        fail("live recent heading missing")
    if TAIL_MARKER not in live:
        fail("live risk tail missing")
    pos = 0
    for entry in manifest["entries"]:
        if entry["dest"] != "live":
            continue
        found = live.find(entry["title"], pos)
        if found < 0:
            fail(f"live entry missing or out of order: {entry['title'][:100]}")
        pos = found + len(entry["title"])
    shards = existing_shards()
    manifest_shards = {item["file"] for item in manifest["preexisting_shards"]} | set(shard_bodies)
    actual_shards = {path.relative_to(ROOT).as_posix() for path in shards}
    if actual_shards != manifest_shards:
        fail(f"shard set drift: missing={sorted(manifest_shards - actual_shards)} extra={sorted(actual_shards - manifest_shards)}")
    expected_rows, old_date, new_date = build_index(shards)
    heading = f"{INDEX_HEADING} ({old_date} .. {new_date})"
    index_start = live.find(heading)
    if index_start < 0:
        fail("live archive index heading missing")
    rules_pos = live.find(INDEX_RULES_HEADING, index_start)
    if rules_pos < 0:
        fail("live index rules missing")
    index_block = live[index_start + len(heading) : rules_pos].strip().splitlines()
    index_rows = [line for line in index_block if line.startswith("- ")]
    if index_rows != expected_rows:
        for number, (got, want) in enumerate(zip(index_rows, expected_rows), 1):
            if got != want:
                fail(f"index row {number} drift: {got!r} != {want!r}")
        fail(f"index row count drift: {len(index_rows)} != {len(expected_rows)}")
    if "python tools/resplit-currentstate.py --verify" not in live:
        fail("live split rules missing")
    print(f"apply-time source: {manifest['source_bytes_lf']} B, sha256 {manifest['source_sha256_lf']}")
    print(f"preexisting shards: {len(manifest['preexisting_shards'])} byte-exact")
    print(f"new shards: {len(manifest['new_shards'])} body-exact")
    print(f"entries: {len(manifest['entries'])} present and hash-exact")
    print(f"live index: {len(index_rows)} rows derived from every shard")
    print(f"RESULT: PASS ({byte_count(live)} B live)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply", action="store_true", help="write shards, live file and manifest")
    group.add_argument(
        "--repair-from-git",
        action="store_true",
        help="rebuild this resplit from a committed pre-split source, then verify",
    )
    group.add_argument("--verify", action="store_true", help="verify the applied split")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "allow --apply to supersede an existing manifest and re-shard shards it "
            "would regenerate. Shards NOT regenerated by this run are recorded as "
            "preexisting and verified byte-exact, so this is a re-cut, not a merge."
        ),
    )
    parser.add_argument("--git-ref", default="HEAD", help="git ref used by --repair-from-git")
    args = parser.parse_args()
    if args.apply:
        main_apply(overwrite_resplit=args.overwrite)
    elif args.repair_from_git:
        if not MANIFEST.exists():
            raise SystemExit("ABORT: --repair-from-git requires the failed resplit manifest")
        failed = json.loads(load(MANIFEST))
        source = subprocess.check_output(
            ["git", "show", f"{args.git_ref}:{LIVE.relative_to(ROOT).as_posix()}"],
            cwd=ROOT,
        ).decode("utf-8").replace("\r\n", "\n")
        if sha256(source) != failed["source_sha256_lf"]:
            raise SystemExit(
                "ABORT: git source does not match the failed manifest's pre-resplit hash"
            )
        main_apply(source_override=source, overwrite_resplit=True)
        main_verify()
    elif args.verify:
        main_verify()
    else:
        main_dry_run()


if __name__ == "__main__":
    main()
