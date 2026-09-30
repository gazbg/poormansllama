#!/usr/bin/env python3
"""T-GPLR-63 A fixtures, host-only, no GPU and no model.

The T-GPLR-12 handoff assertion ("run_start declared handoff_mode on/auto-on but
no handoff events followed") is the rule that keeps the PP->TG promise honest, and
until now NOTHING pinned it. Deleting both assertion sites would have failed no test:
the only two traces on disk that actually violate it (gplr43-diag.jsonl and
gplr49-4k-nopw-control.jsonl) are recorded artifacts of past GPU sessions, and neither
is in a CTest arm. This generator turns that un-pinned rule into a pinned one.

It derives the fixture from a RECORDED trace rather than hand-writing JSON, so the
fixture's field vocabulary is a real emitter's, not this script's imagination. The
recorded trace must itself be clean -- a prefill-only run that BOTH host validators
accept -- because then the ONLY difference between "accepted" and "rejected" is the
edit below, and the negative fixture therefore fails for exactly one reason. A
negative fixture that fails for two reasons pins nothing: delete the rule and it still
fails, on the other one.

    python tools/layer-streaming/gen-gplr63-fixtures.py \\
        --real <recorded-prefill-only-trace.jsonl> --out <handoff-declared-on-no-event.jsonl>

What it changes, in total:

  * run_start.handoff_mode  "auto-off" -> "auto-on"     <- THE LIE
  * run_start.tg_tokens     added, 0                     <- T-GPLR-63 B, and the
                                                            reason the lie is a lie

and it REMOVES every `handoff*` event, so the declared promise has nothing behind it.

WHY tg_tokens:0 is part of the lie and not decoration. The promise a run makes with
"auto-on" is about the TG phase, and the TG phase exists only when the schedule has one
(--tg-tokens > 0). Recording tg_tokens:0 in the same run_start is exactly the fact that
makes the declaration self-refuting: the trace now says, in one record, "I promised a
handoff" and "I had no TG phase to hand off to". Before T-GPLR-63 B the only schedule
witness in a trace was run_end.n_tokens_generated, which is 0 for a run that FAILED
before generating anything -- so it cannot tell a prefill-only run from a crashed one.
The committed fixtures from the --lie/--clean pair below are what the CTest arms
consume; regenerate them only from a trace that still passes both validators.
"""
import argparse
import json
import sys

DECLARED_ON = "auto-on"


def load_events(path):
    events = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
    return events


def main():
    ap = argparse.ArgumentParser(description="T-GPLR-63 handoff-assertion fixtures from a recorded trace")
    ap.add_argument("--real", required=True, help="recorded CLEAN prefill-only trace JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clean-out", default=None,
                    help="also write the un-edited control here (the --lie/--clean pair)")
    a = ap.parse_args()

    events = load_events(a.real)
    if not events:
        print(f"REFUSED: {a.real} has no JSONL events")
        return 2

    starts = [e for e in events if e.get("event") == "run_start"]
    ends = [e for e in events if e.get("event") == "run_end"]
    if len(starts) != 1 or len(ends) != 1:
        print(f"REFUSED: want exactly one run_start/run_end, got {len(starts)}/{len(ends)}")
        return 2
    if ends[0].get("exit_status") != 0 or ends[0].get("failed"):
        print("REFUSED: the source trace did not exit 0; a failed run reports "
              "run_end.exit_status != 0 FIRST, which would mask the handoff rule")
        return 2
    if starts[0].get("handoff_mode") == DECLARED_ON:
        print(f"REFUSED: {a.real} already declares {DECLARED_ON}")
        return 2
    if any(e.get("event", "").startswith("handoff") for e in events):
        print(f"REFUSED: {a.real} already emits handoff events")
        return 2
    n_handoff = len([e for e in events if e.get("event") == "phase_summary"
                     and e.get("phase") not in ("pp", None)])
    if n_handoff:
        print(f"REFUSED: {a.real} has {n_handoff} non-pp phase_summary event(s); "
              "the fixture must describe a run that reached no TG phase")
        return 2

    if a.clean_out:
        with open(a.clean_out, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        print(f"wrote {a.clean_out}: {len(events)} events, UNCHANGED control "
              f"(handoff_mode={starts[0].get('handoff_mode')!r})")

    # Rebuild the event order: the declaration is on the existing run_start line, and
    # the rest of the stream is byte-for-byte the recording. Inserting tg_tokens right
    # after handoff_mode keeps the two facts adjacent in the record.
    out = []
    for e in events:
        if e.get("event") == "run_start":
            e = dict(e)
            e["handoff_mode"] = DECLARED_ON
            e["tg_tokens"] = 0
        elif e.get("event", "").startswith("handoff"):
            continue
        out.append(e)

    with open(a.out, "w") as f:
        for e in out:
            f.write(json.dumps(e) + "\n")
    print(f"wrote {a.out}: {len(out)} events, handoff_mode={DECLARED_ON!r}, tg_tokens=0, "
          "0 handoff events (the lie, isolated to one field)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
