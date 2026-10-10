#!/usr/bin/env python3
"""Recompute the derived arithmetic asserted in the owner answer.

Reads the 50k arm phase_summary lines straight from the traces so the
per-layer, overlap and per-ubatch figures are recomputed, not recalled.
Exit status is the check; nothing here is filtered through a pipe.
"""
import json
import os
import sys

LOG_DIR = os.path.join("build", "llama-cuda", "logs", "tg81-pad50k")
ARMS = ["i1-pad256-50k", "i2-nopad-50k", "i3-nopad-50k", "i4-pad256-50k"]
LAYERS = 40


def phase_summaries(path):
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if '"phase_summary"' not in line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main():
    if not os.path.isdir(LOG_DIR):
        print("MISSING arm directory: %s" % LOG_DIR)
        return 1

    rows = []
    for arm in ARMS:
        path = os.path.join(LOG_DIR, arm + ".jsonl")
        if not os.path.isfile(path):
            print("MISSING arm trace: %s" % path)
            return 1
        phases = {p["phase"]: p for p in phase_summaries(path)}
        for want in ("pp", "tg"):
            if want not in phases:
                print("MISSING %s phase_summary in %s" % (want, arm))
                return 1

        pp = phases["pp"]
        tg = phases["tg"]

        summed_us = (pp["total_transfer_us"] + pp["total_wait_us"]
                     + pp["total_compute_us"])
        ubatches = pp["ubatches"]
        rows.append({
            "arm": arm,
            "pp_tps": pp["tokens_per_s"],
            # wall_us is microseconds; /1e6 gives seconds per ubatch.
            "pp_ubatch_s": pp["wall_us"] / 1e6 / ubatches,
            "pp_tok_per_ubatch": pp["tokens"] / float(ubatches),
            "pp_overlap": summed_us / float(pp["wall_us"]),
            "pp_summed_s": summed_us / 1e6,
            "pp_wall_s": pp["wall_us"] / 1e6,
            "pp_feed_gbs": pp["bytes_transferred"] / float(pp["total_transfer_us"]) / 1e3,
            "tg_tps": tg["tokens_per_s"],
            "tg_ms_per_layer": 1000.0 / tg["tokens_per_s"] / LAYERS,
        })

    print("%-18s %9s %10s %9s %9s %9s %9s %10s" % (
        "arm", "pp tok/s", "ubatch s", "tok/ubat", "overlap",
        "tg tok/s", "ms/layer", "feed GB/s"))
    for r in rows:
        print("%-18s %9.3f %10.3f %9.1f %9.3f %9.3f %9.3f %10.3f" % (
            r["arm"], r["pp_tps"], r["pp_ubatch_s"], r["pp_tok_per_ubatch"],
            r["pp_overlap"], r["tg_tps"], r["tg_ms_per_layer"],
            r["pp_feed_gbs"]))

    ov = [r["pp_overlap"] for r in rows]
    ub = [r["pp_ubatch_s"] for r in rows]
    print("")
    print("PP summed transfer+wait+compute          : %.1f .. %.1f s"
          % (min(r["pp_summed_s"] for r in rows),
             max(r["pp_summed_s"] for r in rows)))
    print("PP wall                                 : %.1f .. %.1f s"
          % (min(r["pp_wall_s"] for r in rows),
             max(r["pp_wall_s"] for r in rows)))
    print("PP overlap vs wall                      : %.2f .. %.2f x"
          % (min(ov), max(ov)))
    print("PP per-ubatch wall                      : %.3f .. %.3f s"
          % (min(ub), max(ub)))

    # Per-layer across every configuration QUOTED in the answer.
    # Pad arms only for the row-set: the no-pad means are contaminated by the
    # single rebuild token (i3 max 1189 ms), which pad-delta.py documents.
    quoted = {
        "stock TG 24.41 ms/tok": 24.41 / LAYERS,
        "handoff-partial 43.0 ms/tok": 43.0 / LAYERS,
    }
    for r in rows:
        # NB: "pad" in "i2-nopad-50k" is True, so match the variant segment.
        if r["arm"].split("-")[1].startswith("pad"):
            quoted["row-set %s" % r["arm"]] = r["tg_ms_per_layer"]
    vals = sorted(quoted.values())
    print("per-layer, configs quoted in answer      : %.3f .. %.3f ms"
          % (vals[0], vals[-1]))

    allvals = sorted(r["tg_ms_per_layer"] for r in rows)
    print("per-layer, all 4 arms incl. no-pad means : %.3f .. %.3f ms"
          % (allvals[0], allvals[-1]))

    # The "about a second" mapping, stock side.
    print("stock 512-token ubatch at 347.19 tok/s  : %.3f s"
          % (512.0 / 347.19))

    # Overlap-model floor cross-check. 83.7 us/layer is the EXPERT LEG only;
    # the document composes it with `other` = 14.25 ms/token (recon 13.4,
    # solved to reproduce the stock anchor). Printing the leg alone would
    # give 298 tok/s and read as a claim the document never made.
    leg_ms = 83.7e-3 * LAYERS
    other_ms = 14.25
    print("model expert leg 83.7us x 40             : %.4f ms (leg alone)"
          % leg_ms)
    print("model composed other %.2f + leg           : %.4f ms -> %.2f tok/s"
          % (other_ms, other_ms + leg_ms, 1000.0 / (other_ms + leg_ms)))
    return 0


if __name__ == "__main__":
    sys.exit(main())