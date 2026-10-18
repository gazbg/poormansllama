#!/usr/bin/env python3
"""T-GPLR-54 A2: build tests/gplr54-tg-logits.jsonl, the validator fixture.

The fixture is a schema-valid streaming trace (both host validators must accept
it) built from the recorded gplr38-u512 arm's shape so its ids and n_tokens
are consistent with a real session: a run carrying
`tg_logit_margin: 8` in run_start and 8 `tg_logits` events for 96 tg ids.
The margin values are transcribed from the recorded T-GPLR-06 stderr margins
(build/llama-cuda/gplr50-ctl-4k-f16.log, steps 0..7 of the same anchor) so the
fixture's numbers are plausible rather than invented; the checker's value
sanity band (1e-4 .. 60.0) brackets them.
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE = os.path.join(HERE, '..', '..', '..', '..', 'build', 'llama-cuda', 'gplr38-u512.jsonl')
OUT = os.path.join(HERE, 'tests', 'gplr54-tg-logits.jsonl')

# steps 0..7 of build/llama-cuda/bare-default-k.log ("TG: step N margin M
# (gold G, runner-up R)") - the same anchor vehicle's real T-GPLR-06 stderr
# margins: 0.44, 16.41, 16.12, 0.97, 0.30, 19.28, 11.99, 6.54.
RECORDED_MARGINS = [
    (0.438701, 248046, 48), (16.412857, 198, 271), (16.118988, 248045, 248068),
    (0.970795, 248068, 74455), (0.297215, 271, 198), (19.276882, 248069, 510),
    (11.994511, 271, 198), (6.535427, 48, 45),
]


def load_ref_ids():
    with io.open(TRACE, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if '"tg_tokens"' not in line:
                continue
            e = json.loads(line)
            if e.get('event') == 'tg_tokens':
                return e['ids']
    raise SystemExit('reference trace has no tg_tokens: ' + TRACE)


def main():
    ids = load_ref_ids()
    n = len(RECORDED_MARGINS)
    if len(ids) < n:
        raise SystemExit('reference stream shorter than the fixture margin list')
    # The fixture's tg ids are the recorded top-1 sequence (a real session's
    # margins carry real gold ids) followed by the reference arm's tail, so the
    # checker's top1==greedy-id self-check holds by construction and the stream
    # length stays a real session's 96.
    ids = [m[1] for m in RECORDED_MARGINS] + ids[n:]

    lines = []
    # validators demand: one pp phase_summary, and NO handoff mode declared when
    # the fixture carries no handoff events (the recorded arms ran auto-on, but
    # the fixture is a minimal single-PP-pass shape, not a session replica).
    lines.append({'event': 'run_start', 'run_id': 42, 't0_us': 42000000,
                  'model': 'Qwen3.6-35B-A3B', 'arch': 'qwen35moe', 'n_layer': 40,
                  'n_slots': 3, 'device': 0, 'schedule': 'layer-major',
                  'copy_source': 'mmap-registered', 'kv_cache': 'f16',
                  'implicit_causal_mask': True, 'telemetry_chunk_stride': 10,
                  'act_slab_bytes': 33554432, 'act_slab_dtype': 'bf16',
                  'pp_tokens': 512, 'ctx_size': 1088, 'handoff_mode': 'auto-off',
                  'static_vram_bytes': 0, 'slot_vram_bytes': 1566584832,
                  'free_vram_bytes': 22226665472, 'total_vram_bytes': 25650855936,
                  'tg_logit_margin': n, 'tg_logit_margin_source': 'explicit'})
    for t, (margin, top1, top2) in enumerate(RECORDED_MARGINS):
        # reconstruct a plausible top-2 from the recorded margin and a nominal
        # top-1 value; only the DIFFERENCE is meaningful to consumers.
        top1_logit = 12.0
        lines.append({'event': 'tg_logits', 'run_id': 42, 'step': t,
                      'top1_id': top1, 'top1_logit': round(top1_logit, 6),
                      'top2_id': top2, 'top2_logit': round(top1_logit - margin, 6),
                      'margin': round(margin, 6), 't_us': 42100000 + 130000 * t})
    lines.append({'event': 'host_ledger', 'run_id': 42, 'embed_us': 248.2,
                  'output_us': 5.0, 'loop_us': -253.2, 't_us': 42200000})
    # phase_summary fields mirror the recorded gplr38-u512 arm's pp summary
    # (validate-telemetry checks graph_builds/graph_prep_us/bytes_transferred/
    # evictions/layers_run; the values here are scaled, plausible, consistent).
    lines.append({'event': 'phase_summary', 'run_id': 42, 'phase': 'pp',
                  'tokens': 512, 'ubatches': 1, 'wall_us': 610797.0,
                  'tokens_per_s': 838.699, 'bytes_transferred': 2290876416.0,
                  'evictions': 37, 'total_transfer_us': 651453.0,
                  'total_wait_us': 8478.0, 'total_compute_us': 277269.0,
                  'layers_run': 40, 'graph_prep_us': 14810, 'graph_builds': 0})
    lines.append({'event': 'tg_tokens', 'run_id': 42, 'n_tokens_generated': len(ids),
                  'ids': ids})
    lines.append({'event': 'phase_summary', 'run_id': 42, 'phase': 'tg',
                  'tokens': len(ids), 'ubatches': 0, 'wall_us': 4197965.0,
                  'tokens_per_s': 22.863, 'bytes_transferred': 748781056.0,
                  'evictions': 0, 'total_transfer_us': 4424006.0,
                  'total_wait_us': 67787.0, 'total_compute_us': 1853447.0,
                  'layers_run': 40, 'graph_prep_us': 14810, 'graph_builds': 0})
    lines.append({'event': 'run_end', 'run_id': 42, 'exit_status': 0,
                  'peak_vram_bytes': 9000000000, 'n_tokens_generated': len(ids),
                  'failed': False, 'error': 'none', 'total_patch_us': 0.0})

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        for e in lines:
            f.write(json.dumps(e) + '\n')
    print('wrote %s (%d tg_logits rows, %d tg ids)' % (OUT, n, len(ids)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
