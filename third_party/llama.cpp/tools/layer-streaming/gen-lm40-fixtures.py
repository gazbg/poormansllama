# Generates static layer-major telemetry fixtures for the host-only validator
# CTest targets (T-Q38LM-05). These are NOT real runs; they exercise the
# validator's 40-chunk/layer ordering and lifecycle rules deterministically.
#
#   lm40-positive.jsonl   valid 40-chunk / 64-layer / 4-slot layer-major PP
#   lm40-badorder.jsonl   one layer's chunks swapped (malformed order)
#   lm40-premature.jsonl  a slot is re-copied (new generation) before the
#                         prior generation's chunks all completed
import json, sys, os

N_LAYER = 64
N_SLOTS = 4
N_CHUNKS = 40
RUN_ID  = 9000000
OUT     = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(OUT, exist_ok=True)

def slot_of(layer):
    return layer % N_SLOTS

def gen_of(layer):
    # deterministic policy: slot s serves layers s + k*N_SLOTS; generation
    # (k+1) on that slot. n_layer=64, n_slots=4 -> one pass, 16 gens/slot.
    return (layer // N_SLOTS) + 1

def positive_events():
    ev = []
    ev.append({"event":"run_start","run_id":RUN_ID,"n_slots":N_SLOTS,"n_layer":N_LAYER,
               "schedule":"layer-major","act_slab_bytes":838860800,"source_cache":"best-effort-cold"})
    # one package copy per layer, consumed by all 40 chunks
    for layer in range(N_LAYER):
        s = slot_of(layer); g = gen_of(layer)
        ev.append({"event":"copy_submit","run_id":RUN_ID,"layer":layer,"slot":s,"generation":g,"phase":0})
        ev.append({"event":"copy_complete","run_id":RUN_ID,"layer":layer,"slot":s,"generation":g,"phase":0})
        # 40 chunks in ascending order
        for c in range(N_CHUNKS):
            ev.append({"event":"layer_compute","run_id":RUN_ID,"ubatch":c+1,"layer":layer,"slot":s,
                       "generation":g,"chunk":c,"compute_us":1000.0,"setup_us":10.0,"phase":0,"t_us":layer*100+c})
    ev.append({"event":"phase_summary","run_id":RUN_ID,"phase":"pp","tokens":N_CHUNKS*512,
               "ubatches":N_CHUNKS,"wall_us":1000000,"tokens_per_s":100.0,"bytes_transferred":1000,
               "evictions":N_LAYER-N_SLOTS,"total_transfer_us":1000.0,"total_wait_us":1.0,
               "total_compute_us":1000.0,"layers_run":N_LAYER,"graph_prep_us":10.0,"graph_builds":0})
    ev.append({"event":"run_end","run_id":RUN_ID,"exit_status":0,"peak_vram_bytes":1000000,
               "n_tokens_generated":0,"failed":False,"error":"none"})
    return ev

def to_jsonl(ev):
    return "\n".join(json.dumps(e) for e in ev) + "\n"

def write(name, ev):
    p = os.path.join(OUT, name)
    open(p, "w").write(to_jsonl(ev))
    print("wrote", p, len(ev), "events")

# positive
pos = positive_events()
write("lm40-positive.jsonl", pos)

# bad order: swap the two middle chunks of layer 10 (chunk 20 <-> 21)
bad = [dict(e) for e in pos]
# find layer 10's layer_compute events (40 of them, chunk 0..39)
l10 = [i for i,e in enumerate(bad) if e.get("event")=="layer_compute" and e.get("layer")==10]
assert len(l10)==N_CHUNKS, len(l10)
bad[l10[20]]["chunk"], bad[l10[21]]["chunk"] = bad[l10[21]]["chunk"], bad[l10[20]]["chunk"]
write("lm40-badorder.jsonl", bad)

# premature eviction: layer 0 (slot 0) chunks 0..1 complete, then slot 0 is
# re-copied for layer 4 (gen 2) BEFORE layer 0's remaining chunks complete.
# This makes layer 0's later layer_compute fail the "before matching
# copy_complete" invariant (last_copy_layer is now 4).
prem = [dict(e) for e in pos]
# locate: copy_submit(0) ... its 40 layer_compute(0), then copy_submit(4)
i_cs0  = next(i for i,e in enumerate(prem) if e.get("event")=="copy_submit"  and e.get("layer")==0)
i_cc0  = next(i for i,e in enumerate(prem) if e.get("event")=="copy_complete"and e.get("layer")==0)
# the 40 layer_compute(0) events immediately after i_cc0
lc0 = [i for i in range(i_cc0+1, len(prem)) if prem[i].get("event")=="layer_compute" and prem[i].get("layer")==0]
first_lc0 = lc0[0]
# locate the copy_submit(4)/copy_complete(4) block (slot 0, gen 2)
i_cs4  = next(i for i,e in enumerate(prem) if e.get("event")=="copy_submit"  and e.get("layer")==4)
i_cc4  = next(i for i,e in enumerate(prem) if e.get("event")=="copy_complete"and e.get("layer")==4)
# move the copy_submit(4)+copy_complete(4) block to right after the 2nd
# layer_compute(0) (i.e. after chunk 1), so slot 0 is re-copied prematurely.
blk = [prem[i_cs4], prem[i_cc4]]
# remove them from their original positions (cc4 immediately after cs4)
del prem[i_cc4]
del prem[i_cs4]
# recompute insertion point: after 2nd layer_compute(0)
# lc0 indices shifted by -2 after deletion; find 2nd layer_compute(0) now
lc0_now = [i for i,e in enumerate(prem) if e.get("event")=="layer_compute" and e.get("layer")==0]
insert_at = lc0_now[1] + 1
prem[insert_at:insert_at] = blk
write("lm40-premature.jsonl", prem)

# ---------------------------------------------------------------------------
# T-Q38CTX-01: 513-token layer-major manifest fixtures (chunk layout [512, 1])
#
#   tail513-positive.jsonl    valid 2-chunk manifest [0:512, 512:1], per-chunk
#                             act_binds with proportional widths, both chunks
#                             computed per layer
#   tail513-badchunk.jsonl    a layer_compute with chunk id 2 (outside the
#                             2-chunk manifest) -> rejected
#   tail513-missing.jsonl     stride 1 but layer 3 never computes chunk 1 ->
#                             rejected (missing final chunk)
N_EMBD = 5120
TAIL_RUN_ID = 9200000
TAIL_CHUNKS = [(0, 512), (512, 1)]      # (token_start, n_tokens)

def tail_events():
    ev = []
    ev.append({"event":"run_start","run_id":TAIL_RUN_ID,"n_slots":N_SLOTS,"n_layer":N_LAYER,
               "schedule":"layer-major","telemetry_chunk_stride":1,
               "act_slab_bytes":2*N_EMBD*(TAIL_CHUNKS[-1][0]+TAIL_CHUNKS[-1][1])*4,
               "pp_tokens":TAIL_CHUNKS[-1][0]+TAIL_CHUNKS[-1][1],"pp_n_chunks":len(TAIL_CHUNKS),
               "pp_chunks":["%d:%d" % c for c in TAIL_CHUNKS]})
    # act_bind per chunk per slab: fixed offsets, widths proportional to chunk width
    off = 0
    for ci,(start,width) in enumerate(TAIL_CHUNKS):
        nbytes = N_EMBD*width*4
        ev.append({"event":"act_bind","run_id":TAIL_RUN_ID,"chunk":ci,"slab":0,"off_bytes":off,"n_bytes":nbytes})
        ev.append({"event":"act_bind","run_id":TAIL_RUN_ID,"chunk":ci,"slab":1,"off_bytes":off,"n_bytes":nbytes})
        off += nbytes
    for layer in range(N_LAYER):
        s = slot_of(layer); g = gen_of(layer)
        ev.append({"event":"copy_submit","run_id":TAIL_RUN_ID,"layer":layer,"slot":s,"generation":g,"phase":0})
        ev.append({"event":"copy_complete","run_id":TAIL_RUN_ID,"layer":layer,"slot":s,"generation":g,"phase":0})
        for ci in range(len(TAIL_CHUNKS)):
            ev.append({"event":"layer_compute","run_id":TAIL_RUN_ID,"ubatch":ci+1,"layer":layer,"slot":s,
                       "generation":g,"chunk":ci,"compute_us":1000.0,"setup_us":10.0,"phase":0,"t_us":layer*10+ci})
    pp_tokens = TAIL_CHUNKS[-1][0]+TAIL_CHUNKS[-1][1]
    ev.append({"event":"phase_summary","run_id":TAIL_RUN_ID,"phase":"pp","tokens":pp_tokens,
               "ubatches":len(TAIL_CHUNKS),"wall_us":1000000,"tokens_per_s":100.0,"bytes_transferred":1000,
               "evictions":N_LAYER-N_SLOTS,"total_transfer_us":1000.0,"total_wait_us":1.0,
               "total_compute_us":1000.0,"layers_run":N_LAYER,"graph_prep_us":10.0,"graph_builds":0})
    ev.append({"event":"run_end","run_id":TAIL_RUN_ID,"exit_status":0,"peak_vram_bytes":1000000,
               "n_tokens_generated":0,"failed":False,"error":"none"})
    return ev

tail_pos = tail_events()
write("tail513-positive.jsonl", tail_pos)

# bad chunk id: give layer 10 an extra layer_compute with chunk 2 (manifest has 2 chunks)
badc = [dict(e) for e in tail_pos]
anchor = next(i for i,e in enumerate(badc) if e.get("event")=="layer_compute" and e.get("layer")==10 and e.get("chunk")==1)
badc.insert(anchor+1, {"event":"layer_compute","run_id":TAIL_RUN_ID,"ubatch":3,"layer":10,
                       "slot":slot_of(10),"generation":gen_of(10),"chunk":2,
                       "compute_us":1.0,"setup_us":1.0,"phase":0,"t_us":99999})
write("tail513-badchunk.jsonl", badc)

# missing final chunk: drop layer 3's chunk-1 compute entirely
missing = [dict(e) for e in tail_pos]
missing = [e for e in missing if not (e.get("event")=="layer_compute" and e.get("layer")==3 and e.get("chunk")==1)]
write("tail513-missing.jsonl", missing)

# ---------------------------------------------------------------------------
# T-GPLR-19: byte_account TOTAL-RESIDENCY-CAP fixtures (host-only, no GPU).
#
# The record now carries `row_cache` (the private VRAM the per-layer cache
# actually allocated) and `residency_cap` (--residency-budget-mib in bytes, 0 =
# no explicit cap), so the checker can hold bounded residency + cache under the
# operator's total cap from the JSONL alone and tie the cache figure to the
# declared retention policy. Every number below is the RECORDED T-GPLR-19
# budget-cap evidence (2026-09-14), not invented arithmetic:
#
#   bounded residency        4,717,996,544 B (static 957,530,112 +
#                            per_layer 1,597,483,520 + slot_window 2,088,763,392
#                            + kv 8,355,840 + state 65,863,680)
#   K=4 cache                2,291,531,776 B   (K=2 is 1,146,093,568 B)
#   arm C1  budget 16384 MiB -> allowance 12,461,872,640 B, K=4 fits
#   arm C2  budget  6000 MiB -> allowance  1,573,459,456 B, downscales to K=2
#   arm C4  budget  6000 MiB + NAMED K=4 -> REFUSED by the runner
#
#   byte-account-cap-positive.jsonl    C1's own configuration: fits, PASS
#   byte-account-uncapped.jsonl       no budget flag (cap 0), K=4: containment
#                                      does not apply, PASS
#   byte-account-cap-over.jsonl       C4's configuration as a TRACE CLAIM: the
#                                      runner refuses it, so a record asserting
#                                      it ran must be rejected
#   byte-account-cap-mismatch.jsonl   cap field not what --residency-budget-mib
#                                      says it is (16385 MiB claimed) -> reject
#   byte-account-cache-noretain.jsonl cache bytes with retention OFF: a zero
#                                      could otherwise pass as a real account
BA_RUN_ID = 9220000
BA_BOUNDED = 4717996544
BA_FULL = 21451547136
BA_FREE = 22226665472
BA_TOTAL = 25650855936
BA_RESERVE = 536870912
BA_K4_CACHE = 2291531776
MIB = 1024 * 1024

def ba_events(budget_mib, retain_k, row_cache, cap_bytes):
    ev = []
    ev.append({"event":"run_start","run_id":BA_RUN_ID,"n_slots":4,"n_layer":64,
               "schedule":"layer-major","rowset":"on","act_slab_bytes":0})
    ev.append({"event":"rowset_init","run_id":BA_RUN_ID,"policy":"moe-row-set","n_slots":4,
               "slot_bytes":BA_BOUNDED,"window_tokens":1,"n_expert":256,"top_k":8,
               "retain_k":retain_k,"t_us":0})
    ev.append({"event":"byte_account","run_id":BA_RUN_ID,
               "full_model_plus_runtime":BA_FULL,"bounded_window":BA_BOUNDED,
               "static":957530112,"per_layer":1597483520,"slot_window":2088763392,
               "kv":8355840,"state":65863680,"act_slab":0,"reserve":BA_RESERVE,
               "free_vram":BA_FREE,"total_vram":BA_TOTAL,
               "residency_budget_mib":budget_mib,"row_cache":row_cache,
               "residency_cap":cap_bytes,
               "full_exceeds_free":False,"bounded_within_free":True})
    ev.append({"event":"phase_summary","run_id":BA_RUN_ID,"phase":"tg","tokens":4,
               "ubatches":4,"wall_us":100000,"tokens_per_s":40.0,"bytes_transferred":BA_K4_CACHE,
               "evictions":0,"total_transfer_us":50000.0,"total_wait_us":100.0,
               "total_compute_us":40000.0,"layers_run":64,"graph_prep_us":10.0,"graph_builds":0})
    ev.append({"event":"run_end","run_id":BA_RUN_ID,"exit_status":0,"peak_vram_bytes":BA_FREE,
               "n_tokens_generated":4,"failed":False,"error":"none"})
    return ev

# The runner's own JSONL is compact, and the rowset opt-in marker is matched as
# the literal "rowset":"on" -- so these fixtures must not space the colon the
# way json.dumps' default separators do.
def write_compact(name, ev):
    p = os.path.join(OUT, name)
    open(p, "w").write("\n".join(json.dumps(e, separators=(",", ":")) for e in ev) + "\n")
    print("wrote", p, len(ev), "events")

# C1: explicit 16384 MiB budget, K=4 cache -> 7,009,528,320 <= 17,179,869,184
write_compact("byte-account-cap-positive.jsonl", ba_events(16384, 4, BA_K4_CACHE, 16384 * MIB))

# no budget flag: cap 0, the bounded window is its own cap (free-derived cache)
write_compact("byte-account-uncapped.jsonl", ba_events(0, 4, BA_K4_CACHE, 0))

# C4: the same K=4 cache under a 6000 MiB budget -> 7,009,528,320 > 6,291,456,000
write_compact("byte-account-cap-over.jsonl", ba_events(6000, 4, BA_K4_CACHE, 6000 * MIB))

# cap must be the flag it claims: 16385 MiB claimed under budget 16384
write_compact("byte-account-cap-mismatch.jsonl", ba_events(16384, 4, BA_K4_CACHE, 16385 * MIB))

# cache bytes with retention OFF: retain_k 0 (reactive policy), no budget
write_compact("byte-account-cache-noretain.jsonl", ba_events(0, 0, BA_K4_CACHE, 0))

# ---------------------------------------------------------------------------
# T-GPLR-22: the PRE-WORK REFUSAL trace.
#   run-refused-positive.jsonl        the real refusal shape: the package
#                                     catalog (emitted by ls.init, before
#                                     run_start) + a structured run_refused,
#                                     and nothing else -> must PASS
#   run-refused-badreason.jsonl       reason outside the enum -> rejected
#   run-refused-noreason.jsonl        no reason field at all -> rejected
#   run-refused-with-run-start.jsonl  a refusal contradicted by run_start and
#                                     phase_summary (it is pre-work) -> rejected
#   run-refused-legacy.jsonl          the PRE-EVENT refusal shape (a bare
#                                     pkg_member fragment with no run_refused).
#                                     Included so the non-breaking claim is
#                                     falsifiable: the old shape must still be
#                                     rejected, never retroactively accepted.
#   run-refused-presink*.jsonl        the T-GPLR-22 PRE-SINK shape: the capacity
#                                     and handoff preflights refuse before
#                                     ls.init, so their trace is a SINGLE
#                                     run_refused record (no pkg_member catalog).
#                                     Positive passes; the unknown-reason one is
#                                     rejected. See the block further down.
# The pkg_member rows reproduce the recorded refusal artifact shape (40 MoE
# expert-bank packages, one per trunk layer).
REFUSE_RUN_ID = 9300000

def refusal_pkgs():
    ev = []
    for layer in range(40):
        ev.append({"event":"pkg_member","run_id":REFUSE_RUN_ID,"adapter":"qwen35moe",
                   "policy":"moe-expert-bank","layer":layer,"cls":"moe-expert-bank",
                   "members":3,"bytes":452984832,"t_us":0})
    return ev

# field order matches the producer's emission (tel_run_refused) so a fixture
# reads the same as a real trace; the checker parses by name either way.
def refusal_event(reason, needed=2264924160, allowed=210310656, cap="budget"):
    ev = {"event":"run_refused","run_id":REFUSE_RUN_ID}
    if reason is not None:
        ev["reason"] = reason
    ev.update({"detail":"row-cache refusal fixture","exit_status":1,
               "bytes_needed":needed,"bytes_allowed":allowed,
               "cap_source":cap,"phase":0,"t_us":0})
    return ev

# positive: exactly what a refused run now writes
write_compact("run-refused-positive.jsonl", refusal_pkgs() + [refusal_event("row_cache_depth")])

# reason outside the enum
write_compact("run-refused-badreason.jsonl", refusal_pkgs() + [refusal_event("not_a_reason")])

# no reason field at all
write_compact("run-refused-noreason.jsonl", refusal_pkgs() + [refusal_event(None)])

# a refusal contradicted by work having started
bad = refusal_pkgs() + [refusal_event("row_cache_fit")]
bad.append({"event":"phase_summary","run_id":REFUSE_RUN_ID,"phase":"pp","tokens":512,
            "ubatches":1,"wall_us":1000000,"tokens_per_s":100.0,"bytes_transferred":1000,
            "evictions":0,"total_transfer_us":1000.0,"total_wait_us":1.0,
            "total_compute_us":1000.0,"layers_run":40,"graph_prep_us":10.0,"graph_builds":0})
bad.insert(0, {"event":"run_start","run_id":REFUSE_RUN_ID,"n_slots":4,"n_layer":40,
               "schedule":"batch-major","act_slab_bytes":0})
write_compact("run-refused-with-run-start.jsonl", bad)

# the pre-event shape: still rejected (nothing is retroactively accepted)
write_compact("run-refused-legacy.jsonl", refusal_pkgs())

# ---------------------------------------------------------------------------
# T-GPLR-22 (PRE-SINK): the capacity/handoff preflights run BEFORE ls.init, so
# their refusal trace contains no pkg_member catalog -- the standalone emitter
# writes `run_refused` as the trace's ONLY record. The existing refusal rule
# already accepts that shape (no second rule was added). An unknown reason on
# this shape is covered by run-refused-badreason: the enum check is
# shape-independent and the enum itself is one array shared by the emit sites
# and the checkers, so no per-shape duplicate of that negative is needed.
#   run-refused-presink.jsonl           a single run_refused line, reason from
#                                       the enum (handoff_preflight) -> PASS.
#                                       Numbers are the real 50k `--handoff`
#                                       arm's refusal (2026-09-14).

def presink_event(reason, detail, needed, allowed, cap):
    ev = {"event":"run_refused","run_id":REFUSE_RUN_ID,"reason":reason,"detail":detail,
          "exit_status":1,"bytes_needed":needed,"bytes_allowed":allowed,
          "cap_source":cap,"phase":0,"t_us":0}
    return ev

write_compact("run-refused-presink.jsonl", [presink_event(
    "handoff_preflight",
    "handoff capacity preflight (T-GPLR-10): weights+KV+state 21989106176 B (KV 545914880 B, "
    "state 65863680 B) + act slabs 0 B + reserve 536870912 B = 22525977088 B exceeds observed "
    "free VRAM 22226665472 B",
    22525977088, 22226665472, "free")])

# ---------------------------------------------------------------------------
# T-GPLR-22 (PRE-TIMED ABORT): the SECOND admissible refusal shape. The run
# began writing -- catalog, byte_account, run_start, and for the graph-budget
# gate graph_footprint -- and a pre-timed check stopped it. run_start IS present;
# what makes it a refusal is that NO phase completed (no phase_summary) and
# run_refused is terminal (no run_end). No checker rule reads graph_footprint, so
# the shape is pinned once rather than once per reason.
#   run-refused-aborted.jsonl           graph_budget, the full shape; numbers are
#                                       the real 2026-09-14 --graph-budget-mib 1 arm
#   run-refused-init-failed.jsonl       init_failed, which fails before anything is
#                                       written -> the PRE-WORK single-line shape
#   run-refused-with-run-end.jsonl      run_refused followed by run_end -> rejected
ABORT_RUN_ID = 9310000

def abort_pkgs():
    ev = []
    for layer in range(40):
        ev.append({"event":"pkg_member","run_id":ABORT_RUN_ID,"adapter":"qwen35moe",
                   "policy":"moe-expert-bank","layer":layer,"cls":"moe-expert-bank",
                   "members":3,"bytes":452984832,"t_us":0})
    return ev

def abort_byte_account():
    return {"event":"byte_account","run_id":ABORT_RUN_ID,"full_model_plus_runtime":21451547136,
            "bounded_window":3151424000,"static":957530112,"per_layer":1597483520,
            "slot_window":522190848,"kv":8355840,"state":65863680,"act_slab":0,
            "reserve":536870912,"free_vram":22226665472,"total_vram":25650855936,
            "residency_budget_mib":0,"row_cache":0,"residency_cap":0,
            "full_exceeds_free":False,"bounded_within_free":True}

def abort_run_start():
    return {"event":"run_start","run_id":ABORT_RUN_ID,"t0_us":ABORT_RUN_ID,
            "model":"Qwen3.6-35B-A3B","arch":"qwen35moe","n_layer":40,"n_slots":1,
            "device":0,"schedule":"batch-major","copy_source":"direct","kv_cache":"q8",
            "implicit_causal_mask":False,"telemetry_chunk_stride":10,"act_slab_bytes":0,
            "act_slab_dtype":"bf16","ctx_size":768,"handoff_mode":"off",
            "static_vram_bytes":0,"slot_vram_bytes":522194944,
            "free_vram_bytes":22226665472,"total_vram_bytes":25650855936}

def abort_graph_footprint():
    return {"event":"graph_footprint","run_id":ABORT_RUN_ID,"n_graph_sets":2,"total_graphs":84,
            "io_ctx_reserved_bytes":201326592,"io_ctx_used_bytes":28866788672,
            "host_input_bytes":68681728,"host_input_used_bytes":5566148,
            "act_slab_bytes":0,"slot_vram_bytes":522194944,"graph_budget_mib":1,"t_us":0}

def abort_refusal(reason, detail, needed, allowed, cap):
    return {"event":"run_refused","run_id":ABORT_RUN_ID,"reason":reason,"detail":detail,
            "exit_status":1,"bytes_needed":needed,"bytes_allowed":allowed,
            "cap_source":cap,"phase":0,"t_us":0}

# graph_budget: the complete aborted shape, with the real arm's arithmetic.
write_compact("run-refused-aborted.jsonl",
              abort_pkgs() + [abort_byte_account(), abort_run_start(), abort_graph_footprint(),
                              abort_refusal("graph_budget",
                                            "graph footprint exceeds the budget: io_ctx_reserved "
                                            "201326592 B > 1048576 B (--graph-budget-mib)",
                                            201326592, 1048576, "budget")])

# init_failed: the pre-work single-line shape, because nothing was written.
write_compact("run-refused-init-failed.jsonl",
              [abort_refusal("init_failed",
                             "layer streaming init failed (preflight): failed to open telemetry file",
                             0, 0, "none")])

# negative: run_refused is terminal, so a run_end after it is a contradiction.
_with_end = abort_pkgs() + [abort_run_start(),
                            abort_refusal("prebuild_failed", "prebuild failed", 0, 0, "none")]
_with_end.append({"event":"run_end","run_id":ABORT_RUN_ID,"exit_status":1,"peak_vram_bytes":0,
                  "n_tokens_generated":0,"failed":True,"error":"aborted","total_patch_us":0.0})
write_compact("run-refused-with-run-end.jsonl", _with_end)

# ---------------------------------------------------------------------------
# T-Q38CAP-01: stride-1 completeness on a 3-chunk manifest [512, 512, 1]
#   tail1025-positive.jsonl  valid 3-chunk manifest, all chunks per layer
#   tail1025-hole.jsonl      layer 5 never computes chunk 1 (interior hole at
#                            stride 1) -> rejected by the completeness rule
TAIL2_RUN_ID = 9210000
TAIL2_CHUNKS = [(0, 512), (512, 512), (1024, 1)]

def tail2_events():
    ev = []
    ev.append({"event":"run_start","run_id":TAIL2_RUN_ID,"n_slots":N_SLOTS,"n_layer":N_LAYER,
               "schedule":"layer-major","telemetry_chunk_stride":1,
               "act_slab_bytes":2*N_EMBD*(TAIL2_CHUNKS[-1][0]+TAIL2_CHUNKS[-1][1])*4,
               "pp_tokens":TAIL2_CHUNKS[-1][0]+TAIL2_CHUNKS[-1][1],"pp_n_chunks":len(TAIL2_CHUNKS),
               "pp_chunks":["%d:%d" % c for c in TAIL2_CHUNKS]})
    off = 0
    for ci,(start,width) in enumerate(TAIL2_CHUNKS):
        nbytes = N_EMBD*width*4
        ev.append({"event":"act_bind","run_id":TAIL2_RUN_ID,"chunk":ci,"slab":0,"off_bytes":off,"n_bytes":nbytes})
        ev.append({"event":"act_bind","run_id":TAIL2_RUN_ID,"chunk":ci,"slab":1,"off_bytes":off,"n_bytes":nbytes})
        off += nbytes
    for layer in range(N_LAYER):
        s = slot_of(layer); g = gen_of(layer)
        ev.append({"event":"copy_submit","run_id":TAIL2_RUN_ID,"layer":layer,"slot":s,"generation":g,"phase":0})
        ev.append({"event":"copy_complete","run_id":TAIL2_RUN_ID,"layer":layer,"slot":s,"generation":g,"phase":0})
        for ci in range(len(TAIL2_CHUNKS)):
            ev.append({"event":"layer_compute","run_id":TAIL2_RUN_ID,"ubatch":ci+1,"layer":layer,"slot":s,
                       "generation":g,"chunk":ci,"compute_us":1000.0,"setup_us":10.0,"phase":0,"t_us":layer*10+ci})
    pp_tokens = TAIL2_CHUNKS[-1][0]+TAIL2_CHUNKS[-1][1]
    ev.append({"event":"phase_summary","run_id":TAIL2_RUN_ID,"phase":"pp","tokens":pp_tokens,
               "ubatches":len(TAIL2_CHUNKS),"wall_us":1000000,"tokens_per_s":100.0,"bytes_transferred":1000,
               "evictions":N_LAYER-N_SLOTS,"total_transfer_us":1000.0,"total_wait_us":1.0,
               "total_compute_us":1000.0,"layers_run":N_LAYER,"graph_prep_us":10.0,"graph_builds":0})
    ev.append({"event":"run_end","run_id":TAIL2_RUN_ID,"exit_status":0,"peak_vram_bytes":1000000,
               "n_tokens_generated":0,"failed":False,"error":"none"})
    return ev

tail2_pos = tail2_events()
write("tail1025-positive.jsonl", tail2_pos)

# interior hole: drop layer 5's chunk-1 compute (chunks 0 and 2 remain)
hole = [dict(e) for e in tail2_pos]
hole = [e for e in hole if not (e.get("event")=="layer_compute" and e.get("layer")==5 and e.get("chunk")==1)]
write("tail1025-hole.jsonl", hole)

print("done")
