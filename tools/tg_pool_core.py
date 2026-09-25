#!/usr/bin/env python3
"""T-GPLR-59 follow-up: host-only counterfactual replay of the 12-position
prediction pool over recorded mode-2 traces.

Replays the ordered expert demand and published predictor candidates
(`spec.candidate_ids`) of an existing mode-2 JSONL trace through a simulated
12-position, per-logical-layer prediction pool under the `ideal-next-reap`
completion model. Evidence tool only: no runtime change, no model, CUDA, GPU,
mutex, or network. It cannot establish CUDA behavior, transfer latency, PP
speed, or a predictor-quality result.

Falsifiable hypothesis: the separate pool creates useful later demand hits
which the occupied real cache could not admit.

Input contract (plan, "T-GPLR-59 follow-up design"): one rowset_init, one
rowset_fetch and one mode-2 `spec` record per (ubatch, layer) stage, paired by
that key (the emitter writes the two records in the opposite line order);
file order and expert-id order are preserved. The `spec` record is the
proposal made AFTER that stage's demand; the LAST proposal of each logical
layer has no later occurrence and is reported as unused, never applied. The
source budget (`row_spec_budget`) is reported separately from the pool budget
(12): a replay over an 8-item source is an explicit `recorded-8` lower bound,
never padded or repeated into a 12-item arm.

Two evidence modes stay distinct:
  overlay      (primary) - replay the real cache including the recorded real
               speculative fills (PENDING/READY/CANCELLED transitions,
               generation boundaries, capacity-LRU evictions, mirroring the
               telemetry validator's own retention model). Exact hit and
               retention parity with the recording is REQUIRED before any pool
               result is produced. The pool never writes real-cache state.
  demand-only  (diagnostic) - ignore recorded fills, rebuild a demand-only
               real cache. The result is labeled a clean-baseline diagnostic
               and is never an overlay measurement.

Completion assumption: ideal-next-reap (the measured lead floor, not a
tunable convenience): a copy issued after stage t is PENDING during demand
t+1, becomes READY after the end of t+1, and can first serve demand t+2. A
PENDING match is cancelled with immediate fallback; a demand never waits
(modeled completion waits are counted and structurally zero).

Roles are explicit session declarations (--leg arm=|null=|off=); the CLI never
infers a role from a filename. T-GPLR-60 uses the same lifecycle core with its
own frozen candidate streams.
"""

import copy
import datetime
import hashlib
import json
import os

SCHEMA = 'gplr59-pool-counterfactual/1'
POOL_BUDGET = 12  # positions per logical layer AND the per-stage admission bound
COMPLETION = ('ideal-next-reap: issued after stage t; PENDING at demand t+1; READY after '
              'the end of t+1; first serveable at t+2; a PENDING match is cancelled with '
              'immediate fallback and never waited on')
ROLES = ('arm', 'null', 'off')
FNV_BASIS = 1469598103934665603  # the engine's (non-standard) FNV-1a-64 basis
FNV_PRIME = 1099511628211
SPEC_TOTAL_FIELDS = ('prop', 'arm_count', 'skipped_present', 'skipped_dup',
                     'plan_refused_capacity', 'submitted', 'consumed',
                     'matched_by_demand', 'cancelled_demand', 'cancelled_generation',
                     'cancelled_capacity', 'dup_refused')


class Refusal(Exception):
    """An input the contract refuses: exit 2, no measurement."""


# ---------------------------------------------------------------- trace input
def _id_list(value, what):
    if value is None:
        return None
    if not isinstance(value, list):
        raise Refusal('%s is not a JSON array' % what)
    out = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise Refusal('%s carries a non-integer id' % what)
        out.append(item)
    return out


def _fnv(text):
    h = FNV_BASIS
    for byte in text.encode('utf-8'):
        h ^= byte
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def load_trace(path, mode):
    """Parse one recorded leg into ordered stages. Roles are NOT inferred here."""
    if not os.path.isfile(path):
        raise Refusal('no such trace: %s' % path)
    run_start = None
    init = None
    fetches = {}
    specs = {}
    order = []
    run_ids = set()
    spec_totals = {}
    pool_recording = False
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if '{"event"' not in line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = ev.get('event')
            if name == 'run_start':
                if run_start is not None:
                    raise Refusal('mixed traces: more than one run_start in %s' % path)
                run_start = ev
            elif name == 'rowset_init':
                if init is not None:
                    raise Refusal('mixed traces: more than one rowset_init in %s' % path)
                init = ev
            elif name in ('rowset_fetch', 'spec'):
                run_ids.add(ev.get('run_id'))
                key = (int(ev.get('ubatch', -1)), int(ev.get('layer', -1)))
                target = fetches if name == 'rowset_fetch' else specs
                if key in target:
                    raise Refusal('mixed traces: duplicate %s stage %s in %s' % (name, key, path))
                target[key] = ev
                if name == 'rowset_fetch':
                    order.append(key)
                else:
                    for field in SPEC_TOTAL_FIELDS:
                        value = ev.get(field)
                        if isinstance(value, int) and not isinstance(value, bool):
                            spec_totals[field] = spec_totals.get(field, 0) + value
            elif name in ('spec_pool', 'spec_pool_init'):
                pool_recording = True
    if len(run_ids) > 1:
        raise Refusal('mixed traces: events of %d runs in %s' % (len(run_ids), path))
    if pool_recording:
        raise Refusal('trace records the runtime prediction pool (spec_pool* events): the '
                      'counterfactual only scores pool-less recordings (%s)' % path)
    if run_start is not None:
        if int(run_start.get('row_spec_pool', 0) or 0) > 0:
            raise Refusal('trace records the runtime prediction pool (run_start.row_spec_pool '
                          '> 0): the counterfactual only scores pool-less recordings (%s)' % path)
        if run_ids and run_start.get('run_id') not in run_ids:
            raise Refusal('mixed traces: row run_id does not match run_start in %s' % path)
    if init is None:
        raise Refusal('no rowset_init in %s' % path)
    if not order:
        raise Refusal('no rowset_fetch demand rows in %s' % path)
    if set(specs) != set(fetches):
        raise Refusal('incomplete stage pairing: %d spec records vs %d rowset_fetch records '
                      'in %s' % (len(specs), len(fetches), path))

    retain_k = int(init.get('retain_k', 0) or 0)
    n_expert = int(init.get('n_expert', 0) or 0)
    top_k = int(init.get('top_k', 0) or 0)
    window_tokens = int(init.get('window_tokens', 1) or 0)
    if retain_k <= 0 or n_expert <= 0 or top_k <= 0:
        raise Refusal('no retention policy (rowset_init needs retain_k, n_expert, top_k > 0)')
    if window_tokens != 1:
        raise Refusal('W != 1 (window_tokens=%d): this replay only scores W=1 streams' % window_tokens)
    source_budget = int((run_start or {}).get('row_spec_budget', 0) or 0)
    if source_budget <= 0:
        raise Refusal('run_start has no positive row_spec_budget (source budget unknown)')

    stages = []
    previous = None
    row_bytes_by_layer = {}
    for key in order:
        if previous is not None and key <= previous:
            raise Refusal('demand rows are not in ascending (ubatch, layer) order in %s '
                          '(mixed or interleaved runs)' % path)
        previous = key
        fetch = fetches[key]
        spec = specs[key]
        if int(spec.get('mode', 0) or 0) != 2:
            raise Refusal('spec %s is not a mode-2 record' % (key,))
        candidates = _id_list(spec.get('candidate_ids'), 'spec %s candidate_ids' % (key,))
        if candidates is None:
            raise Refusal('missing raw candidates: spec %s has no candidate_ids (a missing '
                          'source is UNVERIFIED, never zero)' % (key,))
        if 'prop' in spec and int(spec['prop']) != len(candidates):
            raise Refusal('contradictory spec fields: prop != len(candidate_ids) at %s' % (key,))
        fills = _id_list(spec.get('fill_ids'), 'spec %s fill_ids' % (key,))
        if fills is None:
            if mode == 'overlay':
                raise Refusal('overlay mode needs fill_ids: spec %s has none (use --mode '
                              'demand-only for a clean-baseline diagnostic)' % (key,))
            fills = []
        if len(fills) != len(set(fills)):
            raise Refusal('contradictory spec fields: fill_ids repeats an id at %s' % (key,))
        union_text = str(fetch.get('union_ids', ''))
        union = [int(x) for x in union_text.split(',') if x != '']
        if not union:
            raise Refusal('rowset_fetch %s has an empty union_ids' % (key,))
        if len(union) != len(set(union)):
            raise Refusal('rowset_fetch %s union_ids contains duplicates' % (key,))
        if 'hash' in fetch and str(fetch.get('hash', '')) != '':
            if _fnv(union_text) != int(fetch['hash']):
                raise Refusal('contradictory known field at %s: hash does not match the '
                              'union_ids FNV-1a-64 digest' % (key,))
        route_union = int(fetch.get('route_union', -1) or 0)
        if route_union != len(union) or route_union != top_k:
            raise Refusal('W=1 identity broken at %s: route_union=%d, |union_ids|=%d, top_k=%d'
                          % (key, route_union, len(union), top_k))
        for ident in list(union) + list(candidates) + list(fills):
            if ident < 0 or ident >= n_expert:
                raise Refusal('invalid/out-of-range expert id %d at stage %s' % (ident, key))
        pool_hits = _id_list(fetch.get('pool_hit_ids'), 'rowset_fetch %s pool_hit_ids' % (key,))
        if pool_hits:
            raise Refusal('trace records runtime pool hits (pool_hit_ids at %s): the '
                          'counterfactual only scores pool-less recordings' % (key,))
        bank = int(fetch.get('bytes_full_bank', 0) or 0)
        staged = int(fetch.get('bytes_staged', 0) or 0)
        fetched = int(fetch.get('n_rows_fetched', -1))
        resident = int(fetch.get('n_rows_resident', -1))
        retained = int(fetch.get('n_rows_retained', -1))
        if fetched < 0 or resident < 0 or retained < 0:
            raise Refusal('rowset_fetch %s is missing retention counts' % (key,))
        if resident != retained:
            raise Refusal('contradictory real-cache identity at %s: n_rows_resident=%d != '
                          'n_rows_retained=%d' % (key, resident, retained))
        if resident + fetched != route_union:
            raise Refusal('contradictory real-cache identity at %s: route_union=%d != '
                          'resident=%d + fetched=%d (a recorded pool-less trace must close)'
                          % (key, route_union, resident, fetched))
        if bank <= 0 or bank % n_expert != 0:
            raise Refusal('inconsistent byte arithmetic at %s: bytes_full_bank=%d is not an '
                          'integral expert-slab size' % (key, bank))
        stage_row_bytes = bank // n_expert
        if staged * n_expert != fetched * bank:
            raise Refusal('inconsistent byte arithmetic at %s: bytes_staged/n_rows_fetched '
                          'disagrees with bytes_full_bank/n_expert' % (key,))
        prior = row_bytes_by_layer.get(key[1])
        if prior is None:
            row_bytes_by_layer[key[1]] = stage_row_bytes
        elif prior != stage_row_bytes:
            raise Refusal('inconsistent byte arithmetic: row_bytes changed within layer %d '
                          '(%d then %d at %s)' % (key[1], prior, stage_row_bytes, key))
        retired_value = fetch.get('n_rows_retired', None)
        stages.append({
            'key': key, 'ubatch': key[0], 'layer': key[1],
            'generation': int(fetch.get('generation', 0) or 0),
            'phase': int(fetch.get('phase', 0) or 0),
            'union': union,
            'recorded_resident': resident, 'recorded_retained': retained,
            'recorded_fetched': fetched,
            'recorded_retired': int(retired_value) if isinstance(retired_value, int) else None,
            'candidates': candidates, 'fills': fills,
            'row_bytes': stage_row_bytes,
        })
    # The proposal stream: the LAST proposal of each layer has no later
    # occurrence to serve and stays unused (input contract).
    last_of_layer = {}
    for index, stage in enumerate(stages):
        last_of_layer[stage['layer']] = index
    for index, stage in enumerate(stages):
        stage['apply_proposal'] = index != last_of_layer[stage['layer']]

    return {
        'path': path, 'mode': mode,
        'model': str((run_start or {}).get('model', '')),
        'n_expert': n_expert, 'top_k': top_k, 'retain_k': retain_k,
        'capacity': min(n_expert, retain_k * top_k),
        'source_budget': source_budget,
        'row_bytes_by_layer': row_bytes_by_layer,
        'stages': stages, 'spec_totals': spec_totals,
    }


def stream_signature(stages):
    """The offline screen's demand signature (predictor-screen-gplr.py, verbatim):
    sha256 over `layer|ids` per row, 16 hex chars. Comparable with recorded values."""
    h = hashlib.sha256()
    for stage in stages:
        h.update(('%s|%s\n' % (stage['layer'], ','.join(str(x) for x in stage['union']))).encode())
    return h.hexdigest()[:16]


def demand_rows_signature(stages):
    """The strict pair gate: ordered (ubatch, layer, union_ids) identity."""
    h = hashlib.sha256()
    for stage in stages:
        h.update(('%s|%s|%s\n' % (stage['ubatch'], stage['layer'],
                                  ','.join(str(x) for x in stage['union']))).encode())
    return h.hexdigest()[:16]


# ----------------------------------------------------------- real-cache replay
class RealCacheLayer:
    """Capacity-LRU over min(n_expert, retain_k*top_k) rows, mirroring the telemetry
    validator's `ls_retention_errors` step for step (which calls the runtime's own
    pure rules): classify READY hits, supersede demanded PENDING rows, protect the
    demanded union from eviction (a speculative row is the FIRST victim -- P3 --
    then the least recently routed), place the union in ascending expert-id order,
    reap (PENDING -> READY and speculative, CANCELLED released), then issue the
    stage's fills at fresh recency. The counterfactual pool never writes here.
    """

    PENDING, READY, CANCELLED = 1, 2, 3

    def __init__(self, cap):
        self.cap = cap
        self.rows = {}  # expert -> dict(state, recency, spec)
        self.tick = 0
        self.last_gen = None
        self.last_phase = None

    def boundary(self, generation, phase):
        """Step 0: a generation boundary cancels every in-flight fill issued under
        a stale generation; a PHASE boundary drains (erases) in-flight positions
        before the new phase's first window."""
        if self.last_gen is None or self.last_gen == generation:
            self.last_gen, self.last_phase = generation, phase
            return
        new_phase = self.last_phase is not None and self.last_phase != phase
        for expert in list(self.rows):
            row = self.rows[expert]
            in_flight = row['state'] in (self.PENDING, self.CANCELLED)
            if in_flight and new_phase:
                del self.rows[expert]  # drained at the phase boundary
            elif row['state'] == self.PENDING:
                row['state'] = self.CANCELLED  # stale generation
        self.last_gen, self.last_phase = generation, phase

    def stage(self, union, fills):
        """One demand stage. Returns (hits, evicted, applied_fills)."""
        uniq = list(union)
        uniq_set = set(uniq)
        hits = []
        for expert in uniq:
            row = self.rows.get(expert)
            if row is None:
                continue
            if row['state'] == self.READY:
                hits.append(expert)
            elif row['state'] == self.PENDING:
                row['state'] = self.CANCELLED  # superseded by the demand's own copy
        hit_set = set(hits)
        occupied = len(self.rows)
        need = len(uniq) - len(hits)  # the runtime's own accounting, verbatim
        evicted = 0
        while occupied + need > self.cap:
            victim = None
            for expert, row in self.rows.items():
                if row['state'] != self.READY:
                    continue  # in-flight bytes: released by the reap, never evicted
                if expert in uniq_set:
                    continue  # needed by this window: never evicted
                if victim is None or self._victim_better(row, self.rows[victim]):
                    victim = expert
            if victim is None:
                raise Refusal('recorded retention cannot fit the union in the declared '
                              'capacity (demanded or in-flight rows exceed cap)')
            del self.rows[victim]
            occupied -= 1
            evicted += 1
        for expert in sorted(uniq_set):
            row = self.rows.get(expert)
            if row is None:
                row = {'state': self.READY, 'recency': 0, 'spec': 0}
                self.rows[expert] = row
            # a hit refreshes; a superseded CANCELLED row is revived by the
            # demand's own copy; a consumed fill is an ordinary resident again
            row['state'] = self.READY
            row['spec'] = 0
            row['recency'] = self._stamp()
        if len(self.rows) > self.cap:
            raise Refusal('internal model violation: real cache exceeds declared capacity')
        # reap first (PENDING -> READY and speculative, CANCELLED released), then
        # issue this stage's fills in fill_ids order at fresh recency
        for expert in list(self.rows):
            row = self.rows[expert]
            if row['state'] == self.CANCELLED:
                del self.rows[expert]  # released without a residency claim
            elif row['state'] == self.PENDING:
                row['state'] = self.READY
                row['spec'] = 1  # a ready fill is the first victim
        applied = []
        for expert in fills:
            if expert in self.rows:
                raise Refusal('contradictory real-cache identity: spec fill_ids repeats a '
                              'row the cache already holds')
            self.rows[expert] = {'state': self.PENDING, 'recency': self._stamp(), 'spec': 0}
            applied.append(expert)
        return hits, evicted, applied

    @staticmethod
    def _victim_better(a, b):
        # llama_ls_spec_victim_better, verbatim: P3 -- a speculative row is the
        # FIRST victim; then the least recently routed position loses.
        if a['spec'] != b['spec']:
            return a['spec'] > b['spec']
        return a['recency'] < b['recency']

    def _stamp(self):
        self.tick += 1
        return self.tick


def replay_real(leg):
    """Overlay or demand-only real-cache replay; freezes per-stage facts for the pool.

    Overlay mode gates on exact hit/retention/retired parity with the recording;
    a mismatch refuses the pool result (the pool never explains away a real
    replay divergence).
    """
    stages = leg['stages']
    fills_for_stage = (lambda stage: stage['fills']) if leg['mode'] == 'overlay' else None
    facts = replay_real_facts(stages, leg['capacity'], fills_for_stage)
    counts = {'real_hits': 0, 'real_fetched': 0, 'real_evicted': 0,
              'fill_rows_applied': 0, 'parity_stages': 0, 'divergent_stages': 0}
    parity_bad = []
    for source, fact in zip(stages, facts):
        source['occ'] = fact['occ']
        source['real_hit_ids'] = fact['real_hit_ids']
        source['miss_ids'] = fact['miss_ids']
        source['real_rows_after'] = fact['real_rows_after']
        counts['real_hits'] += len(fact['real_hit_ids'])
        counts['real_fetched'] += len(fact['miss_ids'])
        counts['real_evicted'] += fact['real_evicted']
        counts['fill_rows_applied'] += fact['real_fill_rows_applied']
        if leg['mode'] == 'overlay':
            ok = (len(fact['real_hit_ids']) == source['recorded_resident'] == source['recorded_retained']
                  and len(fact['miss_ids']) == source['recorded_fetched'])
            if ok and source['recorded_retired'] is not None:
                ok = fact['real_evicted'] == source['recorded_retired']
            if ok:
                counts['parity_stages'] += 1
            else:
                parity_bad.append(source['key'])
        elif len(fact['real_hit_ids']) != source['recorded_resident']:
            counts['divergent_stages'] += 1
    if leg['mode'] == 'overlay' and parity_bad:
        raise Refusal('contradictory real-cache identity: overlay replay parity failed at '
                      '%d/%d stages (first: %s); the pool result is refused'
                      % (len(parity_bad), len(stages), parity_bad[0]))
    counts['n_stages'] = len(stages)
    counts['capacity'] = leg['capacity']
    return counts


def replay_real_facts(stages, capacity, fills_for_stage=None):
    """Return demand facts after one shared real-cache replay."""
    layers = {}
    occurrences = {}
    facts = []
    for source in stages:
        layer_id = source['layer']
        layer = layers.setdefault(layer_id, RealCacheLayer(capacity))
        layer.boundary(source['generation'], source['phase'])
        fills = fills_for_stage(source) if fills_for_stage else []
        hits, evicted, applied = layer.stage(source['union'], fills)
        occurrences[layer_id] = occurrences.get(layer_id, 0) + 1
        hit_set = set(hits)
        stage = copy.copy(source)
        stage['occ'] = occurrences[layer_id]
        stage['real_hit_ids'] = hits
        stage['miss_ids'] = [x for x in source['union'] if x not in hit_set]
        stage['real_rows_after'] = frozenset(layer.rows)
        stage['real_evicted'] = evicted
        stage['real_fill_rows_applied'] = len(applied)
        facts.append(stage)
    return facts


# ----------------------------------------------------------------- pool replay
class PoolLayer:
    """Twelve private prediction positions per logical layer (the T-GPLR-59 first
    arm), ideal-next-reap lifecycle. Mirrors the runtime's pure pool rules
    (llama_ls_pool_*): admission prefers FREE, otherwise evicts only the
    least-recently-issued READY row; PENDING/CANCELLED bytes are never
    overwritten; a full pool with no READY victim refuses with a capacity
    reason and no wait.
    """

    FREE, PENDING, READY, CANCELLED = 0, 1, 2, 3

    def __init__(self, capacity):
        self.capacity = capacity
        self.pos = [{'expert': -1, 'state': self.FREE, 'issue_ord': -1,
                     'recency': -1, 'generation': 0} for _ in range(capacity)]
        self.next_ord = 0
        self.next_recency = 0
        self.max_occupancy = 0
        self.overwrites = 0

    def occupied(self):
        return sum(1 for slot in self.pos if slot['state'] != self.FREE)

    def boundary(self, generation):
        """Step 1 at demand: the generation transition. A stale READY row retires;
        a stale PENDING row becomes CANCELLED and stays occupied until modeled
        completion. Returns (retired_ready_ords, cancelled_pending_ords)."""
        retired, cancelled = [], []
        for slot in self.pos:
            if slot['state'] == self.FREE or slot['generation'] == generation:
                continue
            if slot['state'] == self.READY:
                retired.append(slot['issue_ord'])
                self._free(slot)
            elif slot['state'] == self.PENDING:
                slot['state'] = self.CANCELLED
                cancelled.append(slot['issue_ord'])
        return retired, cancelled

    def lookup(self, expert, generation):
        """Step 2: a current-generation READY row is a hit; a current-generation
        PENDING row is a fallback; CANCELLED and stale rows are never hits."""
        hit = pending = -1
        for index, slot in enumerate(self.pos):
            if slot['expert'] != expert or slot['state'] == self.FREE:
                continue
            if slot['generation'] != generation:
                continue  # a stale row never satisfies demand
            if slot['state'] == self.READY and hit < 0:
                hit = index
            elif slot['state'] == self.PENDING and pending < 0:
                pending = index
        return hit, pending

    def contains(self, expert):
        return any(slot['expert'] == expert and slot['state'] != self.FREE
                   for slot in self.pos)

    def cancel_pending(self, index):
        slot = self.pos[index]
        if slot['state'] != self.PENDING:
            return -1
        slot['state'] = self.CANCELLED  # bytes held until modeled completion
        return slot['issue_ord']

    def retire_hit(self, index):
        slot = self.pos[index]
        if slot['state'] != self.READY:
            return -1
        issue_ord = slot['issue_ord']
        self._free(slot)
        return issue_ord

    def reap(self):
        """Step 4a: after the demand ready point -- PENDING becomes READY and
        CANCELLED becomes FREE. Rows are issued only after a stage's reap, so a
        row is PENDING through exactly the next demand and READY after that: the
        ideal-next-reap lead floor of 2. Returns PENDING -> READY completions."""
        completed = 0
        for slot in self.pos:
            if slot['state'] == self.PENDING:
                slot['state'] = self.READY
                completed += 1
            elif slot['state'] == self.CANCELLED:
                self._free(slot)
        return completed

    def admit(self, expert, generation):
        """Step 4b: prefer FREE, else evict only the least-recently-issued READY
        row. Never overwrite PENDING/CANCELLED bytes. Returns (ok, evicted_ord)."""
        index = -1
        for candidate, slot in enumerate(self.pos):
            if slot['state'] == self.FREE and slot['expert'] < 0:
                index = candidate
                break
        evicted_ord = -1
        if index < 0:
            victim = -1
            for candidate, slot in enumerate(self.pos):
                if slot['state'] != self.READY:
                    continue
                if victim < 0 or slot['recency'] < self.pos[victim]['recency']:
                    victim = candidate
            if victim < 0:
                return False, -1  # full and no READY victim: capacity refusal, no wait
            evicted_ord = self.pos[victim]['issue_ord']
            index = victim
        slot = self.pos[index]
        if slot['state'] in (self.PENDING, self.CANCELLED):
            self.overwrites += 1  # must never happen (invariant)
        slot.update(expert=expert, state=self.PENDING,
                    issue_ord=self.next_ord, recency=self.next_recency,
                    generation=generation)
        self.next_ord += 1
        self.next_recency += 1
        self.max_occupancy = max(self.max_occupancy, self.occupied())
        return True, evicted_ord

    def _free(self, slot):
        slot.update(expert=-1, state=self.FREE, issue_ord=-1, recency=-1, generation=0)


def replay_pool_stages(stages, candidates_for_stage, n_expert, capacity=POOL_BUDGET):
    """Replay one ordered stage stream through the shared pool lifecycle."""
    layers = {}
    totals = {
        'proposals': 0, 'admitted': 0, 'hits': 0, 'pending_fallbacks': 0,
        'stale_retirements': 0, 'generation_cancellations': 0,
        'generation_cancelled_pending': 0, 'evictions': 0,
        'modeled_completions': 0, 'modeled_completion_waits': 0,
        'skipped_real': 0, 'skipped_pool': 0, 'skipped_dup': 0, 'skipped_invalid': 0,
        'refused_budget': 0, 'refused_capacity': 0,
        'unused_final_proposals': 0, 'unused_final_candidates': 0,
        'h2d_bytes': 0, 'd2d_bytes': 0,
    }
    fates = {}
    issue_occ = {}
    lead_hist = {}

    def mark(layer_id, issue_ord, fate, lead=None, min_lead=None):
        key = (layer_id, issue_ord)
        if key not in fates or fates[key] != 'live':
            raise Refusal('internal model violation: position %s marked twice' % (key,))
        fates[key] = fate
        if lead is not None:
            if min_lead is not None and lead < min_lead:
                raise Refusal('internal model violation: a %s at lead %d (< %d)'
                              % (fate, lead, min_lead))
            if fate == 'hit':
                lead_hist[lead] = lead_hist.get(lead, 0) + 1

    for stage in stages:
        layer_id = stage['layer']
        pool = layers.setdefault(layer_id, PoolLayer(capacity))

        retired, cancelled = pool.boundary(stage['generation'])
        for issue_ord in retired:
            mark(layer_id, issue_ord, 'stale_retired')
            totals['stale_retirements'] += 1
            totals['generation_cancellations'] += 1
        for issue_ord in cancelled:
            mark(layer_id, issue_ord, 'generation_cancelled')
            totals['generation_cancelled_pending'] += 1
            totals['generation_cancellations'] += 1

        retire = []
        for expert in stage['miss_ids']:
            hit, pending = pool.lookup(expert, stage['generation'])
            if hit >= 0:
                issue_ord = pool.pos[hit]['issue_ord']
                lead = stage['occ'] - issue_occ[(layer_id, issue_ord)]
                mark(layer_id, issue_ord, 'hit', lead=lead, min_lead=2)
                totals['hits'] += 1
                totals['d2d_bytes'] += stage['row_bytes']
                retire.append(hit)
            elif pending >= 0:
                issue_ord = pool.pos[pending]['issue_ord']
                pool.cancel_pending(pending)
                lead = stage['occ'] - issue_occ[(layer_id, issue_ord)]
                if lead != 1:
                    raise Refusal('internal model violation: a PENDING fallback at lead %d' % lead)
                mark(layer_id, issue_ord, 'pending_fallback', lead=lead)
                totals['pending_fallbacks'] += 1
        for index in retire:
            pool.retire_hit(index)

        totals['modeled_completions'] += pool.reap()
        candidates = list(candidates_for_stage(stage))
        if not stage['apply_proposal']:
            totals['unused_final_proposals'] += 1
            totals['unused_final_candidates'] += len(candidates)
            pool.max_occupancy = max(pool.max_occupancy, pool.occupied())
            continue

        seen = set()
        stage_admitted = 0
        counts = {'admitted': 0, 'skipped_real': 0, 'skipped_pool': 0,
                  'skipped_dup': 0, 'skipped_invalid': 0,
                  'refused_budget': 0, 'refused_capacity': 0}
        for expert in candidates:
            totals['proposals'] += 1
            if expert < 0 or expert >= n_expert:
                counts['skipped_invalid'] += 1
                continue
            if expert in seen:
                counts['skipped_dup'] += 1
                continue
            seen.add(expert)
            if expert in stage['real_rows_after']:
                counts['skipped_real'] += 1
                continue
            if pool.contains(expert):
                counts['skipped_pool'] += 1
                continue
            if stage_admitted >= capacity:
                counts['refused_budget'] += 1
                continue
            ok, evicted_ord = pool.admit(expert, stage['generation'])
            if not ok:
                counts['refused_capacity'] += 1
                continue
            counts['admitted'] += 1
            totals['h2d_bytes'] += stage['row_bytes']
            stage_admitted += 1
            issue_ord = pool.next_ord - 1
            fates[(layer_id, issue_ord)] = 'live'
            issue_occ[(layer_id, issue_ord)] = stage['occ']
            if evicted_ord >= 0:
                mark(layer_id, evicted_ord, 'evicted')
                totals['evictions'] += 1
        if sum(counts.values()) != len(candidates):
            raise Refusal('contradictory pool accounting at %s' % (stage.get('key', stage['layer']),))
        for name, value in counts.items():
            totals[name] += value
        pool.max_occupancy = max(pool.max_occupancy, pool.occupied())

    live_at_end = sum(1 for fate in fates.values() if fate == 'live')
    if len(fates) != totals['admitted']:
        raise Refusal('contradictory pool accounting: %d fates for %d admissions'
                      % (len(fates), totals['admitted']))
    terminal = (totals['hits'] + totals['pending_fallbacks'] +
                totals['generation_cancellations'] + totals['evictions'] + live_at_end)
    if totals['admitted'] != terminal:
        raise Refusal('contradictory pool accounting: fate closure %d != %d'
                      % (totals['admitted'], terminal))
    totals['live_at_end'] = live_at_end
    totals['max_occupancy'] = max([0] + [pool.max_occupancy for pool in layers.values()])
    totals['inflight_overwrites'] = sum(pool.overwrites for pool in layers.values())
    if totals['inflight_overwrites']:
        raise Refusal('internal model violation: %d PENDING/CANCELLED overwrites'
                      % totals['inflight_overwrites'])
    if totals['max_occupancy'] > capacity:
        raise Refusal('internal model violation: pool occupancy %d exceeds %d'
                      % (totals['max_occupancy'], capacity))
    return totals, lead_hist


def replay_pool(leg):
    return replay_pool_stages(
        leg['stages'],
        lambda stage: stage['candidates'],
        leg['n_expert'],
        POOL_BUDGET,
    )


# ------------------------------------------------------- metrics and reporting
def _share(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize_leg(leg, real, pool_out, mode):
    totals, lead_hist = pool_out
    demand_rows = len(leg['stages'])
    demand_ids = sum(len(stage['union']) for stage in leg['stages'])
    resolved = totals['hits'] + totals['pending_fallbacks']
    if leg['source_budget'] < POOL_BUDGET:
        source_label = ('recorded-%d lower bound (never padded or repeated into a '
                        '12-item arm)' % leg['source_budget'])
    else:
        source_label = 'recorded-%d' % leg['source_budget']
    return {
        'role': leg['role'], 'path': leg['path'], 'model': leg['model'], 'mode': mode,
        'n_expert': leg['n_expert'], 'top_k': leg['top_k'], 'retain_k': leg['retain_k'],
        'real_capacity': leg['capacity'],
        'source_budget': leg['source_budget'], 'source_label': source_label,
        'pool_budget': POOL_BUDGET, 'admission_bound': POOL_BUDGET,
        'stream_signature': stream_signature(leg['stages']),
        'demand_rows_signature': demand_rows_signature(leg['stages']),
        'row_bytes_by_layer': leg['row_bytes_by_layer'],
        'row_bytes_distinct': sorted(set(leg['row_bytes_by_layer'].values())),
        'demand_rows': demand_rows, 'demand_ids': demand_ids,
        # real replay
        'real_replay_status': ('parity-ok' if mode == 'overlay'
                               else 'diagnostic (demand-only)'),
        'real_parity_stages': real['parity_stages'],
        'real_divergent_stages': real['divergent_stages'],
        'real_fill_rows_applied': real['fill_rows_applied'],
        'real_hits': real['real_hits'],
        'real_fetched': real['real_fetched'],
        'real_evicted': real['real_evicted'],
        'real_share': _share(real['real_hits'], demand_ids),
        # pool lifecycle counters, printed BEFORE the attractive hit rate
        'pool': dict(totals, lead_hist=dict(sorted(lead_hist.items()))),
        'closures': {
            'per_stage_candidate_identity': 'ok',
            'fate_identity': 'ok',
            'occupancy_bound_le_12': 'ok' if totals['max_occupancy'] <= POOL_BUDGET else 'BROKEN',
            'no_inflight_overwrite': 'ok' if totals['inflight_overwrites'] == 0 else 'BROKEN',
        },
        # outcomes
        'hit_rate_admitted': _share(totals['hits'], totals['admitted']),
        'hit_rate_resolved': _share(totals['hits'], resolved),
        'extra_share': _share(totals['hits'], demand_ids),
        'total_share': _share(real['real_hits'] + totals['hits'], demand_ids),
        'next_demand_fallbacks': totals['pending_fallbacks'],
        'carryover_hits': totals['hits'],
        'hits_lead_2': lead_hist.get(2, 0),
        'hits_lead_ge_3': sum(v for k, v in lead_hist.items() if k >= 3),
        'h2d_bytes': totals['h2d_bytes'],
        'd2d_bytes': totals['d2d_bytes'],
        'net_h2d_bytes': totals['h2d_bytes'] - totals['d2d_bytes'],
        'recorded_spec_totals': leg['spec_totals'],
    }


PAIRED_METRICS = ('pool_hits', 'extra_share', 'pool_admitted', 'pending_fallbacks',
                  'evictions', 'net_h2d_bytes')


def paired_delta(arm, null):
    """One arm vs one null: same ordered demand stream, shape, and source budget,
    or REFUSE. Never pools sums across unequal leg counts."""
    for field in ('model', 'n_expert', 'top_k', 'retain_k', 'source_budget'):
        if arm[field] != null[field]:
            raise Refusal('mismatched arm/null demand streams: %s differs (%r vs %r)'
                          % (field, arm[field], null[field]))
    if arm['demand_rows_signature'] != null['demand_rows_signature']:
        raise Refusal('mismatched arm/null demand streams: demand_rows_signature differs '
                      '(%s vs %s)' % (arm['demand_rows_signature'],
                                      null['demand_rows_signature']))
    values = {
        'pool_hits': ('pool', 'hits'),
        'extra_share': ('extra_share',),
        'pool_admitted': ('pool', 'admitted'),
        'pending_fallbacks': ('pool', 'pending_fallbacks'),
        'evictions': ('pool', 'evictions'),
        'net_h2d_bytes': ('net_h2d_bytes',),
    }
    metrics = {}
    for name in PAIRED_METRICS:
        path = values[name]
        arm_value = arm[path[0]] if len(path) == 1 else arm[path[0]][path[1]]
        null_value = null[path[0]] if len(path) == 1 else null[path[0]][path[1]]
        metrics[name] = {'arm': arm_value, 'null': null_value,
                         'delta_arm_minus_null': arm_value - null_value}
    return {
        'arm_path': arm['path'], 'null_path': null['path'],
        'pair_gate': {
            'demand_rows_signature': arm['demand_rows_signature'],
            'model': arm['model'], 'n_expert': arm['n_expert'], 'top_k': arm['top_k'],
            'retain_k': arm['retain_k'], 'source_budget': arm['source_budget'],
        },
        'metrics': metrics,
    }


def evidence_limitations(leg_summaries, mode):
    out = [
        'Host-only counterfactual under a modeled completion: no CUDA behavior, transfer '
        'latency, or PP speed is established; byte figures are logical payload accounting, '
        'not CUDA allocation (exact allocation needs a future spec_pool_init/byte-account field).',
        'Completion is ideal-next-reap (the measured lead floor), and modeled completion '
        'waits are zero by construction: a PENDING match falls back, it is never waited on.',
        'A pool hit is never promoted into the real cache and never raises real retention '
        'counters; it can only avoid a future demand fetch.',
        'The last proposal of each logical layer has no later occurrence and is unused.',
        'Real-duplicate suppression is evaluated against the real cache at proposal time '
        '(rows of any state present then); rows released by the stage\'s reap do not block.',
        'A positive replay is a reason to authorize a fresh owner-run pool arm, not '
        'acceptance of that arm; nothing here estimates wall-clock speedup or changes '
        'runtime admission or T-GPLR-58\'s completed real-cache evidence.',
    ]
    if any(leg['source_budget'] < POOL_BUDGET for leg in leg_summaries):
        out.append('Source budget is below the 12-item pool contract where marked '
                   '(recorded-8 lower bound): a full 12-item result requires 12 recorded '
                   'candidates or a separately versioned host predictor replay, and a null '
                   'source additionally needs its RNG state or seed to reproduce IDs beyond '
                   'the recorded list. Never padded or repeated.')
    if mode == 'demand-only':
        out.append('demand-only mode is a clean-baseline diagnostic that ignores recorded '
                   'fills; it is never an overlay measurement.')
    if any(len(leg['row_bytes_distinct']) > 1 for leg in leg_summaries):
        out.append('row_bytes differs across layers on this model (bytes_full_bank/n_expert is '
                   'layer-constant but not model-constant); byte metrics are summed per row '
                   'at each row\'s own layer rate.')
    if mode == 'overlay' and not any(leg['real_fill_rows_applied'] for leg in leg_summaries):
        out.append('No recorded real speculative fills were replayed in overlay mode; the '
                   'fill lifecycle and generation transitions are fixture-validated only '
                   '(the recorded traces hold generation/phase constant per layer).')
    if all(leg['pool']['refused_capacity'] == 0 for leg in leg_summaries):
        out.append('Pool capacity refusals are structurally unreachable through the stage '
                   'pipeline under ideal-next-reap (every earlier row is READY at proposal '
                   'time); the all-PENDING refusal is proven by the selftest unit case.')
    return out


def analyze(pairs, mode):
    """pairs: [(role, path)] with explicit roles. Returns the full report."""
    seen = {}
    for role, path in pairs:
        if role not in ROLES:
            raise Refusal('unknown role %r (roles: arm, null, off)' % role)
        if role in ('arm', 'null'):
            if role in seen:
                raise Refusal('role %r declared twice (a session is one arm and one null at '
                              'most)' % role)
            seen[role] = path
    legs = []
    for role, path in pairs:
        leg = load_trace(path, mode)
        leg['role'] = role
        real = replay_real(leg)  # parity gate runs BEFORE the pool result
        pool_out = replay_pool(leg)
        legs.append(summarize_leg(leg, real, pool_out, mode))
    by_role = {leg['role']: leg for leg in legs}
    paired = None
    if 'arm' in by_role and 'null' in by_role:
        paired = paired_delta(by_role['arm'], by_role['null'])
    return {
        'schema': SCHEMA,
        'generated_utc': datetime.datetime.now(datetime.timezone.utc)
                            .strftime('%Y-%m-%dT%H:%M:%SZ'),
        'tool': 'tools/analyze-gplr59-pool.py',
        'mode': mode,
        'completion_assumption': COMPLETION,
        'pool_budget': POOL_BUDGET,
        'admission_bound': POOL_BUDGET,
        'legs': legs,
        'paired_deltas': paired,
        'evidence_limitations': evidence_limitations(legs, mode),
    }
