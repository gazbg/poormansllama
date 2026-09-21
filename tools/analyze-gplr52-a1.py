import io, json, collections, statistics, re

def load(p):
    with io.open(p, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line:
                try: yield json.loads(line)
                except Exception: pass

def split_passes(evs):
    passes, prev = [[]], None
    for e in evs:
        if e['layer'] == 0 and prev is not None and prev != 0:
            passes.append([])
        passes[-1].append(e)
        prev = e['layer']
    return passes

def pearson(xs, ys):
    n = len(xs)
    if n < 3: return float('nan')
    mx, my = sum(xs)/n, sum(ys)/n
    sx = sum((x-mx)**2 for x in xs) ** 0.5
    sy = sum((y-my)**2 for y in ys) ** 0.5
    if sx == 0 or sy == 0: return float('nan')
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / (sx*sy)

def quant(v, q):
    s = sorted(v)
    return s[min(len(s)-1, int(len(s)*q))]

TRACES = [
    ('4k/512  ', 'gplr50-4k-f16-default'),
    ('50k/512 ', 'gplr50-50k-f16-default'),
    ('144k/q8 ', 'gplr50-b1-q8-default'),
    ('200k/512', 'gplr50-200k-on'),
    ('200k pin', 'gplr51-a2-pinned'),
]

print('%-9s | %-7s | %-28s | %-9s | %-8s | %-6s | %s' % (
    'shape', 'pass', 'copy p50/p90/max (ms)', 'p50 sub', 'p50 sync', 'conc', 'r(wait, layer)'))
print('-' * 108)
rows = []
for label, name in TRACES:
    try:
        evs = list(load('build/llama-cuda/%s.jsonl' % name))
    except FileNotFoundError:
        print('%-9s  MISSING' % label); continue
    rs = [e for e in evs if e.get('event') == 'run_start']
    cps = split_passes([e for e in evs if e.get('event') == 'copy_complete' and e.get('phase') == 0])
    lc  = split_passes([e for e in evs if e.get('event') == 'layer_compute' and e.get('phase') == 0])
    phs = [e for e in evs if e.get('event') == 'phase_summary' and e.get('phase') == 'pp']
    for pi, cp in enumerate(cps):
        tr = [e['transfer_us'] for e in cp]
        sub = [e.get('host_submit_us', 0) for e in cp]
        syn = [e.get('ready_latency_us', 0) for e in cp]
        # copy intervals [t - transfer, t]; pairwise concurrency
        ivs = sorted((e['t_us'] - e['transfer_us'], e['t_us']) for e in cp)
        conc = []
        for i, (a0, a1) in enumerate(ivs):
            c = sum(1 for b0, b1 in ivs if b0 < a1 and a0 < b1) - 1
            conc.append(c)
        # per-layer wait vs layer index correlation (position drift)
        r_layer = pearson([e['layer'] for e in cp], tr)
        # per-layer wait vs that layer's compute median (same pass where possible)
        r_comp = float('nan')
        if pi < len(lc):
            comp_med = collections.defaultdict(list)
            for e in lc[pi]:
                comp_med[e['layer']].append(e['compute_us'])
            med = {l: statistics.median(v) for l, v in comp_med.items()}
            common = [e for e in cp if e['layer'] in med]
            if len(common) >= 10:
                r_comp = pearson([med[e['layer']] for e in common], [e['transfer_us'] for e in common])
        comp_s = phs[pi]['total_compute_us']/1e6 if pi < len(phs) else float('nan')
        nch = len(set(e['chunk'] for e in lc[pi])) if pi < len(lc) else float('nan')
        chcomp = comp_s/nch if nch else float('nan')
        row = dict(shape=label, p=pi+1, p50=quant(tr,.5)/1e3, p90=quant(tr,.9)/1e3, mx=max(tr)/1e3,
                   sub=quant(sub,.5)/1e3, syn=quant(syn,.5)/1e3,
                   conc=statistics.median(conc), cmax=max(conc),
                   r_layer=r_layer, r_comp=r_comp, chcomp=chcomp,
                   ratio=quant(tr,.5)/1e3/(chcomp*1e3) if chcomp else float('nan'))
        rows.append(row)
        print('%-9s | %-7d | %7.0f /%8.0f /%7.0f | %7.1f | %6.1f | %d/%d | r_l=%.2f r_c=%.2f' % (
            label, pi+1, row['p50'], row['p90'], row['mx'], row['sub'], row['syn'],
            row['conc'], row['cmax'], r_layer, r_comp))

print()
print('%-9s | %-4s | %-12s | %-10s | %s' % ('shape', 'pass', 'chunk comp s', 'p50/comp', 'note'))
print('-' * 80)
for r in rows:
    print('%-9s | %-4d | %12.2f | %10.2f |' % (r['shape'], r['p'], r['chcomp'], r['ratio']))
