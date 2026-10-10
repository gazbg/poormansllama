#!/usr/bin/env python3
"""Per-layer routed-expert bank bytes and stock placement, from the GGUF header.

Host-only.  sec 12.5 prices stock prefill as bytes copied per ubatch, so it
needs two facts nobody had printed:

  1. Banks are NOT uniform.  AGENT_ENVIRONMENT.md records "heavy layers
     36.3 ms = exactly the byte ratio"; this lists which layers and how heavy.
  2. WHICH layers stock leaves on the host.  Given the fitter's logged
     decision it predicts the CUDA0 model buffer AND the KV / recurrent-state
     layer split, three numbers the server logs independently, and it
     itemises what a ubatch must copy.

The placement is computed TWICE from the same three logged numbers
("set ngl_per_device[0].(n_layer, n_part, overflow_type)"), because the two
pieces of code that decide it disagree on how many blocks the model has:

  fitter  hp_ngl = llama_model_n_layer(), the graph layers; the nextn/MTP
          block is added only when MTP is loaded (common/fit.cpp:139-142).
          Overrides start at il0 = hp_ngl + 1 - n_gpu_layers, then n_full
          whole layers, then n_part partial ones (fit.cpp:474-494).  This is
          what the fitter INTENDS.
  loader  i_gpu_start = n_layer_all + 1 - n_gpu_layers, nextn block INCLUDED
          (src/llama-model.cpp:1354).  Blocks below it stay on the CPU whole:
          dense weights, KV and recurrent state.  This is what RUNS.

Overflow patterns are fit.cpp:407-444 verbatim, matched with regex_search as
the loader matches them (src/llama-model-loader.cpp:1182), so GATE's
"ffn_down.*" also takes the shared expert's down projection.  The output layer
goes to the GPU whenever n_gpu_layers >= 1 (llama-model.cpp:1379); token_embd
is the input layer and always stays on the CPU.

Stdlib only: the header is parsed with struct, and the quant size table is
loaded straight from the vendored gguf-py constants.py (whose package
__init__ needs numpy, which this environment does not have).
"""

import argparse
import importlib.util
import os
import re
import struct
import sys
from collections import defaultdict

_spec = importlib.util.spec_from_file_location("gguf_constants", os.path.join(
    os.path.dirname(__file__), "..", "third_party", "llama.cpp", "gguf-py",
    "gguf", "constants.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

MIB = 1024.0 * 1024.0
BLK = re.compile(r"^blk\.(\d+)\.(.+)$")
EXPS = re.compile(r"^ffn_(up|gate|down)_exps\.weight$")
# GGUF metadata value types -> struct format; 8 = string, 9 = array.
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
          10: "Q", 11: "q", 12: "d"}
# common_layer_fraction_t (fit.cpp:18-22) -> get_overflow_pattern (fit.cpp:407-444)
OVERFLOW = {
    1: ("ATTN", r"blk\.%d\.ffn_(gate|up|gate_up|down).*"),
    2: ("UP",   r"blk\.%d\.ffn_(gate|gate_up|down).*"),
    3: ("GATE", r"blk\.%d\.ffn_down.*"),
    4: ("MOE",  r"blk\.%d\.ffn_(up|down|gate_up|gate)_(ch|)exps"),
}


def read(f, fmt):
    size = struct.calcsize("<" + fmt)
    buf = f.read(size)
    if len(buf) != size:
        sys.exit("truncated GGUF header")
    return struct.unpack("<" + fmt, buf)[0]


def string(f):
    return f.read(read(f, "Q")).decode("utf-8", "replace")


def value(f, vtype):
    """Scalar or string value; arrays are skipped and read as None."""
    if vtype in SCALAR:
        return read(f, SCALAR[vtype])
    if vtype == 8:
        return string(f)
    if vtype == 9:
        etype, n = read(f, "I"), read(f, "Q")
        for _ in range(n):
            value(f, etype)
        return None
    sys.exit("unknown GGUF value type %d" % vtype)


def gguf(path):
    """(metadata, {tensor name: (bytes, quant type name)})."""
    kv, all_t = {}, {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            sys.exit("%s is not a GGUF file" % path)
        version, n_tensors, n_kv = read(f, "I"), read(f, "Q"), read(f, "Q")
        if version < 2:
            sys.exit("GGUF version %d unsupported" % version)
        for _ in range(n_kv):
            key = string(f)
            kv[key] = value(f, read(f, "I"))
        for _ in range(n_tensors):
            name = string(f)
            ne = [read(f, "Q") for _ in range(read(f, "I"))]
            qtype = C.GGMLQuantizationType(read(f, "I"))
            read(f, "Q")                                   # data offset
            block, type_size = C.GGML_QUANT_SIZES[qtype]
            if ne[0] % block:
                sys.exit("%s: ne0 %d not a multiple of block %d" % (name, ne[0], block))
            n = ne[0] // block * type_size
            for d in ne[1:]:
                n *= d
            all_t[name] = (n, qtype.name)
    return kv, all_t


def banks_table(all_t):
    banks = defaultdict(dict)
    for name, (n, qname) in all_t.items():
        m = BLK.match(name)
        e = EXPS.match(m.group(2)) if m else None
        if e:
            banks[int(m.group(1))][e.group(1)] = (n, qname)
    if not banks:
        sys.exit("no ffn_*_exps tensors")
    for layer in sorted(banks):
        if len(banks[layer]) != 3:
            sys.exit("layer %d has %s, not up/gate/down" % (layer, sorted(banks[layer])))
    print("%5s %12s %12s %12s %13s  types" % ("layer", "up", "gate", "down", "bank"))
    for layer in sorted(banks):
        b = banks[layer]
        print("%5d %12d %12d %12d %13d  %s/%s/%s" % (
            layer, b["up"][0], b["gate"][0], b["down"][0],
            sum(v[0] for v in b.values()), b["up"][1], b["gate"][1], b["down"][1]))
    sizes = [sum(v[0] for v in banks[layer].values()) for layer in sorted(banks)]
    print("layers %d, bank min %d, max %d, total %d B"
          % (len(sizes), min(sizes), max(sizes), sum(sizes)))


def placement(all_t, label, gpu_from, part_from, n_part, otype, n_layer, interval):
    overrides = [re.compile(OVERFLOW[otype if il == part_from else 4][1] % il)
                 for il in range(part_from, part_from + n_part)]
    gpu, host_exps, host_dense, host_named = 0, 0, 0, []
    inp = all_t["token_embd.weight"][0]
    for name, (n, _) in all_t.items():
        m = BLK.match(name)
        if m is None:
            if name != "token_embd.weight":
                gpu += n
            continue
        il = int(m.group(1))
        if il >= n_layer:
            continue                        # nextn block, not loaded
        if il < gpu_from or any(p.search(name) for p in overrides):
            if EXPS.match(m.group(2)):
                host_exps += n
            else:
                host_dense += n
                if il >= gpu_from:
                    host_named.append(name)
        else:
            gpu += n
    cpu = range(min(gpu_from, n_layer))
    attn_cpu = sum(1 for il in cpu if (il + 1) % interval == 0)
    attn_all = sum(1 for il in range(n_layer) if (il + 1) % interval == 0)
    print("\n%s: blocks < %d on CPU whole, partial %d..%d (first %s)"
          % (label, gpu_from, part_from, part_from + n_part - 1, OVERFLOW[otype][0]))
    print("   predicted CUDA0 model buffer  %10.2f MiB" % (gpu / MIB))
    print("   host routed experts           %10.2f MiB  %14d B" % (host_exps / MIB, host_exps))
    print("   host non-expert weights       %10.2f MiB  %14d B" % (host_dense / MIB, host_dense))
    print("      whole blocks %s%s" % (list(cpu) or "none",
                                       "".join("\n      + " + t for t in host_named)))
    print("   host total with token_embd    %10.2f MiB  (the host model buffer"
          % ((host_exps + host_dense + inp) / MIB))
    print("      when not mmapped; mmap maps whole file ranges instead)")
    print("   KV layers CPU / GPU           %6d / %d" % (attn_cpu, attn_all - attn_cpu))
    print("   recurrent-state layers CPU/GPU %5d / %d"
          % (len(cpu) - attn_cpu, n_layer - attn_all - (len(cpu) - attn_cpu)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ngl", type=int, help="n_layer the fitter logged")
    ap.add_argument("--n-part", type=int, help="n_part the fitter logged")
    ap.add_argument("--overflow-type", type=int, choices=sorted(OVERFLOW), default=4,
                    help="overflow_type the fitter logged (1 ATTN .. 4 MOE)")
    a = ap.parse_args()

    kv, all_t = gguf(a.model)
    arch = kv["general.architecture"]
    n_layer_all = kv["%s.block_count" % arch]
    n_layer = n_layer_all - kv.get("%s.nextn_predict_layers" % arch, 0)
    interval = kv.get("%s.full_attention_interval" % arch)
    banks_table(all_t)
    if a.ngl is None or a.n_part is None:
        return
    if interval is None:
        sys.exit("%s has no full_attention_interval; the KV split is undefined" % arch)
    hp_ngl = n_layer                        # MTP not loaded
    if not 0 <= a.n_part <= a.ngl <= hp_ngl + 1:
        sys.exit("need 0 <= n_part <= ngl <= %d (fit.cpp:473)" % (hp_ngl + 1))
    il0 = hp_ngl + 1 - a.ngl
    part_from = il0 + a.ngl - a.n_part
    i_gpu_start = max(n_layer_all + 1 - a.ngl, 0)
    print("\n%s: block_count %d, nextn %d -> graph layers %d, attention every %d"
          % (arch, n_layer_all, n_layer_all - n_layer, n_layer, interval))
    print("fitter hp_ngl %d -> first GPU block %d; loader n_layer_all %d -> %d"
          % (hp_ngl, il0, n_layer_all, i_gpu_start))
    placement(all_t, "FITTER INTENT", il0, part_from, a.n_part, a.overflow_type,
              n_layer, interval)
    placement(all_t, "LOADER ACTUAL", i_gpu_start, part_from, a.n_part,
              a.overflow_type, n_layer, interval)
    lost = list(range(il0, min(i_gpu_start, n_layer)))
    if lost:
        print("\nblocks the fitter placed on the GPU that run on the CPU: %s" % lost)


if __name__ == "__main__":
    main()
