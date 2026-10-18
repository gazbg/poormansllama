// Host-only test for the G22 CPU-leg plumbing of the layer-major prefill (--pp-cpu-layers).
//
// WHY THIS EXISTS. A layer on the CPU leg has no slot, no copy and no consumer event: the graph of such a layer is built against the model's own host layer
// tensors, and the async feed must skip it when it looks for the next layer to stage. If slot_layer() routed a CPU-leg layer to a slot bundle, the layer would
// compute with whatever another layer left in that slot (a silently wrong answer); if the feed offered a CPU-leg layer to the copy worker, the worker would copy a
// bank nobody consumes over a slot that is still in use. Neither shows in a speed number, so the routing is checked here without a GPU or a model.
//
// NO GPU, NO MODEL. Everything runs through the production symbols llama_layer_streaming::slot_layer, ::af_next_layer, ::set_pp_cpu_layers, ::is_cpu_leg and
// ::has_cpu_leg.

#include "../src/llama-layer-streaming.h"

#include <cstdio>
#include <vector>

static int failures = 0;

static void expect(bool cond, const char * what) {
    if (cond) {
        fprintf(stderr, "ok   %s\n", what);
    } else {
        fprintf(stderr, "FAIL %s\n", what);
        failures++;
    }
}

int main() {
    typedef llama_layer_streaming LS;

    LS ls;
    ls.n_layer = 8;
    ls.n_slots = 2;
    std::vector<llama_layer> host(8);
    for (int il = 0; il < 8; ++il) {
        ls.host_layers.push_back(&host[il]);
    }
    ls.slots.resize(2);
    for (int il = 0; il < 8; ++il) {
        ls.slots[il % 2].layers[il] = llama_layer();
    }

    fprintf(stderr, "\n== default: nothing on the CPU leg ==\n");
    expect(!ls.has_cpu_leg(), "no CPU-leg layer by default");
    expect(!ls.is_cpu_leg(3), "layer 3 streams by default");
    expect(&ls.slot_layer(3) == &ls.slots[1].layers.at(3), "slot_layer(3) is the bundle of slot 1");
    expect(ls.next_streamed_layer(2) == 3, "the feed offers layer 3 after layer 2");

    fprintf(stderr, "\n== set_pp_cpu_layers {1,3,4} ==\n");
    expect(ls.set_pp_cpu_layers({1, 3, 4}), "a valid list is accepted");
    expect(ls.has_cpu_leg(), "has_cpu_leg");
    expect(!ls.is_cpu_leg(0) && ls.is_cpu_leg(1) && !ls.is_cpu_leg(2) && ls.is_cpu_leg(3) && ls.is_cpu_leg(4) && !ls.is_cpu_leg(5), "is_cpu_leg marks exactly 1, 3, 4");
    expect(!ls.is_cpu_leg(-1) && !ls.is_cpu_leg(8) && !ls.is_cpu_leg(100), "out-of-range layers are never CPU-leg");

    fprintf(stderr, "\n== routing: prefill (phase 0) reads host tensors for the CPU-leg layers only ==\n");
    ls.phase = 0;
    expect(&ls.slot_layer(1) == &host[1], "layer 1 reads the host layer");
    expect(&ls.slot_layer(3) == &host[3], "layer 3 reads the host layer");
    expect(&ls.slot_layer(4) == &host[4], "layer 4 reads the host layer");
    expect(&ls.slot_layer(0) == &ls.slots[0].layers.at(0), "layer 0 still reads its slot bundle");
    expect(&ls.slot_layer(2) == &ls.slots[0].layers.at(2), "layer 2 still reads its slot bundle");
    expect(&ls.slot_layer(5) == &ls.slots[1].layers.at(5), "layer 5 still reads its slot bundle");

    fprintf(stderr, "\n== routing: decode (phase 1) is untouched ==\n");
    ls.phase = 1;
    expect(&ls.slot_layer(1) == &ls.slots[1].layers.at(1), "layer 1 reads its slot bundle in phase 1");
    expect(&ls.slot_layer(4) == &ls.slots[0].layers.at(4), "layer 4 reads its slot bundle in phase 1");
    ls.phase = 0;

    fprintf(stderr, "\n== the feed skips CPU-leg layers ==\n");
    expect(ls.next_streamed_layer(0) == 2, "after layer 0 the feed offers layer 2 (1 is CPU)");
    expect(ls.next_streamed_layer(2) == 5, "after layer 2 the feed offers layer 5 (3 and 4 are CPU)");
    expect(ls.next_streamed_layer(5) == 6, "after layer 5 the feed offers layer 6");
    expect(ls.next_streamed_layer(7) == -1, "after the last layer there is nothing to offer");
    ls.exec_layers = {0, 1, 2, 3, 4, 5};
    expect(ls.next_streamed_layer(2) == 5, "with a layer subset the feed still skips 3 and 4");
    expect(ls.next_streamed_layer(5) == -1, "and ends at the end of the subset");
    ls.exec_layers.clear();

    fprintf(stderr, "\n== refusals ==\n");
    expect(!ls.set_pp_cpu_layers({2, 8}), "a layer beyond the last is refused");
    expect(!ls.has_cpu_leg(), "a refused list leaves no CPU-leg layer behind");
    expect(!ls.set_pp_cpu_layers({-1}), "a negative layer is refused");
    expect(ls.set_pp_cpu_layers({}), "an empty list is the all-GPU configuration");
    expect(!ls.has_cpu_leg() && ls.next_streamed_layer(0) == 1, "and streams everything again");

    fprintf(stderr, "\n%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
