#include "telemetry-pool.h"

#include "telemetry-json.h"

#include <algorithm>
#include <map>
#include <unordered_set>
#include <utility>

struct ls_pool_validator::model {
    struct stage {
        long long ubatch = -1;
        long long layer = -1;
        long long generation = -1;
        std::vector<long long> candidates;
        std::vector<long long> fills;
        std::vector<long long> hits;
        long long skipped_real = 0, skipped_pool = 0, skipped_dup = 0, skipped_invalid = 0;
        long long refused_budget = 0, refused_capacity = 0, evicted = 0, stop = 0;
        long long pending = 0, ready = 0, submitted = 0;
        long long consumed = 0, pending_fallbacks = 0;
        long long h2d_bytes = 0, d2d_bytes = 0;
        long long submitted_total = 0, consumed_total = 0, evicted_total = 0;
        long long pending_fallbacks_total = 0, h2d_bytes_total = 0, d2d_bytes_total = 0;
        long long prior_submitted_total = 0, prior_consumed_total = 0, prior_evicted_total = 0;
        long long prior_pending_fallbacks_total = 0, prior_h2d_bytes_total = 0, prior_d2d_bytes_total = 0;
        bool fetch_seen = false;
    };

    bool init_seen = false;
    bool init_valid = false;
    long long init_layers = 0, init_capacity = 0, init_budget = 0, init_bytes = 0;
    bool byte_pool_seen = false;
    long long byte_pool = 0;
    bool run_start_seen = false;
    bool run_pool_seen = false;
    long long run_capacity = -1, run_budget = -1, run_bytes = -1, run_mode = -1;
    long long n_expert = -1;
    long long event_count = 0;
    std::map<std::pair<long long, long long>, stage> stages;
    std::map<long long, std::pair<long long, long long>> occupancy;
    bool latest_valid = false;
    long long submitted_total = 0, consumed_total = 0, evicted_total = 0;
    long long pending_fallbacks_total = 0, h2d_bytes_total = 0, d2d_bytes_total = 0;
    long long generation_cancelled_total = 0;
    long long stale_retired_total = 0, allocation_bytes = 0;

    bool is_enabled() const {
        return run_capacity > 0 || init_valid || byte_pool_seen || event_count > 0;
    }
};

ls_pool_validator::ls_pool_validator() : state_(new model()) {}
ls_pool_validator::~ls_pool_validator() = default;

std::vector<std::string> ls_pool_validator::init_errors(const std::string & line) {
    std::vector<std::string> out;
    model & state = *state_;
    if (state.init_seen) {
        out.push_back("spec_pool_init appears more than once");
    }
    state.init_seen = true;
    bool have_layers = false, have_capacity = false, have_budget = false, have_bytes = false;
    const long long layers = ls_json::integer(line, "layers", &have_layers);
    const long long capacity = ls_json::integer(line, "capacity", &have_capacity);
    const long long budget = ls_json::integer(line, "proposal_budget", &have_budget);
    const long long bytes = ls_json::integer(line, "bytes", &have_bytes);
    if (!have_layers || !have_capacity || !have_budget || !have_bytes ||
        layers <= 0 || capacity != 12 || budget != 12 || bytes <= 0) {
        out.push_back("spec_pool_init must declare positive layers, capacity=12, proposal_budget=12, and bytes>0");
        return out;
    }
    state.init_layers = layers;
    state.init_capacity = capacity;
    state.init_budget = budget;
    state.init_bytes = bytes;
    state.init_valid = true;
    return out;
}

std::vector<std::string> ls_pool_validator::byte_account_errors(const std::string & line) {
    std::vector<std::string> out;
    model & state = *state_;
    bool have_pool = false;
    const long long pool = ls_json::integer(line, "prediction_pool", &have_pool);
    if (!have_pool) {
        return out;
    }
    if (state.byte_pool_seen) {
        out.push_back("byte_account carries more than one prediction_pool figure");
    }
    state.byte_pool_seen = true;
    state.byte_pool = pool;
    if (pool <= 0) {
        out.push_back("byte_account: prediction_pool must be positive when present");
    }
    if (state.init_valid && pool != state.init_bytes) {
        out.push_back("byte_account prediction_pool does not match spec_pool_init bytes");
    }
    return out;
}

std::vector<std::string> ls_pool_validator::run_start_errors(const std::string & line) {
    std::vector<std::string> out;
    model & state = *state_;
    state.run_start_seen = true;
    bool have_pool = false, have_bytes = false, have_budget = false, have_mode = false;
    const long long pool = ls_json::integer(line, "row_spec_pool", &have_pool);
    const long long bytes = ls_json::integer(line, "row_spec_pool_bytes", &have_bytes);
    const long long budget = ls_json::integer(line, "row_spec_budget", &have_budget);
    const long long mode = ls_json::integer(line, "row_spec_mode", &have_mode);
    if (!have_pool) {
        if (state.init_seen || state.byte_pool_seen) {
            out.push_back("run_start has prediction-pool records but no row_spec_pool field");
        }
        return out;
    }
    state.run_pool_seen = true;
    state.run_capacity = pool;
    state.run_bytes = have_bytes ? bytes : -1;
    state.run_budget = have_budget ? budget : -1;
    state.run_mode = have_mode ? mode : -1;
    if (pool < 0 || !have_bytes || bytes < 0 || !have_budget || budget < 0 ||
        !have_mode || (mode != 0 && mode != 1 && mode != 2)) {
        out.push_back("run_start has an invalid prediction-pool configuration");
        return out;
    }
    if (pool > 0) {
        if (mode != 2 || budget != pool || pool != 12 || bytes <= 0) {
            out.push_back("run_start prediction pool must be mode 2 with capacity=budget=12 and bytes>0");
        }
        if (!state.init_valid) {
            out.push_back("run_start enables the prediction pool but no valid spec_pool_init preceded it");
        } else if (bytes != state.init_bytes || pool != state.init_capacity || budget != state.init_budget) {
            out.push_back("run_start prediction-pool configuration does not match spec_pool_init");
        }
        if (!state.byte_pool_seen) {
            out.push_back("run_start enables the prediction pool but byte_account omits prediction_pool");
        } else if (state.byte_pool != bytes) {
            out.push_back("run_start prediction-pool bytes do not match byte_account");
        }
    } else {
        if (bytes != 0) {
            out.push_back("run_start row_spec_pool=0 carries nonzero prediction-pool bytes");
        }
        if (state.byte_pool_seen) {
            out.push_back("byte_account reports a prediction pool while run_start declares it off");
        }
    }
    return out;
}

std::vector<std::string> ls_pool_validator::event_errors(const std::string & line) {
    std::vector<std::string> out;
    model & state = *state_;
    const char * required[] = {
        "ubatch", "layer", "generation", "capacity", "proposal_budget", "candidates",
        "fill_ids", "hit_ids", "skipped_real", "skipped_pool", "skipped_dup", "skipped_invalid",
        "refused_budget", "refused_capacity", "evicted", "stop", "pending", "ready",
        "submitted", "submitted_total", "cancelled_total", "evicted_total", "consumed",
        "consumed_total", "pending_fallbacks", "pending_fallbacks_total", "h2d_bytes",
        "h2d_bytes_total", "d2d_bytes", "d2d_bytes_total", "cancelled_demand_total",
        "cancelled_generation_total", "cancelled_capacity_total", "stale_retired_total",
        "allocation_bytes"
    };
    for (const char * key : required) {
        if (line.find(std::string("\"") + key + "\"") == std::string::npos) {
            out.push_back(std::string("spec_pool missing ") + key);
        }
    }
    if (!out.empty()) {
        return out;
    }
    if (state.n_expert <= 0) {
        out.push_back("spec_pool appears before rowset_init");
        return out;
    }

    model::stage current;
    bool numeric_valid = true;
    auto number = [&](const char * key) {
        bool have = false;
        const long long result = ls_json::integer(line, key, &have);
        numeric_valid = numeric_valid && have;
        return result;
    };
    current.ubatch = number("ubatch");
    current.layer = number("layer");
    current.generation = number("generation");
    const long long capacity = number("capacity");
    const long long budget = number("proposal_budget");
    bool have_candidates = false, have_fills = false, have_hits = false;
    current.candidates = ls_json::integer_list(line, "candidates", &have_candidates);
    current.fills = ls_json::integer_list(line, "fill_ids", &have_fills);
    current.hits = ls_json::integer_list(line, "hit_ids", &have_hits);
    current.skipped_real = number("skipped_real");
    current.skipped_pool = number("skipped_pool");
    current.skipped_dup = number("skipped_dup");
    current.skipped_invalid = number("skipped_invalid");
    current.refused_budget = number("refused_budget");
    current.refused_capacity = number("refused_capacity");
    current.evicted = number("evicted");
    current.stop = number("stop");
    current.pending = number("pending");
    current.ready = number("ready");
    current.submitted = number("submitted");
    current.submitted_total = number("submitted_total");
    current.consumed = number("consumed");
    current.consumed_total = number("consumed_total");
    current.evicted_total = number("evicted_total");
    current.pending_fallbacks = number("pending_fallbacks");
    current.pending_fallbacks_total = number("pending_fallbacks_total");
    current.h2d_bytes = number("h2d_bytes");
    current.h2d_bytes_total = number("h2d_bytes_total");
    current.d2d_bytes = number("d2d_bytes");
    current.d2d_bytes_total = number("d2d_bytes_total");
    const long long ready_total = number("ready_total");
    const long long cancelled_total = number("cancelled_total");
    const long long demand_cancelled = number("cancelled_demand_total");
    const long long generation_cancelled = number("cancelled_generation_total");
    const long long capacity_cancelled = number("cancelled_capacity_total");
    const long long stale_retired = number("stale_retired_total");
    const long long reap_total = number("reap_wait_us_total");
    const long long allocation = number("allocation_bytes");
    if (!numeric_valid || !have_candidates || !have_fills || !have_hits || current.ubatch < 1 ||
        current.layer < 0 || current.generation < 0 || capacity != 12 || budget != 12 ||
        current.submitted < 0 || current.consumed < 0 || current.pending_fallbacks < 0 ||
        current.h2d_bytes < 0 || current.d2d_bytes < 0 || current.submitted_total < 0 ||
        current.consumed_total < 0 || current.evicted_total < 0 ||
        current.pending_fallbacks_total < 0 || current.h2d_bytes_total < 0 ||
        current.d2d_bytes_total < 0 || ready_total < 0 || cancelled_total < 0 ||
        demand_cancelled < 0 || generation_cancelled < 0 || capacity_cancelled != 0 ||
        stale_retired < 0 || reap_total < 0 || allocation <= 0) {
        out.push_back("spec_pool has invalid identity, arrays, counters, or fixed 12/12 configuration");
        return out;
    }
    if (state.run_capacity >= 0 &&
        (capacity != state.run_capacity || budget != state.run_budget || allocation != state.run_bytes)) {
        out.push_back("spec_pool configuration or allocation bytes disagree with run_start");
    }
    if (state.init_valid &&
        (capacity != state.init_capacity || budget != state.init_budget || allocation != state.init_bytes)) {
        out.push_back("spec_pool configuration or allocation bytes disagree with spec_pool_init");
    }
    if (current.submitted != (long long) current.fills.size() ||
        current.pending != (long long) current.fills.size() ||
        current.consumed != (long long) current.hits.size() ||
        current.pending + current.ready > capacity || current.evicted > current.submitted) {
        out.push_back("spec_pool per-stage submitted/pending/hit/occupancy accounting mismatch");
    }
    const long long classified = (long long) current.fills.size() + current.skipped_real +
                                 current.skipped_pool + current.skipped_dup +
                                 current.skipped_invalid + current.refused_budget +
                                 current.refused_capacity;
    if (classified != (long long) current.candidates.size()) {
        out.push_back("spec_pool proposal conservation does not close");
    }
    if (current.stop < 0 || current.stop > 2 ||
        (current.stop == 0 && (current.refused_budget != 0 || current.refused_capacity != 0)) ||
        (current.stop == 1 && current.refused_budget == 0) ||
        (current.stop == 2 && current.refused_capacity == 0)) {
        out.push_back("spec_pool stop does not match its refusal class");
    }
    if (state.n_expert > 0) {
        long long invalid = 0;
        for (long long id : current.candidates) {
            invalid += id < 0 || id >= state.n_expert;
        }
        if (invalid != current.skipped_invalid) {
            out.push_back("spec_pool skipped_invalid does not match the candidate id range");
        }
    }
    std::unordered_set<long long> unique_fills;
    for (long long id : current.fills) {
        if (state.n_expert > 0 && (id < 0 || id >= state.n_expert)) {
            out.push_back("spec_pool fill_ids carries an out-of-range expert id");
            break;
        }
        if (!unique_fills.insert(id).second ||
            std::find(current.candidates.begin(), current.candidates.end(), id) == current.candidates.end()) {
            out.push_back("spec_pool fill_ids carries a duplicate or unpublished id");
            break;
        }
    }
    std::unordered_set<long long> unique_hits;
    for (long long id : current.hits) {
        if (state.n_expert > 0 && (id < 0 || id >= state.n_expert)) {
            out.push_back("spec_pool hit_ids carries an out-of-range expert id");
            break;
        }
        if (!unique_hits.insert(id).second || unique_fills.count(id)) {
            out.push_back("spec_pool hit_ids are duplicated or overlap newly admitted fills");
            break;
        }
    }

    const std::pair<long long, long long> key = { current.ubatch, current.layer };
    if (state.stages.count(key)) {
        out.push_back("spec_pool appears more than once for one (ubatch, layer) stage");
    }
    if (state.latest_valid) {
        current.prior_submitted_total = state.submitted_total;
        current.prior_consumed_total = state.consumed_total;
        current.prior_evicted_total = state.evicted_total;
        current.prior_pending_fallbacks_total = state.pending_fallbacks_total;
        current.prior_h2d_bytes_total = state.h2d_bytes_total;
        current.prior_d2d_bytes_total = state.d2d_bytes_total;
    }
    if (current.submitted_total != current.prior_submitted_total + (long long) current.fills.size() ||
        current.consumed_total != current.prior_consumed_total + (long long) current.hits.size() ||
        current.evicted_total != current.prior_evicted_total + current.evicted ||
        current.pending_fallbacks_total != current.prior_pending_fallbacks_total + current.pending_fallbacks ||
        current.h2d_bytes_total != current.prior_h2d_bytes_total + current.h2d_bytes ||
        current.d2d_bytes_total != current.prior_d2d_bytes_total + current.d2d_bytes) {
        out.push_back("spec_pool cumulative counters do not advance exactly by this stage");
    }
    if (demand_cancelled != current.pending_fallbacks_total ||
        cancelled_total != demand_cancelled + generation_cancelled) {
        out.push_back("spec_pool cancellation totals do not close by cause");
    }

    state.stages[key] = current;
    state.occupancy[current.layer] = { current.pending, current.ready };
    ++state.event_count;
    state.latest_valid = true;
    state.submitted_total = current.submitted_total;
    state.consumed_total = current.consumed_total;
    state.evicted_total = current.evicted_total;
    state.pending_fallbacks_total = current.pending_fallbacks_total;
    state.h2d_bytes_total = current.h2d_bytes_total;
    state.d2d_bytes_total = current.d2d_bytes_total;
    state.generation_cancelled_total = generation_cancelled;
    state.stale_retired_total = stale_retired;
    state.allocation_bytes = allocation;
    return out;
}

std::vector<std::string> ls_pool_validator::fetch_errors(
        const std::string & line, long long layer, long long generation,
        long long ubatch, long long route_union, long long fetched,
        long long resident, long long retained, long long bytes_staged,
        long long bytes_full_bank) {
    std::vector<std::string> out;
    model & state = *state_;
    bool have_pool_ids = false;
    const std::vector<long long> pool_ids = ls_json::integer_list(line, "pool_hit_ids", &have_pool_ids);
    if (!state.is_enabled()) {
        if (have_pool_ids) {
            out.push_back("rowset_fetch carries pool_hit_ids while the prediction pool is off");
        }
        return out;
    }
    if (!have_pool_ids) {
        out.push_back("rowset_fetch omits pool_hit_ids in a prediction-pool trace");
        return out;
    }
    const std::vector<long long> union_ids = ls_json::id_list(ls_json::value(line, "union_ids"));
    if ((long long) union_ids.size() != route_union) {
        out.push_back("rowset_fetch union_ids length does not match route_union");
        return out;
    }
    std::unordered_set<long long> unique_pool;
    for (long long id : pool_ids) {
        if (state.n_expert > 0 && (id < 0 || id >= state.n_expert)) {
            out.push_back("rowset_fetch pool_hit_ids carries an out-of-range expert id");
            break;
        }
        if (!unique_pool.insert(id).second ||
            std::find(union_ids.begin(), union_ids.end(), id) == union_ids.end()) {
            out.push_back("rowset_fetch pool_hit_ids are duplicated or outside union_ids");
            break;
        }
    }
    const auto found = state.stages.find({ ubatch, layer });
    if (found == state.stages.end()) {
        out.push_back("rowset_fetch has no spec_pool event for its (ubatch, layer) stage");
        return out;
    }
    model::stage & current = found->second;
    if (current.fetch_seen) {
        out.push_back("rowset_fetch appears more than once for one pool stage");
    }
    current.fetch_seen = true;
    if (current.layer != layer || current.generation != generation) {
        out.push_back("rowset_fetch and spec_pool disagree on layer/generation");
    }
    if (pool_ids != current.hits) {
        out.push_back("rowset_fetch pool_hit_ids do not match spec_pool hit_ids");
    }
    if (route_union != fetched + resident + (long long) pool_ids.size()) {
        out.push_back("rowset_fetch route_union != real fetched + real resident + pool hits");
    }
    if (retained != resident) {
        out.push_back("rowset_fetch n_rows_retained must equal the real-cache resident count");
    }
    if ((bytes_staged > 0) != (fetched > 0)) {
        out.push_back("rowset_fetch prediction-pool D2D bytes leak into bytes_staged");
    }
    if (state.n_expert > 0 && bytes_full_bank > 0) {
        if (bytes_full_bank % state.n_expert != 0) {
            out.push_back("rowset_fetch bytes_full_bank is not an integral expert-slab size");
        } else {
            const long long row_bytes = bytes_full_bank / state.n_expert;
            if (current.h2d_bytes != row_bytes * (long long) current.fills.size() ||
                current.d2d_bytes != row_bytes * (long long) current.hits.size()) {
                out.push_back("spec_pool H2D/D2D bytes do not match admitted/hit row bytes");
            }
        }
    }
    return out;
}

std::vector<std::string> ls_pool_validator::final_errors(bool refusal) const {
    std::vector<std::string> out;
    const model & state = *state_;
    if (refusal) {
        if (state.event_count > 0) {
            out.push_back("refusal trace contains prediction-pool demand events");
        }
        return out;
    }
    if (!state.is_enabled()) {
        return out;
    }
    if (!state.run_start_seen || !state.run_pool_seen || state.run_capacity <= 0) {
        out.push_back("prediction-pool records are not declared by a positive run_start row_spec_pool");
    }
    if (!state.init_valid) {
        out.push_back("prediction-pool trace has no valid spec_pool_init");
    }
    if (!state.byte_pool_seen) {
        out.push_back("prediction-pool trace has no byte_account.prediction_pool field");
    }
    if (state.event_count == 0) {
        out.push_back("prediction-pool trace has no spec_pool demand-stage events");
    }
    for (const auto & entry : state.stages) {
        if (!entry.second.fetch_seen) {
            out.push_back("prediction-pool stage has spec_pool but no paired rowset_fetch");
        }
    }
    if (state.latest_valid) {
        long long live = 0;
        for (const auto & row : state.occupancy) {
            live += row.second.first + row.second.second;
        }
        const long long terminal = state.consumed_total + state.pending_fallbacks_total +
                                   state.generation_cancelled_total + state.evicted_total +
                                   state.stale_retired_total;
        if (state.submitted_total != terminal + live) {
            out.push_back("prediction-pool submitted total does not close against terminal and live rows");
        }
        if (state.pending_fallbacks_total > state.submitted_total ||
            state.generation_cancelled_total > state.submitted_total ||
            state.evicted_total > state.submitted_total ||
            state.stale_retired_total > state.submitted_total ||
            state.consumed_total > state.submitted_total) {
            out.push_back("prediction-pool terminal counter exceeds submitted rows");
        }
    }
    return out;
}

bool ls_pool_validator::enabled() const {
    return state_->is_enabled();
}

void ls_pool_validator::set_n_expert(long long n_expert) {
    state_->n_expert = n_expert;
}
