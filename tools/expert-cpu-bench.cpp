// expert-cpu-bench.cpp - HOST-ONLY microbenchmark for the per-expert MoE FFN cost.
//
// Purpose. The T-GPLR-16 record carries a DERIVED per-expert CPU number
// (~54 us, from a difference of two means in one -ncmoe 20 A/B) and the
// 2026-09-10 record carries a derived ~35-40 GB/s DDR closure. Both feed the
// T-GPLR-61 cost gate. This tool replaces the derivation with a direct
// measurement of the same quantity, on the real quantized bytes, with no GPU,
// no model load, no VRAM and no shared-GPU mutex.
//
// It links ggml + ggml-cpu ONLY. It never links or loads ggml-cuda, so it
// cannot initialize a CUDA context even by accident.
//
// What is measured, per cell (threads x experts_used x n_tokens):
//   the exact op chain build_moe_ffn runs for one TG layer -
//     mul_mat_id(gate_exps, x, ids) -> mul_mat_id(up_exps, x, ids)
//     -> swiglu_split(gate, up) -> mul_mat_id(down_exps, act, ids)
//     -> mul by router weight -> add chain over the expert axis
//   with the weights pointing at the real Q4_K/Q6_K slabs of the real GGUF,
//   resident in ordinary host RAM, cycling over a pool of experts large
//   enough that the working set cannot sit in L3.
//
// Separately it measures the FEED (page-in) cost of the same bytes from the
// mmap, which is the uncontrolled-Windows-file-cache quantity and is reported
// as such, never as a cold-cache guarantee.
//
// Numbers this tool does NOT produce: tok/s, PP or TG speed, any equivalence
// or quality claim, and any statement about the PCIe path. It measures one
// quantity. Comparisons to recorded link rates are arithmetic on constants
// measured elsewhere, and are labelled as such in the output.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX          // windows.h otherwise defines min/max macros
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

// ---------------------------------------------------------------------------
// recorded constants from elsewhere in this repo, used ONLY to print a
// comparison column. Never used in a measurement.
// ---------------------------------------------------------------------------
namespace ref {
    // 2026-09-13 TG row-set analysis: rowset_fetch transfer_us 1,835-2,273 us
    // for the ~14.15 MB 8-row union, 6.5-8.2 GB/s effective pageable host rate.
    constexpr double LINK_PAGEABLE_LO_GBPS = 6.5;
    constexpr double LINK_PAGEABLE_HI_GBPS = 8.2;
    // 200k recon: copy_gpu_ms p50 31.5 ms ~= 14.6 GB/s over PCIe (registered).
    constexpr double LINK_REGISTERED_GBPS  = 14.6;
    constexpr double ROW_BYTES             = 1769472.0;   // one expert, q4_K layer
    constexpr double N_LAYERS              = 40.0;
}

// ---------------------------------------------------------------------------
// minimal GGUF v3 reader - header, KV, tensor infos. No tensor data is read
// here; the caller mmaps the file for that.
// ---------------------------------------------------------------------------
struct tensor_info {
    std::string          name;
    uint32_t             n_dims = 0;
    uint64_t             dims[4] = {1, 1, 1, 1};
    uint32_t             type  = 0;
    uint64_t             offset = 0;   // relative to the data section
};

struct gguf_meta {
    uint32_t            version   = 0;
    uint64_t            alignment = 32;
    uint64_t            data_offset = 0;
    uint64_t            n_tensors = 0;
    std::vector<tensor_info> tensors;

    // scalar metadata we care about (0 when absent)
    uint64_t n_expert = 0, n_expert_used = 0, n_embd = 0, n_layer = 0;
    std::string arch;
};

struct reader {
    const uint8_t * p   = nullptr;
    size_t          n   = 0;
    size_t          off = 0;
    bool            bad = false;

    void need(size_t k) { if (off + k > n) { bad = true; off = n; } }

    template <typename T> T pod() {
        need(sizeof(T));
        T v{};
        if (!bad) { memcpy(&v, p + off, sizeof(T)); off += sizeof(T); }
        return v;
    }

    std::string str() {
        const uint64_t len = pod<uint64_t>();
        if (bad || len > (1ull << 30)) { bad = true; return {}; }
        need((size_t) len);
        std::string s((const char *) p + off, (size_t) len);
        off += (size_t) len;
        return s;
    }

    void skip_value(uint32_t type);
};

void reader::skip_value(uint32_t type) {
    switch (type) {
        case 0: case 1:  pod<uint8_t>();  break;   // u8, i8
        case 2: case 3:  pod<uint16_t>(); break;   // u16, i16
        case 4: case 5:  pod<uint32_t>(); break;   // u32, i32
        case 6:          pod<float>();    break;   // f32
        case 7:          pod<uint8_t>();  break;   // bool
        case 10:         pod<uint64_t>(); break;   // u64
        case 11:         pod<int64_t>();  break;   // i64
        case 12:         pod<double>();   break;   // f64
        case 8:          (void) str();    break;   // string
        case 9: {                                     // array
            const uint32_t et = pod<uint32_t>();
            const uint64_t cnt = pod<uint64_t>();
            if (et == 8) { for (uint64_t i = 0; i < cnt && !bad; ++i) (void) str(); }
            else if (et == 9) { bad = true; break; }
            else {
                // fixed-size element: re-derive the width and skip in bulk
                uint32_t w = 0;
                switch (et) {
                    case 0: case 1: case 7: w = 1; break;
                    case 2: case 3:         w = 2; break;
                    case 4: case 5: case 6: w = 4; break;
                    case 10: case 11: case 12: w = 8; break;
                    default: bad = true;
                }
                if (bad) break;
                need((size_t) (cnt * w));
                off += (size_t) (cnt * w);
            }
            break;
        }
        default: bad = true;
    }
}

static bool gguf_read(const uint8_t * base, size_t n, gguf_meta & out) {
    reader r{ base, n, 0, false };
    if (r.pod<uint32_t>() != 0x46554747u /* 'GGUF' */) return false;
    out.version = r.pod<uint32_t>();
    out.n_tensors = r.pod<uint64_t>();
    const uint64_t n_kv = r.pod<uint64_t>();
    if (r.bad) return false;

    for (uint64_t i = 0; i < n_kv && !r.bad; ++i) {
        const std::string key = r.str();
        const uint32_t vt    = r.pod<uint32_t>();
        const size_t   start = r.off;
        r.skip_value(vt);
        if (r.bad) return false;

        auto as_u64 = [&]() -> uint64_t {
            reader v{ base, n, start, false };
            switch (vt) {
                case 0: case 1: case 7: return v.pod<uint8_t>();
                case 2: case 3:         return v.pod<uint16_t>();
                case 4: case 5:         return v.pod<uint32_t>();
                case 10: case 11:       return v.pod<uint64_t>();
                default: return 0;
            }
        };
        if (key == "general.architecture")        { reader v{ base, n, start, false }; out.arch = v.str(); }
        else if (key == "general.alignment")      { out.alignment = as_u64(); }
        else if (key == "qwen35moe.expert_count")      { out.n_expert      = as_u64(); }
        else if (key == "qwen35moe.expert_used_count") { out.n_expert_used = as_u64(); }
        else if (key == "qwen35moe.embedding_length")  { out.n_embd        = as_u64(); }
        else if (key == "qwen35moe.block_count")       { out.n_layer       = as_u64(); }
    }

    out.tensors.resize((size_t) out.n_tensors);
    for (uint64_t i = 0; i < out.n_tensors && !r.bad; ++i) {
        tensor_info & t = out.tensors[(size_t) i];
        t.name = r.str();
        t.n_dims = r.pod<uint32_t>();
        if (t.n_dims > 4) return false;
        for (uint32_t d = 0; d < t.n_dims; ++d) t.dims[d] = r.pod<uint64_t>();
        t.type   = r.pod<uint32_t>();
        t.offset = r.pod<uint64_t>();
    }
    if (r.bad) return false;

    size_t data = r.off;
    const size_t a = (size_t) out.alignment;
    if (a && (data % a)) data += a - (data % a);
    out.data_offset = data;
    return true;
}

static const tensor_info * find_tensor(const gguf_meta & m, const std::string & name) {
    for (const tensor_info & t : m.tensors) if (t.name == name) return &t;
    return nullptr;
}

// ---------------------------------------------------------------------------
// read-only file mapping
// ---------------------------------------------------------------------------
struct mapping {
    void * p = nullptr;
    size_t n = 0;
#ifdef _WIN32
    HANDLE fh = INVALID_HANDLE_VALUE;
    HANDLE mh = nullptr;
    bool open(const std::string & path) {
        fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (fh == INVALID_HANDLE_VALUE) return false;
        LARGE_INTEGER li; if (!GetFileSizeEx(fh, &li)) return false;
        n = (size_t) li.QuadPart;
        mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mh) return false;
        p = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
        return p != nullptr;
    }
    void close() {
        if (p) UnmapViewOfFile(p);
        if (mh) CloseHandle(mh);
        if (fh != INVALID_HANDLE_VALUE) CloseHandle(fh);
        p = nullptr; mh = nullptr; fh = INVALID_HANDLE_VALUE;
    }
#else
    bool open(const std::string & path) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st{}; if (fstat(fd, &st) != 0) { ::close(fd); return false; }
        n = (size_t) st.st_size;
        p = mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        return p != nullptr && p != MAP_FAILED;
    }
    void close() { if (p) munmap(p, n); p = nullptr; }
#endif
};

// ---------------------------------------------------------------------------
// timing helpers
// ---------------------------------------------------------------------------
static double now_us() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::micro>(clock::now().time_since_epoch()).count();
}

static double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = p * (double) (v.size() - 1);
    const size_t lo = (size_t) std::floor(idx);
    const size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (idx - (double) lo);
}

struct cell_result {
    int    threads = 0;
    int    experts_used = 0;
    int    n_tokens = 0;
    double med_us = 0, p90_us = 0, min_us = 0;
    double bytes = 0, gbps = 0;
};

// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    std::string model;
    int    layer        = 0;
    int    iters        = 200;
    int    warmup       = 20;
    int    pool_experts = 64;
    int    id_stride    = 1;
    std::vector<int> thread_list = {1, 2, 4, 8, 16};
    std::vector<int> experts_list = {1, 2, 4, 8};
    std::vector<int> token_list   = {1, 8};
    std::string json_path;
    bool   selftest = false;
    bool   do_feed  = true;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if      (a == "--model")   model = next();
        else if (a == "--layer")   layer = std::stoi(next());
        else if (a == "--iters")   iters = std::stoi(next());
        else if (a == "--warmup")  warmup = std::stoi(next());
        else if (a == "--pool")    pool_experts = std::stoi(next());
        else if (a == "--id-stride") id_stride = std::stoi(next());
        else if (a == "--threads") { thread_list.clear(); std::string s = next();
                                     size_t p = 0; while (p <= s.size()) {
                                         size_t c = s.find(',', p);
                                         if (c == std::string::npos) c = s.size();
                                         thread_list.push_back(std::stoi(s.substr(p, c - p)));
                                         p = c + 1; } }
        else if (a == "--experts") { experts_list.clear(); std::string s = next();
                                     size_t p = 0; while (p <= s.size()) {
                                         size_t c = s.find(',', p);
                                         if (c == std::string::npos) c = s.size();
                                         experts_list.push_back(std::stoi(s.substr(p, c - p)));
                                         p = c + 1; } }
        else if (a == "--tokens")  { token_list.clear(); std::string s = next();
                                     size_t p = 0; while (p <= s.size()) {
                                         size_t c = s.find(',', p);
                                         if (c == std::string::npos) c = s.size();
                                         token_list.push_back(std::stoi(s.substr(p, c - p)));
                                         p = c + 1; } }
        else if (a == "--json")    json_path = next();
        else if (a == "--selftest") selftest = true;
        else if (a == "--no-feed") do_feed = false;
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }

    if (model.empty()) {
        fprintf(stderr,
            "expert-cpu-bench - HOST ONLY, no CUDA, no model load, no GPU mutex.\n"
            "  --model PATH     GGUF with routed-expert *_exps tensors (required)\n"
            "  --layer N        trunk layer to read (default 0; 34/38/39 are q6_K down)\n"
            "  --threads A,B,C  CPU backend thread counts (default 1,2,4,8,16)\n"
            "  --experts A,B,C  experts per call, i.e. n_expert_used (default 1,2,4,8)\n"
            "  --tokens A,B     tokens per call (default 1,8)\n"
            "  --pool N         distinct experts cycled through (default 64)\n"
            "  --id-stride N    expert ids advance by N per iteration (default 1).\n"
            "                   1 makes consecutive iterations overlap in k-1 of k\n"
            "                   experts, so all but one are L2/L3-warm and the\n"
            "                   marginal cost is a CACHE cost, not a DRAM cost.\n"
            "                   Set this to the largest --experts value for\n"
            "                   iteration-disjoint expert sets.\n"
            "  --iters N        timed iterations per cell (default 200)\n"
            "  --warmup N       untimed iterations per cell (default 20)\n"
            "  --no-feed        skip the mmap page-in measurement\n"
            "  --json PATH      write machine-readable results\n"
            "  --selftest       parse and validate metadata only; exit 0/1\n");
        return 2;
    }

    std::error_code ec;
    if (!std::filesystem::exists(model, ec)) {
        fprintf(stderr, "model not found: %s\n", model.c_str());
        return 2;
    }
    const uint64_t file_size = std::filesystem::file_size(model, ec);
    if (ec) { fprintf(stderr, "cannot stat model\n"); return 2; }

    // read the header with a plain ifstream; map the whole file for the data
    std::vector<uint8_t> head(64ull * 1024 * 1024);
    {
        std::ifstream f(model, std::ios::binary);
        if (!f) { fprintf(stderr, "cannot open model\n"); return 2; }
        f.read((char *) head.data(), (std::streamsize) head.size());
        if (f.gcount() < 32) { fprintf(stderr, "model too small\n"); return 2; }
        head.resize((size_t) f.gcount());
    }

    gguf_meta meta;
    if (!gguf_read(head.data(), head.size(), meta)) {
        fprintf(stderr, "GGUF parse FAILED\n");
        return 1;
    }

    char gate_name[128], up_name[128], down_name[128];
    snprintf(gate_name, sizeof gate_name, "blk.%d.ffn_gate_exps.weight", layer);
    snprintf(up_name,   sizeof up_name,   "blk.%d.ffn_up_exps.weight",   layer);
    snprintf(down_name, sizeof down_name, "blk.%d.ffn_down_exps.weight", layer);

    const tensor_info * ti_gate = find_tensor(meta, gate_name);
    const tensor_info * ti_up   = find_tensor(meta, up_name);
    const tensor_info * ti_down = find_tensor(meta, down_name);

    printf("== model ==\n");
    printf("  path              %s\n", model.c_str());
    printf("  bytes             %llu\n", (unsigned long long) file_size);
    printf("  gguf version      %u   alignment %llu   data_offset %llu\n",
           meta.version, (unsigned long long) meta.alignment,
           (unsigned long long) meta.data_offset);
    printf("  architecture      %s\n", meta.arch.c_str());
    printf("  expert_count      %llu   expert_used %llu   n_embd %llu   block_count %llu\n",
           (unsigned long long) meta.n_expert, (unsigned long long) meta.n_expert_used,
           (unsigned long long) meta.n_embd, (unsigned long long) meta.n_layer);
    printf("  layer             %d\n", layer);

    if (!ti_gate || !ti_up || !ti_down) {
        fprintf(stderr, "missing expert tensors for layer %d (gate=%p up=%p down=%p)\n",
                layer, (const void *) ti_gate, (const void *) ti_up, (const void *) ti_down);
        return 1;
    }

    // byte arithmetic - this is the invariant the whole exercise rests on
    const size_t row_g = ggml_row_size((enum ggml_type) ti_gate->type, (int64_t) ti_gate->dims[0]);
    const size_t row_u = ggml_row_size((enum ggml_type) ti_up->type,   (int64_t) ti_up->dims[0]);
    const size_t row_d = ggml_row_size((enum ggml_type) ti_down->type, (int64_t) ti_down->dims[0]);
    const uint64_t per_expert =
        (uint64_t) row_g * ti_gate->dims[1] +
        (uint64_t) row_u * ti_up->dims[1]   +
        (uint64_t) row_d * ti_down->dims[1];
    const uint64_t bank_bytes =
        (uint64_t) row_g * ti_gate->dims[1] * ti_gate->dims[2] +
        (uint64_t) row_u * ti_up->dims[1]   * ti_up->dims[2]   +
        (uint64_t) row_d * ti_down->dims[1] * ti_down->dims[2];

    printf("  gate              %s  ne=[%llu,%llu,%llu]  type %s\n", ti_gate->name.c_str(),
           (unsigned long long) ti_gate->dims[0], (unsigned long long) ti_gate->dims[1],
           (unsigned long long) ti_gate->dims[2], ggml_type_name((enum ggml_type) ti_gate->type));
    printf("  up                %s  ne=[%llu,%llu,%llu]  type %s\n", ti_up->name.c_str(),
           (unsigned long long) ti_up->dims[0], (unsigned long long) ti_up->dims[1],
           (unsigned long long) ti_up->dims[2], ggml_type_name((enum ggml_type) ti_up->type));
    printf("  down              %s  ne=[%llu,%llu,%llu]  type %s\n", ti_down->name.c_str(),
           (unsigned long long) ti_down->dims[0], (unsigned long long) ti_down->dims[1],
           (unsigned long long) ti_down->dims[2], ggml_type_name((enum ggml_type) ti_down->type));
    printf("  bytes / expert    %llu   (gate %llu + up %llu + down %llu)\n",
           (unsigned long long) per_expert,
           (unsigned long long) ((uint64_t) row_g * ti_gate->dims[1]),
           (unsigned long long) ((uint64_t) row_u * ti_up->dims[1]),
           (unsigned long long) ((uint64_t) row_d * ti_down->dims[1]));
    printf("  bytes / bank      %llu   (per_expert x n_expert = %llu)\n",
           (unsigned long long) bank_bytes,
           (unsigned long long) (per_expert * ti_gate->dims[2]));

    // selftest: metadata present and the byte arithmetic closes
    if (selftest) {
        int fails = 0;
        auto check = [&](bool ok, const char * what) {
            printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
            if (!ok) ++fails;
        };
        check(meta.n_expert > 0, "expert_count present");
        check(meta.n_expert_used > 0, "expert_used_count present");
        check(ti_gate->dims[2] == meta.n_expert, "gate ne[2] == expert_count");
        check(ti_down->dims[2] == meta.n_expert, "down ne[2] == expert_count");
        check(ti_gate->dims[0] == meta.n_embd, "gate ne[0] == n_embd (mul_mat_id lhs)");
        check(ti_up->dims[0]   == meta.n_embd, "up   ne[0] == n_embd (mul_mat_id lhs)");
        check(ti_down->dims[0] == ti_gate->dims[1], "down ne[0] == n_ff (mul_mat_id lhs on the activation)");
        check(ti_down->dims[1] == meta.n_embd, "down ne[1] == n_embd (output width)");
        printf("  [note] down_exps is stored [%llu,%llu,%llu] = [n_ff,n_embd,n_expert]\n",
               (unsigned long long) ti_down->dims[0], (unsigned long long) ti_down->dims[1],
               (unsigned long long) ti_down->dims[2]);
        check(per_expert * ti_gate->dims[2] == bank_bytes, "bank == per_expert x n_expert");
        check(meta.data_offset < file_size, "data_offset inside file");
        if (per_expert != (uint64_t) ref::ROW_BYTES) {
            printf("  [note] per-expert bytes %llu differ from the recorded %llu "
                   "(expected on a q6_K layer such as 34/38/39)\n",
                   (unsigned long long) per_expert, (unsigned long long) ref::ROW_BYTES);
        }
        printf("selftest: %s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
        return fails ? 1 : 0;
    }

    mapping map;
    if (!map.open(model)) { fprintf(stderr, "mmap FAILED\n"); return 1; }

    const int64_t n_embd = (int64_t) ti_gate->dims[0];
    const int64_t n_ff   = (int64_t) ti_gate->dims[1];
    const int64_t n_exp  = (int64_t) ti_gate->dims[2];

    const size_t gate_off = (size_t) (meta.data_offset + ti_gate->offset);
    const size_t up_off   = (size_t) (meta.data_offset + ti_up->offset);
    const size_t down_off = (size_t) (meta.data_offset + ti_down->offset);
    // Per-tensor expert stride. NOTE: down_exps is stored [n_ff, n_embd, n_expert]
    // while gate/up are [n_embd, n_ff, n_expert], so a shared stride would be an
    // assumption, not a fact. Each stride is derived from its own ne[].
    const size_t stride_gate = (size_t) row_g * (size_t) ti_gate->dims[1];
    const size_t stride_up   = (size_t) row_u * (size_t) ti_up->dims[1];
    const size_t stride_down = (size_t) row_d * (size_t) ti_down->dims[1];
    const size_t len_gate = stride_gate, len_up = stride_up, len_down = stride_down;

    if (gate_off + (size_t) n_exp * stride_gate > map.n ||
        up_off   + (size_t) n_exp * stride_up   > map.n ||
        down_off + (size_t) n_exp * stride_down > map.n) {
        fprintf(stderr, "expert tensors fall outside the mapping\n");
        map.close();
        return 1;
    }

    // ---------------- FEED: page-in cost of the expert bytes ----------------
    double feed_cold_us = 0.0, feed_hot_us = 0.0;
    if (do_feed) {
        const size_t nbytes = (size_t) per_expert;
        std::vector<uint8_t> scratch(nbytes);
        volatile uint64_t sink = 0;

        // cold = first touch of this expert's bytes, wherever they are in the
        // Windows file cache. Uncontrolled by construction; reported as such.
        const double t0 = now_us();
        for (int k = 0; k < 3; ++k) {
            const uint8_t * srcs[3] = {
                (const uint8_t *) map.p + gate_off + stride_gate * (size_t) k,
                (const uint8_t *) map.p + up_off   + stride_up   * (size_t) k,
                (const uint8_t *) map.p + down_off + stride_down * (size_t) k,
            };
            const size_t   lens[3] = { len_gate, len_up, len_down };
            size_t at = 0;
            for (int t = 0; t < 3; ++t) {
                memcpy(scratch.data() + at, srcs[t], lens[t]);
                at += lens[t];
            }
        }
        const double t1 = now_us();
        sink += scratch[0];
        feed_cold_us = (t1 - t0) / 3.0;

        // hot = the same bytes already in ordinary RAM
        std::vector<double> hot;
        hot.reserve((size_t) iters);
        for (int it = 0; it < iters; ++it) {
            const int k = it % (int) n_exp;
            const uint8_t * srcs[3] = {
                (const uint8_t *) map.p + gate_off + stride_gate * (size_t) k,
                (const uint8_t *) map.p + up_off   + stride_up   * (size_t) k,
                (const uint8_t *) map.p + down_off + stride_down * (size_t) k,
            };
            const size_t lens[3] = { len_gate, len_up, len_down };
            const double a = now_us();
            size_t at = 0;
            for (int t = 0; t < 3; ++t) { memcpy(scratch.data() + at, srcs[t], lens[t]); at += lens[t]; }
            const double b = now_us();
            sink += scratch[nbytes - 1];
            hot.push_back(b - a);
        }
        feed_hot_us = pct(hot, 0.5);
        printf("\n== feed (host memcpy of one expert's bytes, page-in) ==\n");
        printf("  first touch (UNCONTROLLED file cache)  %8.1f us  %6.2f GB/s\n",
               feed_cold_us, (double) nbytes / feed_cold_us / 1000.0);
        printf("  already resident (median of %d)      %8.1f us  %6.2f GB/s\n",
               iters, feed_hot_us, (double) nbytes / feed_hot_us / 1000.0);
        printf("  note: this is a page-in rate, not a cold-cache guarantee. The Windows\n"
               "        file cache state is uncontrolled and is reported as such.\n");
    }

    // ---------------- COMPUTE: the real op chain, weights in host RAM -------
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "ggml_backend_cpu_init FAILED\n"); map.close(); return 1; }

    // One weight context holding the full three tensors. Only the pool experts
    // are populated, so the untouched pages are never faulted in.
    const size_t wmem = ggml_tensor_overhead() * 8 + 1024 * 1024;
    ggml_init_params wip{ wmem, nullptr, /*no_alloc*/ true };
    ggml_context * ctx_w = ggml_init(wip);
    ggml_tensor * w_gate = ggml_new_tensor_4d(ctx_w, (enum ggml_type) ti_gate->type,
                                              ti_gate->dims[0], ti_gate->dims[1], ti_gate->dims[2], 1);
    ggml_tensor * w_up   = ggml_new_tensor_4d(ctx_w, (enum ggml_type) ti_up->type,
                                              ti_up->dims[0], ti_up->dims[1], ti_up->dims[2], 1);
    ggml_tensor * w_down = ggml_new_tensor_4d(ctx_w, (enum ggml_type) ti_down->type,
                                              ti_down->dims[0], ti_down->dims[1], ti_down->dims[2], 1);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    if (!wbuf) { fprintf(stderr, "weight buffer alloc FAILED\n"); map.close(); return 1; }

    pool_experts = std::min(pool_experts, (int) n_exp);
    for (int e = 0; e < pool_experts; ++e) {
        ggml_backend_tensor_set(w_gate, (const uint8_t *) map.p + gate_off + stride_gate * (size_t) e,
                                stride_gate * (size_t) e, len_gate);
        ggml_backend_tensor_set(w_up,   (const uint8_t *) map.p + up_off   + stride_up   * (size_t) e,
                                stride_up   * (size_t) e, len_up);
        ggml_backend_tensor_set(w_down, (const uint8_t *) map.p + down_off + stride_down * (size_t) e,
                                stride_down * (size_t) e, len_down);
    }

    std::vector<cell_result> results;
    volatile float sink_f = 0.0f;

    printf("\n== compute (ggml CPU backend, weights in host RAM, cycling %d experts, id-stride %d) ==\n",
           pool_experts, id_stride);
    printf("  %7s %8s %7s %11s %10s %10s %9s\n",
           "threads", "experts", "tokens", "median_us", "p90_us", "bytes", "GB/s");

    for (int nt : thread_list) {
        if (nt < 1) continue;
        ggml_backend_cpu_set_n_threads(backend, nt);

        for (int eu : experts_list) {
            if (eu < 1 || eu > pool_experts) continue;
            for (int ntok : token_list) {
                if (ntok < 1) continue;

                // build the whole graph in ONE context so the weights and the
                // activations share a single buffer
                const size_t mem = ggml_tensor_overhead() * 64 + ggml_graph_overhead() + 8 * 1024 * 1024;
                ggml_init_params ip{ mem, nullptr, true };
                ggml_context * ctx = ggml_init(ip);
                if (!ctx) { fprintf(stderr, "context alloc FAILED\n"); return 1; }

                ggml_tensor * wg = ggml_view_tensor(ctx, w_gate);
                ggml_tensor * wu = ggml_view_tensor(ctx, w_up);
                ggml_tensor * wd = ggml_view_tensor(ctx, w_down);
                ggml_set_name(wg, "w_gate"); ggml_set_name(wu, "w_up"); ggml_set_name(wd, "w_down");

                // mul_mat_id requires ids->ne[1] == b->ne[2], so the engine's
                // expert-id tensor is 2-D: [n_expert_used, n_tokens].
                ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, eu, ntok);
                ggml_set_input(ids);
                ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, ntok);
                ggml_set_input(x);
                ggml_tensor * rwt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, eu);
                ggml_set_input(rwt);

                ggml_tensor * cur = ggml_reshape_3d(ctx, x, n_embd, 1, ntok);
                ggml_tensor * up    = ggml_mul_mat_id(ctx, wu, cur, ids);
                ggml_tensor * gate  = ggml_mul_mat_id(ctx, wg, cur, ids);
                ggml_tensor * act   = ggml_swiglu_split(ctx, gate, up);
                ggml_tensor * down  = ggml_mul_mat_id(ctx, wd, act, ids);
                down = ggml_mul(ctx, down, rwt);

                // down is [n_embd, n_expert_used, n_tokens] - the expert axis is
                // ne[1]. build_moe_ffn slices expert i with offset i*nb[1] and
                // strides by nb[2] to span every token; mirror that exactly.
                ggml_tensor * moe = ggml_view_3d(ctx, down, n_embd, ntok, 1,
                                                  down->nb[2], down->nb[2], 0);
                for (int i = 1; i < eu; ++i) {
                    ggml_tensor * v = ggml_view_3d(ctx, down, n_embd, ntok, 1,
                                                   down->nb[2], down->nb[2],
                                                   (size_t) i * down->nb[1]);
                    moe = ggml_add(ctx, moe, v);
                }
                ggml_set_output(moe);

                ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, moe);

                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
                if (!buf) { fprintf(stderr, "graph buffer alloc FAILED\n"); return 1; }

                // inputs
                {
                    std::vector<float> xin((size_t) n_embd * ntok);
                    for (size_t i = 0; i < xin.size(); ++i) {
                        xin[i] = 0.001f * (float) ((i * 2654435761u) % 1000u) - 0.5f;
                    }
                    ggml_backend_tensor_set(x, xin.data(), 0, ggml_nbytes(x));
                    std::vector<float> rin((size_t) n_embd * eu, 0.05f);
                    ggml_backend_tensor_set(rwt, rin.data(), 0, ggml_nbytes(rwt));
                }
                std::vector<int32_t> idbuf((size_t) eu * (size_t) ntok);

                // rotate the window each iteration so the working set cannot
                // sit in cache; every (expert slot, token) gets a live id
                auto set_ids = [&](int base) {
                    for (int t = 0; t < ntok; ++t) {
                        for (int i = 0; i < eu; ++i) {
                            idbuf[(size_t) t * (size_t) eu + (size_t) i] =
                                (base + i + t) % pool_experts;
                        }
                    }
                    ggml_backend_tensor_set(ids, idbuf.data(), 0, ggml_nbytes(ids));
                };

                for (int it = 0; it < warmup; ++it) {
                    set_ids(it * id_stride);
                    ggml_backend_graph_compute(backend, gf);
                }

                std::vector<double> samples;
                samples.reserve((size_t) iters);
                for (int it = 0; it < iters; ++it) {
                    set_ids(it * id_stride);
                    const double a = now_us();
                    ggml_backend_graph_compute(backend, gf);
                    const double b = now_us();
                    samples.push_back(b - a);
                }
                {
                    std::vector<float> out(1);
                    ggml_backend_tensor_get(moe, out.data(), 0, sizeof(float));
                    sink_f += out[0];
                }

                cell_result cr;
                cr.threads      = nt;
                cr.experts_used = eu;
                cr.n_tokens     = ntok;
                cr.med_us  = pct(samples, 0.5);
                cr.p90_us  = pct(samples, 0.9);
                cr.min_us  = pct(samples, 0.0);
                cr.bytes   = (double) per_expert * (double) eu;
                cr.gbps    = cr.med_us > 0 ? (cr.bytes / cr.med_us) / 1000.0 : 0.0;
                results.push_back(cr);

                printf("  %7d %8d %7d %11.1f %10.1f %10.0f %9.2f\n",
                       cr.threads, cr.experts_used, cr.n_tokens,
                       cr.med_us, cr.p90_us, cr.bytes, cr.gbps);

                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
            }
        }
    }

    // ---------------- derived: the numbers the T-GPLR-61 gate needs ----------
    printf("\n== derived (arithmetic on the measured cells above; NOT a measurement) ==\n");
    for (const cell_result & cr : results) {
        if (cr.n_tokens != 1 || cr.threads != (thread_list.empty() ? 1 : thread_list.back())) continue;
        const double per_layer_us = cr.med_us;                       // 1 token, k experts
        const double per_tok_ms   = per_layer_us * ref::N_LAYERS / 1000.0;
        const double tok_s         = per_tok_ms > 0 ? 1000.0 / per_tok_ms : 0.0;
        const double link_pageable_us = (cr.bytes / 1e9) /
            ((ref::LINK_PAGEABLE_LO_GBPS + ref::LINK_PAGEABLE_HI_GBPS) / 2.0) * 1e6;
        const double link_reg_us = (cr.bytes / 1e9) / ref::LINK_REGISTERED_GBPS * 1e6;
        printf("  k=%d experts, 1 token, %d threads: CPU %7.1f us | "
               "link(pageable ~7.35 GB/s) %7.1f us | link(registered 14.6) %7.1f us"
               " | CPU/40 layers = %6.2f ms/token -> %6.2f tok/s if it were the ONLY cost\n",
               cr.experts_used, cr.threads, cr.med_us, link_pageable_us, link_reg_us,
               per_tok_ms, tok_s);
    }
    printf("  the tok/s figure above is a floor-style arithmetic bound for the expert FFN alone:\n"
           "  it excludes attention, the router, the shared expert, KV traffic, graph launch\n"
           "  and all staging, and it assumes the CPU work is SERIAL with everything else.\n");

    if (!json_path.empty()) {
        FILE * f = fopen(json_path.c_str(), "w");
        if (f) {
            fprintf(f, "{\n  \"tool\": \"expert-cpu-bench\",\n");
            fprintf(f, "  \"model_bytes\": %llu,\n", (unsigned long long) file_size);
            fprintf(f, "  \"layer\": %d,\n", layer);
            fprintf(f, "  \"per_expert_bytes\": %llu,\n", (unsigned long long) per_expert);
            fprintf(f, "  \"bank_bytes\": %llu,\n", (unsigned long long) bank_bytes);
            fprintf(f, "  \"pool_experts\": %d,\n", pool_experts);
            fprintf(f, "  \"id_stride\": %d,\n", id_stride);
            fprintf(f, "  \"iters\": %d,\n", iters);
            if (do_feed) {
                fprintf(f, "  \"feed_first_touch_us\": %.3f,\n", feed_cold_us);
                fprintf(f, "  \"feed_resident_median_us\": %.3f,\n", feed_hot_us);
            } else {
                fprintf(f, "  \"feed_first_touch_us\": null,\n");
                fprintf(f, "  \"feed_resident_median_us\": null,\n");
            }
            fprintf(f, "  \"cells\": [\n");
            for (size_t i = 0; i < results.size(); ++i) {
                const cell_result & c = results[i];
                fprintf(f, "    {\"threads\": %d, \"experts_used\": %d, \"n_tokens\": %d, "
                           "\"median_us\": %.3f, \"p90_us\": %.3f, \"min_us\": %.3f, "
                           "\"bytes\": %.0f, \"gbps\": %.3f}%s\n",
                        c.threads, c.experts_used, c.n_tokens, c.med_us, c.p90_us, c.min_us,
                        c.bytes, c.gbps, (i + 1 < results.size()) ? "," : "");
            }
            fprintf(f, "  ]\n}\n");
            fclose(f);
            printf("\n  wrote %s\n", json_path.c_str());
        }
    }

    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx_w);
    ggml_backend_free(backend);
    map.close();
    return 0;
}
