// T-GPLR-36 follow-up (2026-09-18): HOST-memory sampler for the layer-residency
// telemetry.
//
// WHY THIS IS ITS OWN FILE. The sampler needs <windows.h>, which defines
// `min`, `max` and `small` as MACROS. llama-layer-streaming.cpp legitimately
// uses std::min/std::max (14 sites) and locals named `small` (the telemetry
// line buffer), so including the platform header there would either break the
// build or force a scatter of NOMINMAX/undef workarounds through a 6k-line
// file. Isolating the include here keeps all of that in one place and keeps the
// main source exactly as it was.
//
// The counters answer a question the trace could not: the 200k ledger run
// attributed the time residual, but the owner's observed host-RSS climb
// (~19 GB -> ~30 GB across a run) had no instrument at all -- every byte figure
// emitted was VRAM-side. See the header for the field semantics, in particular
// why `ws_bytes - ws_private_bytes` is the attribution axis (the process's own
// anonymous pages versus the mapped model's file-backed pages).
#include "llama-layer-streaming.h"

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX      // std::min/std::max must survive this include
#  endif
#  include <windows.h>
#  include <psapi.h>
#endif

// Cheap (two kernel queries), allocation-free and side-effect-free, so it can be
// called at a pass boundary without perturbing the timing it is meant to
// explain. A platform without these counters returns ok == false rather than a
// fabricated zero: the test skips on it instead of pretending to measure.
llama_ls_host_mem llama_ls_host_mem_sample() {
    llama_ls_host_mem out;
#if defined(_WIN32)
    // EX2 carries PrivateWorkingSetSize, which is exactly the anonymous share
    // that separates the runner's own pages from the mapped model's.
    PROCESS_MEMORY_COUNTERS_EX2 pmc = {};
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &pmc, sizeof(pmc))) {
        out.ok               = true;
        out.ws_bytes         = (uint64_t) pmc.WorkingSetSize;
        out.ws_peak_bytes    = (uint64_t) pmc.PeakWorkingSetSize;
        out.ws_private_bytes = (uint64_t) pmc.PrivateWorkingSetSize;
        out.ws_private_known = true;
        out.commit_bytes     = (uint64_t) pmc.PrivateUsage;
    } else {
        // Older SDK/OS fallback: every field above except PrivateWorkingSetSize,
        // which then reads 0 and is disclosed as unavailable rather than
        // reported as "no private pages".
        PROCESS_MEMORY_COUNTERS_EX ex = {};
        ex.cb = sizeof(ex);
        if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &ex, sizeof(ex))) {
            out.ok            = true;
            out.ws_bytes      = (uint64_t) ex.WorkingSetSize;
            out.ws_peak_bytes = (uint64_t) ex.PeakWorkingSetSize;
            out.commit_bytes  = (uint64_t) ex.PrivateUsage;
        }
    }
    MEMORYSTATUSEX ms = {};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        out.sys_total_bytes        = (uint64_t) ms.ullTotalPhys;
        out.sys_free_bytes         = (uint64_t) ms.ullAvailPhys;
        out.sys_commit_total_bytes = (uint64_t) ms.ullTotalPageFile;
        out.sys_commit_free_bytes  = (uint64_t) ms.ullAvailPageFile;
    }
#endif
    return out;
}
