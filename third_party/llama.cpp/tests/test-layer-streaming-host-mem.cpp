// T-GPLR-36 follow-up (2026-09-18): host-only test for the host-memory
// instrument (llama_ls_host_mem_sample).
//
// The instrument exists because a measured quantity had no instrument at all:
// the runner's host resident set, observed climbing ~19 GB -> ~30 GB across a
// run while the trace carried VRAM accounting only. A wrong instrument is worse
// than a missing one, so this pins the two things that make it usable:
//
//   1. INVARIANTS hold -- the fields are self-consistent (peak >= current,
//      private resident <= working set, free <= total, and the derived
//      file-backed share is exact by construction).
//   2. It reads THIS process -- a forced, touched allocation must move the
//      reading. That is the property a constant, a cached value, or a
//      wrong-process query would fail while still producing plausible numbers.
//
// No CUDA, no model, no GPU: pure host counters, so this runs anywhere.
#include "../src/llama-layer-streaming.h"

#include <cstdint>
#include <cstdio>
#include <vector>

static int failures = 0;
static const uint64_t MiB = 1024ull * 1024ull;

int main(void) {
    const llama_ls_host_mem a = llama_ls_host_mem_sample();

    if (!a.ok) {
        // Honest absence, not a fabricated zero: the residency path is
        // CUDA/Windows-only, so a platform without these counters reports
        // ok == false and there is nothing to assert.
        printf("SKIP host-mem: sampler reports no counters on this platform\n");
        return 0;
    }

    // 1. self-consistency of one sample
    if (a.ws_bytes == 0) {
        failures++;
        fprintf(stderr, "FAIL ws_bytes: working set reads 0 in a live process\n");
    }
    if (a.ws_peak_bytes < a.ws_bytes) {
        failures++;
        fprintf(stderr, "FAIL ws_peak_bytes: lifetime peak %llu < current %llu\n",
                (unsigned long long) a.ws_peak_bytes, (unsigned long long) a.ws_bytes);
    }
    if (a.commit_bytes == 0) {
        failures++;
        fprintf(stderr, "FAIL commit_bytes: private commit reads 0\n");
    }
    if (a.ws_private_bytes > a.ws_bytes) {
        failures++;
        fprintf(stderr, "FAIL ws_private_bytes: private resident %llu > working set %llu\n",
                (unsigned long long) a.ws_private_bytes, (unsigned long long) a.ws_bytes);
    }
    // The private/file-backed split is the attribution axis, so pin its contract
    // rather than trusting a zero: known => a real nonzero reading;
    // not known => explicitly zero AND flagged. A silent zero would make the
    // derived file-backed share read as the entire working set.
    if (a.ws_private_known) {
        if (a.ws_private_bytes == 0) {
            failures++;
            fprintf(stderr, "FAIL ws_private_bytes: flagged known but reads 0\n");
        }
    } else {
        fprintf(stderr, "WARN host-mem: PROCESS_MEMORY_COUNTERS_EX2 private-resident split unavailable "
                        "(ws_private_known=false); the file-backed share is NOT derivable on this box\n");
        if (a.ws_private_bytes != 0) {
            failures++;
            fprintf(stderr, "FAIL ws_private_bytes: flagged unknown but reads %llu\n",
                    (unsigned long long) a.ws_private_bytes);
        }
    }
    if (a.sys_total_bytes == 0 || a.sys_free_bytes > a.sys_total_bytes) {
        failures++;
        fprintf(stderr, "FAIL system memory: free %llu total %llu\n",
                (unsigned long long) a.sys_free_bytes, (unsigned long long) a.sys_total_bytes);
    }
    if (a.sys_commit_total_bytes != 0 && a.sys_commit_free_bytes > a.sys_commit_total_bytes) {
        failures++;
        fprintf(stderr, "FAIL system commit: free %llu total %llu\n",
                (unsigned long long) a.sys_commit_free_bytes,
                (unsigned long long) a.sys_commit_total_bytes);
    }

    const uint64_t file_bytes = a.ws_bytes - a.ws_private_bytes;   // valid only when ws_private_known
    printf("sample 1: ws %llu MiB (private %llu MiB, file-backed/shared %llu MiB), "
           "commit %llu MiB, os peak %llu MiB, sys free %llu MiB of %llu MiB\n",
           (unsigned long long) (a.ws_bytes / MiB), (unsigned long long) (a.ws_private_bytes / MiB),
           (unsigned long long) (file_bytes / MiB), (unsigned long long) (a.commit_bytes / MiB),
           (unsigned long long) (a.ws_peak_bytes / MiB), (unsigned long long) (a.sys_free_bytes / MiB),
           (unsigned long long) (a.sys_total_bytes / MiB));

    // 2. it measures THIS process: 256 MiB allocated and touched page-by-page
    //    must show up in the working set. The threshold is deliberately loose
    //    (64 MiB) because the OS may trim the working set concurrently -- the
    //    check is "the reading moved with our allocation", not an exact size.
    const size_t n_bytes = (size_t) (256 * MiB);
    std::vector<uint8_t> block(n_bytes, 0);
    for (size_t i = 0; i < n_bytes; i += 4096) {
        block[i] = (uint8_t) (i & 0xff);
    }
    // defeat any optimizer elision of the touches above
    volatile uint8_t sink = block[n_bytes - 4096];
    (void) sink;

    const llama_ls_host_mem b = llama_ls_host_mem_sample();
    const uint64_t grew = b.ws_bytes > a.ws_bytes ? b.ws_bytes - a.ws_bytes : 0;
    printf("sample 2 after touching 256 MiB: ws %llu MiB (%+lld MiB), commit %llu MiB (%+lld MiB)\n",
           (unsigned long long) (b.ws_bytes / MiB), (long long) (b.ws_bytes / MiB) - (long long) (a.ws_bytes / MiB),
           (unsigned long long) (b.commit_bytes / MiB),
           (long long) (b.commit_bytes / MiB) - (long long) (a.commit_bytes / MiB));

    if (grew < 64 * MiB) {
        failures++;
        fprintf(stderr, "FAIL reactivity: 256 MiB of touched private memory moved the "
                        "working set by only %llu MiB (>= 64 MiB required)\n",
                (unsigned long long) (grew / MiB));
    }
    if (b.ws_peak_bytes < a.ws_peak_bytes) {
        failures++;
        fprintf(stderr, "FAIL os peak: %llu fell below the earlier %llu (it is a lifetime max)\n",
                (unsigned long long) b.ws_peak_bytes, (unsigned long long) a.ws_peak_bytes);
    }
    if (b.commit_bytes + 1 < a.commit_bytes) {
        failures++;
        fprintf(stderr, "FAIL commit: fell from %llu to %llu while 256 MiB is held\n",
                (unsigned long long) a.commit_bytes, (unsigned long long) b.commit_bytes);
    }

    // 3. the derived attribution axis is exact by construction: private resident
    //    plus file-backed/shared equals the working set, so a reader subtracting
    //    the two cannot accumulate error.
    if (file_bytes + a.ws_private_bytes != a.ws_bytes) {
        failures++;
        fprintf(stderr, "FAIL attribution split: file %llu + private %llu != ws %llu\n",
                (unsigned long long) file_bytes, (unsigned long long) a.ws_private_bytes,
                (unsigned long long) a.ws_bytes);
    }

    if (failures != 0) {
        fprintf(stderr, "\nFAILED: %d check(s)\n", failures);
        return 1;
    }
    printf("\ntest-layer-streaming-host-mem: all cases PASS\n");
    return 0;
}
