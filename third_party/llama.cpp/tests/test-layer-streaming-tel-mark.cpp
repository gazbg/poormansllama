// Host-only unit test for the T-GPLR-56 A1 pre-sink telemetry markers, extended
// 2026-09-22 with the post-flush (late) marker case.
// No GPU, no model, no CUDA: proves that a marker recorded BEFORE the telemetry
// sink exists survives to a flush, keeps call order, carries a monotonic t_us,
// declares pre_run (no run_id), and escapes a Windows path. This is the shape
// proof for Q = pass1_end_ts - proc_start_ts; the end-to-end trace is an arm.
#include "../src/llama-layer-streaming.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;

static void expect(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        failures++;
    }
}

static std::vector<std::string> read_lines(const char * path) {
    std::vector<std::string> out;
    FILE * f = fopen(path, "r");
    if (f == nullptr) {
        return out;
    }
    char buf[4096];
    while (fgets(buf, sizeof(buf), f) != nullptr) {
        std::string s(buf);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
            s.pop_back();
        }
        out.push_back(s);
    }
    fclose(f);
    return out;
}

static long long json_num(const std::string & line, const char * key) {
    const std::string pat = std::string("\"") + key + "\":";
    const size_t p = line.find(pat);
    if (p == std::string::npos) {
        return -1;
    }
    return strtoll(line.c_str() + p + pat.size(), nullptr, 10);
}

static bool json_str_is(const std::string & line, const char * key, const char * val) {
    const std::string pat = std::string("\"") + key + "\":\"";
    const size_t p = line.find(pat);
    if (p == std::string::npos) {
        return false;
    }
    return line.compare(p + pat.size(), strlen(val), val) == 0;
}

int main() {
    const char * path = "test-layer-streaming-tel-mark.jsonl";

    // The order the runner uses, including the real Windows-style model path the
    // load_begin marker has to carry without breaking the JSON.
    llama_ls_tel_mark("proc_start", nullptr);
    llama_ls_tel_mark("load_begin", "C:\\models\\Qwen3.6-35B-A3B.gguf");
    llama_ls_tel_mark("load_end", "ok");

    // No sink yet: flushing to NULL must drop the buffer, not crash or leak it
    // into the next run.
    llama_ls_tel_flush_early(nullptr);

    FILE * f = fopen(path, "w");
    expect(f != nullptr, "test sink opens");
    if (f == nullptr) {
        return 1;
    }
    llama_ls_tel_flush_early(f);
    fclose(f);

    const std::vector<std::string> lines = read_lines(path);
    expect(lines.size() == 0, "a NULL flush consumes the buffer (no markers leak into the sink)");

    // Second run in the same process: marks -> flush, and the shape is asserted.
    llama_ls_tel_mark("proc_start", nullptr);
    llama_ls_tel_mark("load_begin", "C:\\models\\Qwen3.6-35B-A3B.gguf");
    llama_ls_tel_mark("load_end", "ok");
    f = fopen(path, "w");
    expect(f != nullptr, "test sink reopens");
    if (f == nullptr) {
        return 1;
    }
    llama_ls_tel_flush_early(f);
    fclose(f);

    const std::vector<std::string> got = read_lines(path);
    expect(got.size() == 3, "exactly three marker lines are written");
    if (got.size() != 3) {
        for (const std::string & s : got) {
            fprintf(stderr, "  got: %s\n", s.c_str());
        }
        return 1;
    }

    expect(json_str_is(got[0], "event", "proc_start"),   "line 1 is proc_start");
    expect(json_str_is(got[1], "event", "load_begin"),   "line 2 is load_begin");
    expect(json_str_is(got[2], "event", "load_end"),     "line 3 is load_end");
    expect(json_num(got[0], "seq") == 0,                  "seq starts at 0 (call order)");
    expect(json_num(got[1], "seq") == 1,                  "seq increments in call order");
    expect(json_num(got[2], "seq") == 2,                  "seq increments in call order");
    expect(json_num(got[0], "pre_run") == 1,              "pre_run marks a pre-run_start line");
    expect(got[0].find("\"run_id\"") == std::string::npos, "pre-run markers carry no run_id");

    const long long t0 = json_num(got[0], "t_us");
    const long long t1 = json_num(got[1], "t_us");
    const long long t2 = json_num(got[2], "t_us");
    expect(t0 >= 0 && t1 >= t0 && t2 >= t1, "t_us is present and monotonic across markers");

    // The escape is the whole point for the load marker: the backslashes must survive
    // as JSON escapes, so the line stays parseable and the path is recoverable.
    expect(got[1].find("C:\\\\models\\\\Qwen3.6-35B-A3B.gguf") != std::string::npos,
           "load_begin detail carries the escaped model path");
    expect(json_str_is(got[1], "detail", "C:\\\\models\\\\Qwen3.6-35B-A3B.gguf"),
           "escaped detail starts the field (no quote break)");

    // An empty event name is refused rather than written as a malformed line.
    llama_ls_tel_mark("", "ignored");
    llama_ls_tel_mark(nullptr, "ignored");
    f = fopen(path, "w");
    expect(f != nullptr, "test sink reopens for the refusal check");
    if (f != nullptr) {
        llama_ls_tel_flush_early(f);
        fclose(f);
    }
    expect(read_lines(path).empty(), "an empty/NULL event name writes nothing");

    // Case 4 (added 2026-09-22): a marker taken AFTER a sink has been flushed must be
    // WRITTEN. Before the fix it was pushed into the buffer that had already been flushed,
    // and nothing ever flushed it again - silent telemetry loss, with no error and no flag
    // (the engine's close paths now re-flush, which is what turns it into a line). The line
    // is also required to say `late:1`, because otherwise it is indistinguishable from a
    // second emission of the same event name.
    f = fopen(path, "w");
    expect(f != nullptr, "test sink reopens for the late-marker check");
    if (f != nullptr) {
        llama_ls_tel_flush_early(f);                                   // sink live, buffer empty
        llama_ls_tel_mark("load_end", "C:\\models\\Qwen3.6-35B-A3B.gguf");  // taken AFTER the sink
        llama_ls_tel_flush_early(f);                                   // the engine's close-time flush
        llama_ls_tel_flush_early(nullptr);                             // ... then the sink is gone
        fclose(f);
    }
    const std::vector<std::string> late = read_lines(path);
    expect(late.size() == 1, "a marker taken after the flush is WRITTEN, not silently lost");
    if (late.size() == 1) {
        expect(json_str_is(late[0], "event", "load_end"), "the late line carries its event");
        expect(late[0].find("\"late\":1") != std::string::npos, "the late line is flagged late:1");
        expect(json_num(late[0], "pre_run") == 1, "a late line still precedes run_start (pre_run)");
        expect(late[0].find("\"run_id\"") == std::string::npos, "a late line carries no run_id");
        expect(late[0].find("C:\\\\models\\\\Qwen3.6-35B-A3B.gguf") != std::string::npos, "the late line escapes its detail");
    }

    // ... and with NO sink at all a marker is still buffered (as late), so a later flush
    // reaches it: the loss case is closed, not merely relocated.
    llama_ls_tel_mark("after_close", nullptr);
    f = fopen(path, "w");
    expect(f != nullptr, "test sink reopens for the after-close check");
    if (f != nullptr) {
        llama_ls_tel_flush_early(f);
        fclose(f);
        llama_ls_tel_flush_early(nullptr);
    }
    const std::vector<std::string> after = read_lines(path);
    expect(after.size() == 1, "a marker taken with no sink is emitted by the next flush, not dropped");
    if (after.size() == 1) {
        expect(json_str_is(after[0], "event", "after_close"), "the buffered line carries its event");
        expect(after[0].find("\"late\":1") != std::string::npos, "a post-sink line is flagged late:1");
    }

    remove(path);

    if (failures != 0) {
        fprintf(stderr, "test-layer-streaming-tel-mark: %d FAILURE(S)\n", failures);
        return 1;
    }
    printf("test-layer-streaming-tel-mark: ALL CHECKS PASSED (host-only, no GPU)\n");
    return 0;
}
