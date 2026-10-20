#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

// T-GPLR-35 experimental (layer-residency research): staging DMAs on an
// auxiliary stream (No. 1) of the backend's OWN CUDA context, so the ordering
// waits between those DMAs and the context's compute stream (No. 0) stay
// intra-context. Used only by the opt-in layer-streaming experiment; no normal
// execution path calls these, and nothing here changes device math.
GGML_BACKEND_API bool ggml_backend_cuda_aux_stream_ok(ggml_backend_t backend);
GGML_BACKEND_API void ggml_backend_cuda_memcpy_1d_aux(ggml_backend_t backend, void * dst, const void * src, size_t nbytes);
GGML_BACKEND_API void ggml_backend_cuda_event_record_aux(ggml_backend_t backend, ggml_backend_event_t event);
GGML_BACKEND_API void ggml_backend_cuda_event_wait_aux(ggml_backend_t backend, ggml_backend_event_t event);
GGML_BACKEND_API void ggml_backend_cuda_aux_synchronize(ggml_backend_t backend);

// T-GPLR-37 experimental (layer-residency research): page-lock a MAPPED source
// range so a bank DMA can be sourced straight from the model's own mapping
// instead of from a CPU-filled pinned mirror. Unlike
// ggml_backend_cuda_register_host_buffer this is not gated behind an env var
// and it REPORTS the driver error, because the layer-residency path must refuse
// loudly rather than silently fall back to the copy hop. Measured on this box
// (tools/gplr37-mmapreg-probe.cu): registration 0.16 ms per 452 MiB window and
// the DMA then runs at 14.44 GB/s, the pinned ceiling.
// Dense decode streaming (private fork): the CUDA backend calls fn(user, node, phase) for every node flagged GGML_TENSOR_FLAG_STREAM_MARK that it launches, phase 0 right
// before the node's kernel is launched and phase 1 right after. The function runs on the thread that computes the graph and may only ENQUEUE work (CUDA events, async copies).
// A marked node is never fused into a chain (the backend aborts if it would be). Without a registered function the flag does nothing. CUDA graphs are not used for a graph that has marked nodes.
typedef void (*ggml_backend_cuda_stream_mark_fn)(void * user, const struct ggml_tensor * node, int phase);
GGML_BACKEND_API void ggml_backend_cuda_set_stream_mark(ggml_backend_cuda_stream_mark_fn fn, void * user);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_range(const void * ptr, size_t nbytes, char * err, size_t err_len);
GGML_BACKEND_API bool ggml_backend_cuda_unregister_host_range(const void * ptr);

// T-GPLR-35 recon (diagnostics only): stream identity plus timing-enabled
// events, so the layer-streaming attribution pass can measure a staging copy's
// own GPU window and its start offset against the compute stream's backlog.
// ggml's own backend events use cudaEventDisableTiming, so cudaEventElapsedTime
// cannot be used on them. Nothing here orders work or sits on the critical path.
GGML_BACKEND_API void * ggml_backend_cuda_stream_handle(ggml_backend_t backend, int stream_no);

// T-Q36-AER: submit count disjoint host->device region copies (dsts[i] <-
// srcs[i], sizes[i] bytes) as ONE batched async memcpy (cudaMemcpyBatchAsync,
// CUDA 12.8+) on the backend's stream. Copies within the batch may execute in
// any order, so regions must be disjoint. Returns false when the runtime
// lacks or refuses the batch API; the caller then falls back to per-copy
// ggml_backend_tensor_set_async (same stream, same bytes).
GGML_BACKEND_API bool ggml_backend_cuda_copy_batch_async(ggml_backend_t backend,
                                                         void * const * dsts, const void * const * srcs,
                                                         const size_t * sizes, size_t count);
GGML_BACKEND_API void * ggml_backend_cuda_timing_event_new(ggml_backend_t backend, int stream_no);
GGML_BACKEND_API void   ggml_backend_cuda_timing_event_record(void * handle);
GGML_BACKEND_API double ggml_backend_cuda_timing_event_elapsed_ms(void * a, void * b); // < 0 = not ready
GGML_BACKEND_API void   ggml_backend_cuda_timing_event_free(void * handle);

// T-Q36-AER experimental hybrid MoE decode (ggml_moe_hybrid_publish / _wait_add):
// host mailbox shared by the CUDA kernels and the host miss engine. It lives in
// pinned, device-mapped host memory; req_seq and res_seq sit on their own cache
// lines. The device posts misses then req_seq; the host writes y then res_seq.
#define GGML_MOE_HYBRID_MAX_EMBD  4096
#define GGML_MOE_HYBRID_MAX_USED  8
#define GGML_MOE_HYBRID_MAX_LAYER 128
struct ggml_moe_hybrid_mailbox {
    volatile int32_t req_seq;
    int32_t pad0[15];
    int32_t layer;
    int32_t nmiss;
    int32_t err;
    int32_t pad1;
    int32_t ids[GGML_MOE_HYBRID_MAX_USED];
    float   w[GGML_MOE_HYBRID_MAX_USED];
    int32_t pad2[12];
    float   x[GGML_MOE_HYBRID_MAX_EMBD];
    volatile int32_t res_seq;
    int32_t pad3[15];
    float   y[GGML_MOE_HYBRID_MAX_EMBD];
    // per-layer timeline (diagnostics only): device globaltimer ns at publish, wait start, wait end
    // and result-add end (after the host result was loaded), then the host serve time in ns
    // (left 0 by the host when the layer had no miss)
    uint64_t prof[GGML_MOE_HYBRID_MAX_LAYER][5];
};
// device pointer of a pinned host allocation, NULL if the memory is not device-mapped
GGML_BACKEND_API void * ggml_backend_cuda_host_device_ptr(void * host_ptr);
// ggml_backend_cuda_copy_batch_async on the auxiliary stream (No. 1)
GGML_BACKEND_API bool ggml_backend_cuda_copy_batch_aux(ggml_backend_t backend,
                                                       void * const * dsts, const void * const * srcs,
                                                       const size_t * sizes, size_t count);
// non-blocking completion test of a backend event (cudaEventQuery)
GGML_BACKEND_API bool ggml_backend_cuda_event_ready(ggml_backend_event_t event);
// hand the auxiliary stream's queued work to the GPU now (cudaStreamQuery): under WDDM the driver
// batches submissions and would otherwise start them only at the next synchronizing call
GGML_BACKEND_API void ggml_backend_cuda_aux_flush(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
