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
GGML_BACKEND_API bool ggml_backend_cuda_register_host_range(const void * ptr, size_t nbytes, char * err, size_t err_len);
GGML_BACKEND_API bool ggml_backend_cuda_unregister_host_range(const void * ptr);

// T-GPLR-35 recon (diagnostics only): stream identity plus timing-enabled
// events, so the layer-streaming attribution pass can measure a staging copy's
// own GPU window and its start offset against the compute stream's backlog.
// ggml's own backend events use cudaEventDisableTiming, so cudaEventElapsedTime
// cannot be used on them. Nothing here orders work or sits on the critical path.
GGML_BACKEND_API void * ggml_backend_cuda_stream_handle(ggml_backend_t backend, int stream_no);
GGML_BACKEND_API void * ggml_backend_cuda_timing_event_new(ggml_backend_t backend, int stream_no);
GGML_BACKEND_API void   ggml_backend_cuda_timing_event_record(void * handle);
GGML_BACKEND_API double ggml_backend_cuda_timing_event_elapsed_ms(void * a, void * b); // < 0 = not ready
GGML_BACKEND_API void   ggml_backend_cuda_timing_event_free(void * handle);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
