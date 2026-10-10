#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_moe_hybrid_publish(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_moe_hybrid_wait_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
