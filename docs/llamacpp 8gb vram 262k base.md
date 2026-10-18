2026-10-10 17:57:13 PS C:\Users\<user>>  C:\Development\AI\llamacppCUDA\llama-server.exe `
>>    --host 0.0.0.0 --port 8033 `
>>    --model c:\Models\ALL\other\qwen-3.6\Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf `
>>    -ctk q8_0 -ctv q8_0 `
>>    --flash-attn on `
>>    --parallel 1 `
>>    --batch-size 8192 `
>>    --ubatch-size 512 `
>>    --alias "Qwen3.6-35B-A3B-MTP-UD-Q4_K_S" `
>>    --jinja `
>>    -c 262000 `
>>    --log-prefix `
>>    --log-timestamps `
>>    -lv 4 --fit-target 16000 `
>>
0.00.068.867 I cmn  common_param: common_params_print_info: build 10892 (e5a8d439c) with MSVC 19.44.35224.0 for x64
0.00.068.879 I cmn  common_param: common_params_print_info: verbosity = 4 (adjust with the `-lv N` CLI arg)
0.00.068.879 I cmn  common_param: device_info:
0.00.239.797 I cmn  common_param:   - CUDA0   : NVIDIA GeForce RTX 5090 Laptop GPU (24462 MiB, 23119 MiB free)
0.00.239.814 I cmn  common_param:   - CPU     : AMD Ryzen 9 9955HX 16-Core Processor            (63166 MiB, 48901 MiB free)
0.00.239.910 I cmn  common_param: system_info: n_threads = 16 (n_threads_batch = 16) / 32 | CUDA : ARCHS = 1200 | USE_GRAPHS = 1 | FA_QUANTS = q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16 | BLACKWELL_NATIVE_FP4 = 1 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | AVX512 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |
0.00.239.966 I srv          init: running without SSL
0.00.240.068 I srv          init: using 31 threads for HTTP server
0.00.240.589 W srv  llama_server: -----------------
0.00.240.595 W srv  llama_server: CORS is set to allow all origins ('*') and no API key is set
0.00.240.595 W srv  llama_server: this can be a security risk (cross-origin attacks)
0.00.240.596 W srv  llama_server: more info: https://github.com/ggml-org/llama.cpp/pull/25655
0.00.240.596 W srv  llama_server: -----------------
0.00.240.629 I srv         start: binding port with default address family
0.00.250.600 I srv    load_model: loading model 'c:\Models\ALL\other\qwen-3.6\Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
0.00.250.611 I srv    load_model: local path 'c:\Models\ALL\other\qwen-3.6\Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
0.00.250.629 I cmn  common_init_: fitting params to device memory ...
0.00.250.630 I cmn  common_init_: (for bugs during this step try to reproduce them with -fit off, or provide --verbose logs if the bug only occurs with -fit on)
0.00.250.684 I common_params_fit_impl: getting device memory data for initial parameters:
0.00.900.315 I common_memory_breakdown_print: | memory breakdown [MiB]          | total    free     self   model   context   compute    unaccounted |
0.00.900.327 I common_memory_breakdown_print: |   - CUDA0 (RTX 5090 Laptop GPU) | 24462 = 23055 + (23002 = 19399 +    2782 +     820) +      -21594 |
0.00.900.327 I common_memory_breakdown_print: |   - Host                        |                    779 =   515 +       0 +     264                |
0.00.972.113 I common_params_fit_impl: projected to use 23002 MiB of device memory vs. 23055 MiB of free device memory
0.00.972.123 I common_params_fit_impl: cannot meet free memory target of 16000 MiB, need to reduce device memory by 15947 MiB
0.00.972.124 I common_params_fit_impl: context size set by user to 262000 -> no change
0.00.972.129 I common_params_fit_impl: getting device memory data with all MoE tensors moved to system memory:
0.01.634.473 I common_memory_breakdown_print: | memory breakdown [MiB]          | total    free     self   model   context   compute    unaccounted |
0.01.634.483 I common_memory_breakdown_print: |   - CUDA0 (RTX 5090 Laptop GPU) | 24462 = 23055 + ( 5664 =  1921 +    2782 +     960) +       -4256 |
0.01.634.483 I common_memory_breakdown_print: |   - Host                        |                  18257 = 17993 +       0 +     264                |
0.01.706.863 I common_params_fit_impl: with only dense weights in device memory there is a total surplus of 1390 MiB
0.01.706.880 I common_params_fit_impl: id=0, target=7055 MiB
0.02.371.745 I common_memory_breakdown_print: | memory breakdown [MiB]          | total    free     self   model   context   compute    unaccounted |
0.02.371.756 I common_memory_breakdown_print: |   - CUDA0 (RTX 5090 Laptop GPU) | 24462 = 23119 + ( 1228 =     0 +       0 +    1228) +         115 |
0.02.371.757 I common_memory_breakdown_print: |   - Host                        |                  22973 = 19914 +    2782 +     276                |
0.02.445.393 I common_params_fit_impl: memory for test allocation by device:
0.02.445.402 I common_params_fit_impl: id=0, n_layer= 0, n_part= 0, overflow_type=4, mem=  1228 MiB
0.02.445.413 I common_params_fit_impl: filling dense-only layers back-to-front:
0.03.199.941 I common_memory_breakdown_print: | memory breakdown [MiB]          | total    free     self   model   context   compute    unaccounted |
0.03.199.950 I common_memory_breakdown_print: |   - CUDA0 (RTX 5090 Laptop GPU) | 24462 = 23057 + ( 5630 =  1881 +    2780 +     968) +       -4225 |
0.03.199.951 I common_memory_breakdown_print: |   - Host                        |                  18311 = 18033 +       2 +     276                |
0.03.272.208 I common_params_fit_impl: memory for test allocation by device:
0.03.272.219 I common_params_fit_impl: id=0, n_layer=41, n_part=40, overflow_type=4, mem=  5630 MiB
0.03.272.224 I common_params_fit_impl: set ngl_per_device[0].n_layer=41
0.03.272.226 I common_params_fit_impl:   - CUDA0 (NVIDIA GeForce RTX 5090 Laptop GPU): 41 layers,   5630 MiB used,  17424 MiB free
0.03.272.228 I common_params_fit_impl: converting dense-only layers to full layers and filling them front-to-back with overflow to next device/system memory:
0.03.925.383 I common_memory_breakdown_print: | memory breakdown [MiB]          | total    free     self   model   context   compute    unaccounted |
0.03.925.393 I common_memory_breakdown_print: |   - CUDA0 (RTX 5090 Laptop GPU) | 24462 = 23057 + (22528 = 18927 +    2780 +     820) +      -21123 |
0.03.925.394 I common_memory_breakdown_print: |   - Host                        |                   1265 =   987 +       2 +     276                |
0.03.997.144 I common_params_fit_impl: memory for test allocation by device:
0.03.997.157 I common_params_fit_impl: id=0, n_layer=41, n_part= 0, overflow_type=4, mem= 22528 MiB
0.04.861.586 I common_memory_breakdown_print: | memory breakdown [MiB]          | total    free     self   model   context   compute    unaccounted |
0.04.861.598 I common_memory_breakdown_print: |   - CUDA0 (RTX 5090 Laptop GPU) | 24462 = 23057 + ( 6778 =  3177 +    2780 +     820) +       -5373 |
0.04.861.599 I common_memory_breakdown_print: |   - Host                        |                  17015 = 16737 +       2 +     276                |
0.04.935.093 I common_params_fit_impl: memory for test allocation by device:
0.04.935.105 I common_params_fit_impl: id=0, n_layer=41, n_part=37, overflow_type=4, mem=  6778 MiB
0.04.935.110 I common_params_fit_impl: set ngl_per_device[0].(n_layer, n_part)=(41, 37), id_dense_start=0
0.05.798.172 I common_memory_breakdown_print: | memory breakdown [MiB]          | total    free     self   model   context   compute    unaccounted |
0.05.798.181 I common_memory_breakdown_print: |   - CUDA0 (RTX 5090 Laptop GPU) | 24462 = 23057 + ( 7210 =  3609 +    2780 +     820) +       -5805 |
0.05.798.182 I common_memory_breakdown_print: |   - Host                        |                  16583 = 16305 +       2 +     276                |
0.05.870.020 I common_params_fit_impl: memory for test allocation by device:
0.05.870.032 I common_params_fit_impl: id=0, n_layer=41, n_part=36, overflow_type=4, mem=  7210 MiB
0.05.870.042 I common_params_fit_impl: set ngl_per_device_high[0].(n_layer, n_part)=(41, 36), id_dense_start_high=0
0.05.870.045 I common_params_fit_impl: trying to fit one extra layer with overflow_type=LAYER_FRACTION_UP
0.06.623.946 I common_memory_breakdown_print: | memory breakdown [MiB]          | total    free     self   model   context   compute    unaccounted |
0.06.623.957 I common_memory_breakdown_print: |   - CUDA0 (RTX 5090 Laptop GPU) | 24462 = 23057 + ( 6918 =  3317 +    2780 +     820) +       -5513 |
0.06.623.958 I common_memory_breakdown_print: |   - Host                        |                  16875 = 16597 +       2 +     276                |
0.06.696.837 I common_params_fit_impl: memory for test allocation by device:
0.06.696.849 I common_params_fit_impl: id=0, n_layer=41, n_part=37, overflow_type=2, mem=  6918 MiB
0.06.696.854 I common_params_fit_impl: set ngl_per_device[0].(n_layer, n_part, overflow_type)=(41, 37, UP), id_dense_start=0
0.06.696.855 I common_params_fit_impl: trying to fit one extra layer with overflow_type=LAYER_FRACTION_GATE
0.07.450.638 I common_memory_breakdown_print: | memory breakdown [MiB]          | total    free     self   model   context   compute    unaccounted |
0.07.450.647 I common_memory_breakdown_print: |   - CUDA0 (RTX 5090 Laptop GPU) | 24462 = 23057 + ( 7065 =  3464 +    2780 +     820) +       -5660 |
0.07.450.648 I common_memory_breakdown_print: |   - Host                        |                  16728 = 16450 +       2 +     276                |
0.07.524.180 I common_params_fit_impl: memory for test allocation by device:
0.07.524.191 I common_params_fit_impl: id=0, n_layer=41, n_part=37, overflow_type=3, mem=  7065 MiB
0.07.524.198 I common_params_fit_impl:   - CUDA0 (NVIDIA GeForce RTX 5090 Laptop GPU): 41 layers (37 overflowing),   6918 MiB used,  16136 MiB free
0.07.524.239 I common_fit_params: successfully fit params to free device memory
0.07.524.245 I common_fit_params: fitting params to free memory took 7.27 seconds
0.07.640.116 I llama_model_loader: loaded meta data with 55 key-value pairs and 753 tensors from c:\Models\ALL\other\qwen-3.6\Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf (version GGUF V3 (latest))
0.07.640.152 I llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
0.07.640.158 I llama_model_loader: - kv   0:                       general.architecture str              = qwen35moe
0.07.640.160 I llama_model_loader: - kv   1:                               general.type str              = model
0.07.640.162 I llama_model_loader: - kv   2:                     general.sampling.top_k i32              = 20
0.07.640.171 I llama_model_loader: - kv   3:                     general.sampling.top_p f32              = 0.950000
0.07.640.173 I llama_model_loader: - kv   4:                      general.sampling.temp f32              = 1.000000
0.07.640.175 I llama_model_loader: - kv   5:                               general.name str              = Qwen3.6-35B-A3B
0.07.640.175 I llama_model_loader: - kv   6:                           general.basename str              = Qwen3.6-35B-A3B
0.07.640.176 I llama_model_loader: - kv   7:                       general.quantized_by str              = Unsloth
0.07.640.177 I llama_model_loader: - kv   8:                         general.size_label str              = 35B-A3B
0.07.640.178 I llama_model_loader: - kv   9:                            general.license str              = apache-2.0
0.07.640.182 I llama_model_loader: - kv  10:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3.6-3...
0.07.640.184 I llama_model_loader: - kv  11:                           general.repo_url str              = https://huggingface.co/unsloth
0.07.640.185 I llama_model_loader: - kv  12:                   general.base_model.count u32              = 1
0.07.640.186 I llama_model_loader: - kv  13:                  general.base_model.0.name str              = Qwen3.6 35B A3B
0.07.640.187 I llama_model_loader: - kv  14:          general.base_model.0.organization str              = Qwen
0.07.640.189 I llama_model_loader: - kv  15:              general.base_model.0.repo_url str              = https://huggingface.co/Qwen/Qwen3.6-3...
0.07.640.208 I llama_model_loader: - kv  16:                               general.tags arr[str,3]       = ["qwen3_5_moe", "qwen", "image-text-t...
0.07.640.209 I llama_model_loader: - kv  17:                      qwen35moe.block_count u32              = 41
0.07.640.210 I llama_model_loader: - kv  18:                   qwen35moe.context_length u32              = 262144
0.07.640.211 I llama_model_loader: - kv  19:                 qwen35moe.embedding_length u32              = 2048
0.07.640.212 I llama_model_loader: - kv  20:             qwen35moe.attention.head_count u32              = 16
0.07.640.212 I llama_model_loader: - kv  21:          qwen35moe.attention.head_count_kv u32              = 2
0.07.640.216 I llama_model_loader: - kv  22:          qwen35moe.rope.dimension_sections arr[i32,4]       = [11, 11, 10, 0]
0.07.640.219 I llama_model_loader: - kv  23:                   qwen35moe.rope.freq_base f32              = 10000000.000000
0.07.640.222 I llama_model_loader: - kv  24: qwen35moe.attention.layer_norm_rms_epsilon f32              = 0.000001
0.07.640.223 I llama_model_loader: - kv  25:                     qwen35moe.expert_count u32              = 256
0.07.640.223 I llama_model_loader: - kv  26:                qwen35moe.expert_used_count u32              = 8
0.07.640.224 I llama_model_loader: - kv  27:             qwen35moe.attention.key_length u32              = 256
0.07.640.225 I llama_model_loader: - kv  28:           qwen35moe.attention.value_length u32              = 256
0.07.640.226 I llama_model_loader: - kv  29:       qwen35moe.expert_feed_forward_length u32              = 512
0.07.640.226 I llama_model_loader: - kv  30: qwen35moe.expert_shared_feed_forward_length u32              = 512
0.07.640.227 I llama_model_loader: - kv  31:                  qwen35moe.ssm.conv_kernel u32              = 4
0.07.640.228 I llama_model_loader: - kv  32:                   qwen35moe.ssm.state_size u32              = 128
0.07.640.229 I llama_model_loader: - kv  33:                  qwen35moe.ssm.group_count u32              = 16
0.07.640.229 I llama_model_loader: - kv  34:               qwen35moe.ssm.time_step_rank u32              = 32
0.07.640.230 I llama_model_loader: - kv  35:                   qwen35moe.ssm.inner_size u32              = 4096
0.07.640.231 I llama_model_loader: - kv  36:          qwen35moe.full_attention_interval u32              = 4
0.07.640.232 I llama_model_loader: - kv  37:             qwen35moe.rope.dimension_count u32              = 64
0.07.640.232 I llama_model_loader: - kv  38:             qwen35moe.nextn_predict_layers u32              = 1
0.07.640.233 I llama_model_loader: - kv  39:                       tokenizer.ggml.model str              = gpt2
0.07.640.234 I llama_model_loader: - kv  40:                         tokenizer.ggml.pre str              = qwen35
0.07.711.584 I llama_model_loader: - kv  41:                      tokenizer.ggml.tokens arr[str,248320]  = ["!", "\"", "#", "$", "%", "&", "'", ...
0.07.734.561 I llama_model_loader: - kv  42:                  tokenizer.ggml.token_type arr[i32,248320]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
0.07.805.975 I llama_model_loader: - kv  43:                      tokenizer.ggml.merges arr[str,247587]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
0.07.805.986 I llama_model_loader: - kv  44:                tokenizer.ggml.eos_token_id u32              = 248046
0.07.805.988 I llama_model_loader: - kv  45:            tokenizer.ggml.padding_token_id u32              = 248055
0.07.805.990 I llama_model_loader: - kv  46:                tokenizer.ggml.bos_token_id u32              = 248044
0.07.805.992 I llama_model_loader: - kv  47:               tokenizer.ggml.add_bos_token bool             = false
0.07.806.001 I llama_model_loader: - kv  48:                    tokenizer.chat_template str              = {%- set image_count = namespace(value...
0.07.806.002 I llama_model_loader: - kv  49:               general.quantization_version u32              = 2
0.07.806.004 I llama_model_loader: - kv  50:                          general.file_type u32              = 14
0.07.806.006 I llama_model_loader: - kv  51:                      quantize.imatrix.file str              = Qwen3.6-35B-A3B-GGUF/imatrix_unsloth....
0.07.806.009 I llama_model_loader: - kv  52:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3.6-35B-A3B.txt
0.07.806.010 I llama_model_loader: - kv  53:             quantize.imatrix.entries_count u32              = 510
0.07.806.012 I llama_model_loader: - kv  54:              quantize.imatrix.chunks_count u32              = 77
0.07.806.018 I llama_model_loader: - type  f32:  368 tensors
0.07.806.020 I llama_model_loader: - type q8_0:  259 tensors
0.07.806.020 I llama_model_loader: - type q4_K:  120 tensors
0.07.806.022 I llama_model_loader: - type q6_K:    4 tensors
0.07.806.023 I llama_model_loader: - type bf16:    2 tensors
0.07.806.028 I print_info: file format = GGUF V3 (latest)
0.07.806.030 I print_info: file type   = Q4_K - Small
0.07.806.037 I print_info: file size   = 19.91 GiB (4.82 BPW)
0.07.806.120 I llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 5090 Laptop GPU) (0000:01:00.0) - 23119 MiB free
0.08.002.379 I load: 0 unused tokens
0.08.048.319 I load: printing all EOG tokens:
0.08.048.327 I load:   - 248044 ('<|endoftext|>')
0.08.048.328 I load:   - 248046 ('<|im_end|>')
0.08.048.328 I load:   - 248063 ('<|fim_pad|>')
0.08.048.329 I load:   - 248064 ('<|repo_name|>')
0.08.048.329 I load:   - 248065 ('<|file_sep|>')
0.08.048.617 I load: special tokens cache size = 33
0.08.127.358 I load: token to piece cache size = 1.7581 MB
0.08.127.380 I print_info: arch                  = qwen35moe
0.08.127.381 I print_info: vocab_only            = 0
0.08.127.381 I print_info: no_alloc              = 0
0.08.127.382 I print_info: n_ctx_train           = 262144
0.08.127.383 I print_info: n_embd_inp            = 2048
0.08.127.383 I print_info: n_embd                = 2048
0.08.127.384 I print_info: n_embd_out            = 2048
0.08.127.385 I print_info: n_layer               = 40
0.08.127.385 I print_info: n_layer_all           = 41
0.08.127.401 I print_info: n_head                = 16
0.08.127.405 I print_info: n_head_kv             = 2
0.08.127.405 I print_info: n_rot                 = 64
0.08.127.406 I print_info: n_swa                 = 0
0.08.127.407 I print_info: is_swa_any            = 0
0.08.127.407 I print_info: non_causal_type       = 0
0.08.127.408 I print_info: n_embd_head_k         = 256
0.08.127.408 I print_info: n_embd_head_v         = 256
0.08.127.411 I print_info: n_gqa                 = 8
0.08.127.415 I print_info: n_embd_k_gqa          = 512
0.08.127.418 I print_info: n_embd_v_gqa          = 512
0.08.127.421 I print_info: f_norm_eps            = 0.0e+00
0.08.127.423 I print_info: f_norm_rms_eps        = 1.0e-06
0.08.127.424 I print_info: f_clamp_kqv           = 0.0e+00
0.08.127.425 I print_info: f_max_alibi_bias      = 0.0e+00
0.08.127.425 I print_info: f_logit_scale         = 0.0e+00
0.08.127.426 I print_info: f_attn_scale          = 0.0e+00
0.08.127.427 I print_info: f_attn_value_scale    = 0.0000
0.08.127.430 I print_info: n_ff                  = 0
0.08.127.431 I print_info: n_expert              = 256
0.08.127.431 I print_info: n_expert_used         = 8
0.08.127.432 I print_info: n_expert_groups       = 0
0.08.127.432 I print_info: n_group_used          = 0
0.08.127.433 I print_info: causal attn           = 1
0.08.127.434 I print_info: pooling type          = -1
0.08.127.434 I print_info: rope type             = 40
0.08.127.435 I print_info: rope scaling          = linear
0.08.127.436 I print_info: freq_base_train       = 10000000.0
0.08.127.439 I print_info: freq_scale_train      = 1
0.08.127.440 I print_info: n_ctx_orig_yarn       = 262144
0.08.127.441 I print_info: rope_yarn_log_mul     = 0.0000
0.08.127.441 I print_info: rope_finetuned        = unknown
0.08.127.442 I print_info: mrope sections        = [11, 11, 10, 0]
0.08.127.443 I print_info: ssm_d_conv            = 4
0.08.127.444 I print_info: ssm_d_inner           = 4096
0.08.127.444 I print_info: ssm_d_state           = 128
0.08.127.444 I print_info: ssm_dt_rank           = 32
0.08.127.445 I print_info: ssm_n_group           = 16
0.08.127.445 I print_info: ssm_dt_b_c_rms        = 0
0.08.127.447 I print_info: model type            = 35B.A3B
0.08.127.448 I print_info: model params          = 35.51 B
0.08.127.449 I print_info: general.name          = Qwen3.6-35B-A3B
0.08.127.451 I print_info: vocab type            = BPE
0.08.127.451 I print_info: n_vocab               = 248320
0.08.127.452 I print_info: n_merges              = 247587
0.08.127.453 I print_info: BOS token             = 248044 '<|endoftext|>'
0.08.127.453 I print_info: EOS token             = 248046 '<|im_end|>'
0.08.127.454 I print_info: EOT token             = 248046 '<|im_end|>'
0.08.127.455 I print_info: PAD token             = 248055 '<|vision_pad|>'
0.08.127.456 I print_info: LF token              = 198 'Ċ'
0.08.127.456 I print_info: FIM PRE token         = 248060 '<|fim_prefix|>'
0.08.127.457 I print_info: FIM SUF token         = 248062 '<|fim_suffix|>'
0.08.127.457 I print_info: FIM MID token         = 248061 '<|fim_middle|>'
0.08.127.458 I print_info: FIM PAD token         = 248063 '<|fim_pad|>'
0.08.127.459 I print_info: FIM REP token         = 248064 '<|repo_name|>'
0.08.127.459 I print_info: FIM SEP token         = 248065 '<|file_sep|>'
0.08.127.460 I print_info: EOG token             = 248044 '<|endoftext|>'
0.08.127.461 I print_info: EOG token             = 248046 '<|im_end|>'
0.08.127.462 I print_info: EOG token             = 248063 '<|fim_pad|>'
0.08.127.462 I print_info: EOG token             = 248064 '<|repo_name|>'
0.08.127.463 I print_info: EOG token             = 248065 '<|file_sep|>'
0.08.127.463 I print_info: max token length      = 256
0.08.127.506 I load_tensors: loading model tensors, this can take a while... (load_mode = mmap)
0.08.140.868 W llama_model_loader: tensor overrides to CPU are used with mmap enabled - consider using --load-mode none for better performance
0.08.231.486 W model has unused tensor blk.40.attn_norm.weight (size = 8192 bytes) -- ignoring
0.08.231.506 W model has unused tensor blk.40.post_attention_norm.weight (size = 8192 bytes) -- ignoring
0.08.231.544 W model has unused tensor blk.40.attn_q.weight (size = 17825792 bytes) -- ignoring
0.08.231.555 W model has unused tensor blk.40.attn_k.weight (size = 1114112 bytes) -- ignoring
0.08.231.567 W model has unused tensor blk.40.attn_v.weight (size = 1114112 bytes) -- ignoring
0.08.231.609 W model has unused tensor blk.40.attn_output.weight (size = 8912896 bytes) -- ignoring
0.08.231.621 W model has unused tensor blk.40.attn_q_norm.weight (size = 1024 bytes) -- ignoring
0.08.231.633 W model has unused tensor blk.40.attn_k_norm.weight (size = 1024 bytes) -- ignoring
0.08.231.646 W model has unused tensor blk.40.ffn_gate_inp.weight (size = 1048576 bytes) -- ignoring
0.08.231.658 W model has unused tensor blk.40.ffn_down_exps.weight (size = 150994944 bytes) -- ignoring
0.08.231.683 W model has unused tensor blk.40.ffn_gate_exps.weight (size = 150994944 bytes) -- ignoring
0.08.231.695 W model has unused tensor blk.40.ffn_up_exps.weight (size = 150994944 bytes) -- ignoring
0.08.231.709 W model has unused tensor blk.40.ffn_gate_inp_shexp.weight (size = 4096 bytes) -- ignoring
0.08.231.722 W model has unused tensor blk.40.ffn_gate_shexp.weight (size = 1114112 bytes) -- ignoring
0.08.231.735 W model has unused tensor blk.40.ffn_up_shexp.weight (size = 1114112 bytes) -- ignoring
0.08.231.748 W model has unused tensor blk.40.ffn_down_shexp.weight (size = 1114112 bytes) -- ignoring
0.08.231.764 W model has unused tensor blk.40.nextn.eh_proj.weight (size = 8912896 bytes) -- ignoring
0.08.231.778 W model has unused tensor blk.40.nextn.enorm.weight (size = 8192 bytes) -- ignoring
0.08.231.792 W model has unused tensor blk.40.nextn.hnorm.weight (size = 8192 bytes) -- ignoring
0.08.231.833 W model has unused tensor blk.40.nextn.shared_head_norm.weight (size = 8192 bytes) -- ignoring
0.10.827.764 I load_tensors: offloading output layer to GPU
0.10.827.776 I load_tensors: offloading 40 repeating layers to GPU
0.10.827.777 I load_tensors: offloaded 41/42 layers to GPU
0.10.827.785 I load_tensors:   CPU_Mapped model buffer size = 19515.72 MiB
0.10.827.793 I load_tensors:        CUDA0 model buffer size =  3317.37 MiB
0.13.769.789 I cmn  common_init_: added <|endoftext|> logit bias = -inf
0.13.769.802 I cmn  common_init_: added <|im_end|> logit bias = -inf
0.13.769.805 I cmn  common_init_: added <|fim_pad|> logit bias = -inf
0.13.769.806 I cmn  common_init_: added <|repo_name|> logit bias = -inf
0.13.769.807 I cmn  common_init_: added <|file_sep|> logit bias = -inf
0.13.770.817 I llama_context: constructing llama_context
0.13.770.830 I llama_context: n_seq_max             = 1
0.13.770.831 I llama_context: n_ctx                 = 262144
0.13.770.832 I llama_context: n_ctx_seq             = 262144
0.13.770.833 I llama_context: n_batch               = 8192
0.13.770.833 I llama_context: n_ubatch              = 512
0.13.770.834 I llama_context: causal_attn           = 1
0.13.770.836 I llama_context: flash_attn            = enabled
0.13.770.836 I llama_context: kv_unified            = false
0.13.770.842 I llama_context: freq_base             = 10000000.0
0.13.770.845 I llama_context: freq_scale            = 1
0.13.770.846 I llama_context: n_rs_seq              = 0
0.13.770.847 I llama_context: n_outputs_max         = 1
0.13.770.848 I llama_context: n_outputs_max_per_seq = 1
0.13.771.814 I llama_context:  CUDA_Host  output buffer size =     0.95 MiB
0.13.782.913 I llama_kv_cache:      CUDA0 KV buffer size =  2720.00 MiB
0.13.965.181 I llama_kv_cache: size = 2720.00 MiB (262144 cells,  10 layers,  1/1 seqs), K (q8_0): 1360.00 MiB, V (q8_0): 1360.00 MiB
0.13.965.202 I llama_kv_cache: attn_rot_k = 1, n_embd_head_k_all = 256
0.13.965.203 I llama_kv_cache: attn_rot_v = 1, n_embd_head_k_all = 256
0.13.965.862 I llama_memory_recurrent:        CPU RS buffer size =     2.09 MiB
0.13.975.422 I llama_memory_recurrent:      CUDA0 RS buffer size =    60.72 MiB
0.13.975.443 I llama_memory_recurrent: size =   62.81 MiB (     1 cells,  40 layers,  1 seqs  0 rs_seq), R (f32):    2.81 MiB, S (f32):   60.00 MiB, P (f32):    0.00 MiB
0.13.975.454 I sched_reserve: reserving ...
0.13.978.995 I resolve_fused_ops: resolving fused DeepSeek V4 HC support:
0.13.981.272 I resolve_fused_ops: fused DeepSeek V4 HC pre enabled
0.13.983.528 I resolve_fused_ops: fused DeepSeek V4 HC comb enabled
0.13.985.385 I resolve_fused_ops: fused DeepSeek V4 HC post enabled
0.14.038.451 I sched_reserve:      CUDA0 compute buffer size =   820.53 MiB
0.14.038.465 I sched_reserve:  CUDA_Host compute buffer size =   276.38 MiB
0.14.038.466 I sched_reserve: graph nodes  = 3907
0.14.038.471 I sched_reserve: graph splits = 135 (with bs=512), 80 (with bs=1)
0.14.038.473 I sched_reserve: reserve took 63.02 ms, sched copies = 1
0.14.038.989 I cmn          init: llama threadpool init, n_threads = 16
0.14.039.020 I cmn  common_init_: warming up the model with an empty run - please wait ... (--no-warmup to disable)
0.14.825.846 I cmn  common_conte: the context does not support partial sequence removal
0.14.832.137 I srv    load_model: speculative decoding will use checkpoints
0.14.832.144 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 262144, kv_unified = 'false'
0.14.832.168 I spec common_specu: no implementations specified for speculative decoding
0.14.832.174 I slot   load_model: id  0 | task -1 | new slot, n_ctx = 262144
0.14.832.683 I srv    load_model: prompt cache is enabled, size limit: 8192 MiB
0.14.832.683 I srv    load_model: use `--cache-ram 0` to disable the prompt cache
0.14.832.685 I srv    load_model: for more info see https://github.com/ggml-org/llama.cpp/pull/16391
0.14.832.686 I srv    load_model: context checkpoints enabled, max = 32, min spacing = 8192
0.14.832.711 I srv          init: idle slots will be saved to prompt cache upon starting a new task
0.14.845.356 I srv          init: init: chat template, example_format: '<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
<think>

</think>

Hi there<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
<think>
'
0.14.846.791 I srv          init: init: chat template, thinking = 1
0.14.846.808 I srv          init: preserve_reasoning kwarg: true
0.14.846.809 W srv          init: chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)
0.14.846.837 I srv  llama_server: model loaded
0.14.846.841 I srv  llama_server: listening on http://0.0.0.0:8033
0.14.846.954 I srv  update_slots: all slots are idle
0.21.854.874 I srv  server_strea: conv_id=ce521e50-156f-4e27-b7d6-4548050e8617 (empty=0)
0.21.856.597 I srv   operator (): chat format: peg-native
0.21.857.167 I slot get_availabl: id  0 | task -1 |  - skipping, slot is empty
0.21.857.182 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = -1
0.21.857.183 I srv  get_availabl: updating prompt cache
0.21.857.198 I srv          load:  - looking for better prompt, base f_keep = -1.000, f_sim = 0.000
0.21.857.208 I srv        update:  - cache state: 0 prompts, 0.000 MiB (limits: 8192.000 MiB, 262144 tokens, 8589934592 est)
0.21.857.210 I srv  get_availabl: prompt cache update took 0.02 ms
0.21.857.428 I cmn  common_reaso: activated, budget=2147483647 tokens
0.21.857.474 I slot launch_slot_: id  0 | task -1 | sampler chain: logits -> ?penalties -> ?dry -> ?top-n-sigma -> top-k -> ?typical -> top-p -> min-p -> ?xtc -> ?temp-ext -> dist
0.21.857.495 I slot launch_slot_: id  0 | task -1 | sampler params:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        dry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = 64
        top_k = 20, top_p = 0.950, min_p = 0.050, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, top_n_sigma = -1.000, temp = 1.000
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000, adaptive_target = -1.000, adaptive_decay = 0.900
0.21.857.498 I slot launch_slot_: id  0 | task 0 | processing task, is_child = 0
0.21.857.521 I slot  operator (): id  0 | task 0 | new prompt, n_ctx_slot = 262144, n_keep = 0, task.n_tokens = 367
0.21.857.545 I slot  operator (): id  0 | task 0 | cached n_tokens = 0, memory_seq_rm [0, end)
0.32.537.532 I slot print_timing: id  0 | task 0 | prompt processing, n_tokens =    353, progress = 0.96, t =  10.68 s / 33.05 tokens per second
0.32.537.541 I slot  operator (): id  0 | task 0 | cached n_tokens = 353, memory_seq_rm [353, end)
0.32.573.195 I slot create_check: id  0 | task 0 | created context checkpoint 1 of 32 (pos_min = 352, pos_max = 352, n_tokens = 353, size = 62.813 MiB)
0.32.838.860 I slot print_timing: id  0 | task 0 | prompt processing, n_tokens =    363, progress = 0.99, t =  10.98 s / 33.06 tokens per second
0.32.838.870 I slot  operator (): id  0 | task 0 | cached n_tokens = 363, memory_seq_rm [363, end)
0.32.839.250 I slot init_sampler: id  0 | task 0 | init sampler, took 0.10 ms, tokens: text = 367, total = 367
0.32.877.674 I slot create_check: id  0 | task 0 | created context checkpoint 2 of 32 (pos_min = 362, pos_max = 362, n_tokens = 363, size = 62.813 MiB)
0.36.687.706 I slot print_timing: id  0 | task 0 | n_gen =    100, tg =  27.30 t/s, tg_3s =  27.57 t/s
0.39.702.300 I slot print_timing: id  0 | task 0 | n_gen =    191, tg =  28.61 t/s, tg_3s =  30.19 t/s
0.42.708.355 I slot print_timing: id  0 | task 0 | n_gen =    285, tg =  29.44 t/s, tg_3s =  31.27 t/s
0.42.935.229 I cmn  common_reaso: deactivated (natural end)
0.43.413.906 I slot print_timing: id  0 | task 0 | prompt eval time =   11203.67 ms /   367 tokens (   30.53 ms per token,    32.76 tokens per second)
0.43.413.915 I slot print_timing: id  0 | task 0 |        eval time =   10352.69 ms /   307 tokens (   33.83 ms per token,    29.56 tokens per second)
0.43.413.917 I slot print_timing: id  0 | task 0 |       total time =   21556.36 ms /   674 tokens
0.43.413.918 I slot print_timing: id  0 | task 0 |    graphs reused =        305
0.43.414.003 I slot      release: id  0 | task 0 | stop processing: n_tokens = 673, truncated = 0
0.43.414.023 I srv  update_slots: all slots are idle
0.43.460.156 I srv  server_strea: conv_id=ce521e50-156f-4e27-b7d6-4548050e8617 (empty=0)
0.43.462.948 I srv   operator (): chat format: peg-native
0.43.463.435 I slot get_availabl: id  0 | task -1 |  - checking sim = 0.916 (673/735) > 0.100
0.43.463.445 I slot get_availabl: id  0 | task -1 | selected slot by LCP similarity, f_sim_best = 0.916 (> 0.100 thold), f_keep = 1.000
0.43.463.626 I cmn  common_reaso: activated, budget=2147483647 tokens
0.43.463.929 I slot launch_slot_: id  0 | task -1 | sampler chain: logits -> ?penalties -> ?dry -> ?top-n-sigma -> top-k -> ?typical -> top-p -> min-p -> ?xtc -> ?temp-ext -> dist
0.43.463.952 I slot launch_slot_: id  0 | task -1 | sampler params:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        dry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = 64
        top_k = 20, top_p = 0.950, min_p = 0.050, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, top_n_sigma = -1.000, temp = 1.000
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000, adaptive_target = -1.000, adaptive_decay = 0.900
0.43.463.956 I slot launch_slot_: id  0 | task 310 | processing task, is_child = 0
0.43.463.969 I slot  operator (): id  0 | task 310 | new prompt, n_ctx_slot = 262144, n_keep = 0, task.n_tokens = 735
0.43.463.984 I slot  operator (): id  0 | task 310 | cached n_tokens = 673, memory_seq_rm [673, end)
0.43.487.067 I slot create_check: id  0 | task 310 | created context checkpoint 3 of 32 (pos_min = 672, pos_max = 672, n_tokens = 673, size = 62.813 MiB)
0.44.500.500 I slot  operator (): id  0 | task 310 | cached n_tokens = 731, memory_seq_rm [731, end)
0.44.500.958 I slot init_sampler: id  0 | task 310 | init sampler, took 0.17 ms, tokens: text = 735, total = 735
0.44.522.920 I slot create_check: id  0 | task 310 | created context checkpoint 4 of 32 (pos_min = 730, pos_max = 730, n_tokens = 731, size = 62.813 MiB)
0.45.600.126 I cmn  common_reaso: deactivated (natural end)
0.46.102.378 I slot print_timing: id  0 | task 310 | prompt eval time =    1140.98 ms /    62 tokens (   18.40 ms per token,    54.34 tokens per second)
0.46.102.389 I slot print_timing: id  0 | task 310 |        eval time =    1497.39 ms /    43 tokens (   35.65 ms per token,    28.05 tokens per second)
0.46.102.391 I slot print_timing: id  0 | task 310 |       total time =    2638.37 ms /   105 tokens
0.46.102.392 I slot print_timing: id  0 | task 310 |    graphs reused =        345
0.46.102.477 I slot      release: id  0 | task 310 | stop processing: n_tokens = 777, truncated = 0
0.46.102.530 I srv  update_slots: all slots are idle
0.46.145.762 I srv  server_strea: conv_id=ce521e50-156f-4e27-b7d6-4548050e8617 (empty=0)
0.46.148.563 I srv   operator (): chat format: peg-native
0.46.149.038 I slot get_availabl: id  0 | task -1 |  - checking sim = 0.926 (777/839) > 0.100
0.46.149.054 I slot get_availabl: id  0 | task -1 | selected slot by LCP similarity, f_sim_best = 0.926 (> 0.100 thold), f_keep = 1.000
0.46.149.202 I cmn  common_reaso: activated, budget=2147483647 tokens
0.46.149.464 I slot launch_slot_: id  0 | task -1 | sampler chain: logits -> ?penalties -> ?dry -> ?top-n-sigma -> top-k -> ?typical -> top-p -> min-p -> ?xtc -> ?temp-ext -> dist
0.46.149.481 I slot launch_slot_: id  0 | task -1 | sampler params:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        dry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = 64
        top_k = 20, top_p = 0.950, min_p = 0.050, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, top_n_sigma = -1.000, temp = 1.000
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000, adaptive_target = -1.000, adaptive_decay = 0.900
0.46.149.484 I slot launch_slot_: id  0 | task 355 | processing task, is_child = 0
0.46.149.497 I slot  operator (): id  0 | task 355 | new prompt, n_ctx_slot = 262144, n_keep = 0, task.n_tokens = 839
0.46.149.512 I slot  operator (): id  0 | task 355 | cached n_tokens = 777, memory_seq_rm [777, end)
0.46.173.345 I slot create_check: id  0 | task 355 | created context checkpoint 5 of 32 (pos_min = 776, pos_max = 776, n_tokens = 777, size = 62.813 MiB)
0.47.287.741 I slot  operator (): id  0 | task 355 | cached n_tokens = 835, memory_seq_rm [835, end)
0.47.288.224 I slot init_sampler: id  0 | task 355 | init sampler, took 0.19 ms, tokens: text = 839, total = 839
0.47.309.816 I slot create_check: id  0 | task 355 | created context checkpoint 6 of 32 (pos_min = 834, pos_max = 834, n_tokens = 835, size = 62.813 MiB)
0.50.656.853 I slot print_timing: id  0 | task 355 | n_gen =    100, tg =  30.34 t/s, tg_3s =  30.64 t/s
0.53.668.698 I slot print_timing: id  0 | task 355 | n_gen =    195, tg =  30.92 t/s, tg_3s =  31.54 t/s
0.54.074.762 I cmn  common_reaso: deactivated (natural end)
0.56.681.761 I slot print_timing: id  0 | task 355 | n_gen =    291, tg =  31.22 t/s, tg_3s =  31.86 t/s
0.59.714.819 I slot print_timing: id  0 | task 355 | n_gen =    382, tg =  30.92 t/s, tg_3s =  30.00 t/s
1.02.728.379 I slot print_timing: id  0 | task 355 | n_gen =    474, tg =  30.84 t/s, tg_3s =  30.53 t/s
1.05.731.156 I slot print_timing: id  0 | task 355 | n_gen =    567, tg =  30.87 t/s, tg_3s =  30.97 t/s
1.08.732.735 I slot print_timing: id  0 | task 355 | n_gen =    659, tg =  30.84 t/s, tg_3s =  30.65 t/s
1.11.738.041 I slot print_timing: id  0 | task 355 | n_gen =    747, tg =  30.64 t/s, tg_3s =  29.28 t/s
1.14.766.070 I slot print_timing: id  0 | task 355 | n_gen =    840, tg =  30.65 t/s, tg_3s =  30.71 t/s
1.17.284.664 I slot print_timing: id  0 | task 355 | prompt eval time =    1243.91 ms /    62 tokens (   20.06 ms per token,    49.84 tokens per second)
1.17.284.675 I slot print_timing: id  0 | task 355 |        eval time =   29891.22 ms /   918 tokens (   32.60 ms per token,    30.68 tokens per second)
1.17.284.677 I slot print_timing: id  0 | task 355 |       total time =   31135.14 ms /   980 tokens
1.17.284.678 I slot print_timing: id  0 | task 355 |    graphs reused =       1258
1.17.284.758 I slot      release: id  0 | task 355 | stop processing: n_tokens = 1756, truncated = 0
1.17.284.794 I srv  update_slots: all slots are idle
1.27.636.391 I srv   operator (): operator (): cleaning up before exit...
1.27.638.848 I common_memory_breakdown_print: | memory breakdown [MiB]          | total    free     self   model   context   compute    unaccounted |
1.27.638.858 I common_memory_breakdown_print: |   - CUDA0 (RTX 5090 Laptop GPU) | 24462 = 16047 + ( 6918 =  3317 +    2780 +     820) +        1496 |
1.27.638.859 I common_memory_breakdown_print: |   - Host                        |                  19794 = 19515 +       2 +     276                |
2026-10-10 17:58:48 PS C:\Users\<user>>