/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */
/**
 * mhc_res Multi-DType Test (fp32/fp16/bf16)
 * Tests stream mixing: out[b*S+t] = Σ_s h_res[s,t] * in[b*S+s]
 */

#include "acl/acl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <random>
template <typename To, typename From>
inline To bit_copy(const From& src) {
    static_assert(sizeof(To) == sizeof(From), "size mismatch");
    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}

extern "C" void mhc_res_do_fp32(
    uint32_t blockDim, void* stream,
    uint8_t* input, uint8_t* h_res, uint8_t* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
);

extern "C" void mhc_res_do_fp16(
    uint32_t blockDim, void* stream,
    uint8_t* input, uint8_t* h_res, uint8_t* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
);

extern "C" void mhc_res_do_bf16(
    uint32_t blockDim, void* stream,
    uint8_t* input, uint8_t* h_res_fp32, uint8_t* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
);

#define CHECK_ACL(x) do { \
    aclError err = (x); \
    if (err != ACL_SUCCESS) { \
        printf("ACL Error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

uint16_t float_to_half(float f) {
    uint32_t x;
    x = bit_copy<uint32_t>(f);
    uint16_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (exp << 10) | (mant >> 13);
}

float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t result;
    if (exp == 0) result = sign;
    else if (exp == 31) result = sign | 0x7F800000 | (mant << 13);
    else result = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float f = bit_copy<float>(result);
    return f;
}

uint16_t float_to_bf16(float f) {
    uint32_t x;
    x = bit_copy<uint32_t>(f);
    return (uint16_t)(x >> 16);
}

float bf16_to_float(uint16_t h) {
    uint32_t x = (uint32_t)h << 16;
    float f = bit_copy<float>(x);
    return f;
}

void cpu_reference_fp32(
    const float* input, const float* h_res, float* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
) {
    int64_t stream_elements = seq_len * dim;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t t = 0; t < num_streams; ++t) {
            for (int64_t i = 0; i < stream_elements; ++i) {
                float sum = 0.0f;
                for (int64_t s = 0; s < num_streams; ++s) {
                    int64_t in_idx = (b * num_streams + s) * stream_elements + i;
                    sum += input[in_idx] * h_res[s * num_streams + t];
                }
                int64_t out_idx = (b * num_streams + t) * stream_elements + i;
                output[out_idx] = sum;
            }
        }
    }
}

struct AllCloseResult {
    bool pass;
    float max_abs_err;
    float max_rel_err;
    int64_t fail_count;
    int64_t total;
};

AllCloseResult allclose(
    const float* actual, const float* expected, int64_t size,
    float atol, float rtol
) {
    AllCloseResult r = {true, 0, 0, 0, size};
    for (int64_t i = 0; i < size; ++i) {
        float abs_err = std::abs(actual[i] - expected[i]);
        float threshold = atol + rtol * std::abs(expected[i]);
        if (abs_err > threshold) {
            r.fail_count++;
            r.pass = false;
        }
        r.max_abs_err = std::max(r.max_abs_err, abs_err);
        float denom = std::max(std::abs(expected[i]), std::abs(actual[i]));
        if (denom > 1e-6f)
            r.max_rel_err = std::max(r.max_rel_err, abs_err / denom);
    }
    return r;
}

bool test_fp32(int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams) {
    printf("FP32 Test: [%ld, %ld, %ld] x %ld streams\n", batch, seq_len, dim, num_streams);

    int64_t io_size = batch * num_streams * seq_len * dim;
    int64_t weight_size = num_streams * num_streams;

    std::vector<float> h_input(io_size), h_weight(weight_size);
    std::vector<float> h_output(io_size), h_ref(io_size);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_pos(0.1f, 1.1f);
    for (int64_t i = 0; i < io_size; ++i)
        h_input[i] = dist(rng);
    
    // Initialize h_res as a doubly stochastic-ish matrix (rows sum to 1)
    for (int64_t s = 0; s < num_streams; ++s) {
        float sum = 0.0f;
        for (int64_t t = 0; t < num_streams; ++t) {
            h_weight[s * num_streams + t] = dist_pos(rng);
            sum += h_weight[s * num_streams + t];
        }
        for (int64_t t = 0; t < num_streams; ++t)
            h_weight[s * num_streams + t] /= sum;
    }

    cpu_reference_fp32(h_input.data(), h_weight.data(), h_ref.data(),
                       batch, seq_len, dim, num_streams);

    void *d_input, *d_weight, *d_output;
    CHECK_ACL(aclrtMalloc(&d_input, io_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_weight, weight_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_output, io_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(d_input, io_size * sizeof(float), h_input.data(),
                          io_size * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_weight, weight_size * sizeof(float), h_weight.data(),
                          weight_size * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));
    mhc_res_do_fp32(0, stream,
                    (uint8_t*)d_input, (uint8_t*)d_weight, (uint8_t*)d_output,
                    batch, seq_len, dim, num_streams);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(h_output.data(), io_size * sizeof(float),
                          d_output, io_size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

    auto r = allclose(h_output.data(), h_ref.data(), io_size, 1e-6f, 1e-5f);

    CHECK_ACL(aclrtFree(d_input));
    CHECK_ACL(aclrtFree(d_weight));
    CHECK_ACL(aclrtFree(d_output));
    CHECK_ACL(aclrtDestroyStream(stream));

    printf("  max_abs=%.2e  max_rel=%.2e  fail=%ld/%ld  %s\n\n",
           r.max_abs_err, r.max_rel_err, r.fail_count, r.total, r.pass ? "PASS" : "FAIL");
    return r.pass;
}

bool test_fp16(int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams) {
    printf("FP16 Test: [%ld, %ld, %ld] x %ld streams\n", batch, seq_len, dim, num_streams);

    int64_t io_size = batch * num_streams * seq_len * dim;
    int64_t weight_size = num_streams * num_streams;

    std::vector<float> h_input_f(io_size), h_weight_f(weight_size);
    std::vector<uint16_t> h_input(io_size), h_weight(weight_size), h_output(io_size);
    std::vector<float> h_ref(io_size), h_npu_f(io_size);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_pos(0.1f, 1.1f);
    for (int64_t i = 0; i < io_size; ++i) {
        h_input_f[i] = dist(rng);
        h_input[i] = float_to_half(h_input_f[i]);
        h_input_f[i] = half_to_float(h_input[i]);
    }
    for (int64_t s = 0; s < num_streams; ++s) {
        float sum = 0.0f;
        for (int64_t t = 0; t < num_streams; ++t) {
            h_weight_f[s * num_streams + t] = dist_pos(rng);
            sum += h_weight_f[s * num_streams + t];
        }
        for (int64_t t = 0; t < num_streams; ++t) {
            h_weight_f[s * num_streams + t] /= sum;
            h_weight[s * num_streams + t] = float_to_half(h_weight_f[s * num_streams + t]);
            h_weight_f[s * num_streams + t] = half_to_float(h_weight[s * num_streams + t]);
        }
    }

    cpu_reference_fp32(h_input_f.data(), h_weight_f.data(), h_ref.data(),
                       batch, seq_len, dim, num_streams);

    void *d_input, *d_weight, *d_output;
    CHECK_ACL(aclrtMalloc(&d_input, io_size * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_weight, weight_size * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_output, io_size * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(d_input, io_size * sizeof(uint16_t), h_input.data(),
                          io_size * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_weight, weight_size * sizeof(uint16_t), h_weight.data(),
                          weight_size * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));

    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));
    mhc_res_do_fp16(0, stream,
                    (uint8_t*)d_input, (uint8_t*)d_weight, (uint8_t*)d_output,
                    batch, seq_len, dim, num_streams);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(h_output.data(), io_size * sizeof(uint16_t),
                          d_output, io_size * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));

    for (int64_t i = 0; i < io_size; ++i)
        h_npu_f[i] = half_to_float(h_output[i]);

    auto r = allclose(h_npu_f.data(), h_ref.data(), io_size, 1e-3f, 1e-2f);

    CHECK_ACL(aclrtFree(d_input));
    CHECK_ACL(aclrtFree(d_weight));
    CHECK_ACL(aclrtFree(d_output));
    CHECK_ACL(aclrtDestroyStream(stream));

    printf("  max_abs=%.2e  max_rel=%.2e  fail=%ld/%ld  %s\n\n",
           r.max_abs_err, r.max_rel_err, r.fail_count, r.total, r.pass ? "PASS" : "FAIL");
    return r.pass;
}

bool test_bf16(int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams) {
    printf("BF16 Test: [%ld, %ld, %ld] x %ld streams\n", batch, seq_len, dim, num_streams);

    int64_t io_size = batch * num_streams * seq_len * dim;
    int64_t weight_size = num_streams * num_streams;

    std::vector<float> h_input_f(io_size), h_weight_f(weight_size);
    std::vector<uint16_t> h_input(io_size), h_output(io_size);
    std::vector<float> h_ref(io_size), h_npu_f(io_size);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_pos(0.1f, 1.1f);
    for (int64_t i = 0; i < io_size; ++i) {
        h_input_f[i] = dist(rng);
        h_input[i] = float_to_bf16(h_input_f[i]);
        h_input_f[i] = bf16_to_float(h_input[i]);
    }
    for (int64_t s = 0; s < num_streams; ++s) {
        float sum = 0.0f;
        for (int64_t t = 0; t < num_streams; ++t) {
            h_weight_f[s * num_streams + t] = dist_pos(rng);
            sum += h_weight_f[s * num_streams + t];
        }
        for (int64_t t = 0; t < num_streams; ++t)
            h_weight_f[s * num_streams + t] /= sum;
    }

    cpu_reference_fp32(h_input_f.data(), h_weight_f.data(), h_ref.data(),
                       batch, seq_len, dim, num_streams);

    void *d_input, *d_weight, *d_output;
    CHECK_ACL(aclrtMalloc(&d_input, io_size * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_weight, weight_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_output, io_size * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(d_input, io_size * sizeof(uint16_t), h_input.data(),
                          io_size * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_weight, weight_size * sizeof(float), h_weight_f.data(),
                          weight_size * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));
    mhc_res_do_bf16(0, stream,
                    (uint8_t*)d_input, (uint8_t*)d_weight, (uint8_t*)d_output,
                    batch, seq_len, dim, num_streams);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(h_output.data(), io_size * sizeof(uint16_t),
                          d_output, io_size * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));

    for (int64_t i = 0; i < io_size; ++i)
        h_npu_f[i] = bf16_to_float(h_output[i]);

    auto r = allclose(h_npu_f.data(), h_ref.data(), io_size, 1e-3f, 4e-3f);

    CHECK_ACL(aclrtFree(d_input));
    CHECK_ACL(aclrtFree(d_weight));
    CHECK_ACL(aclrtFree(d_output));
    CHECK_ACL(aclrtDestroyStream(stream));

    printf("  max_abs=%.2e  max_rel=%.2e  fail=%ld/%ld  %s\n\n",
           r.max_abs_err, r.max_rel_err, r.fail_count, r.total, r.pass ? "PASS" : "FAIL");
    return r.pass;
}

int main() {
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));

    printf("=== mhc_res Multi-DType Test ===\n");
    printf("Stream mixing: out[b*S+t] = sum_s h_res[s,t] * in[b*S+s]\n");
    printf("Precision criteria:\n");
    printf("  fp32: allclose(atol=1e-6, rtol=1e-5)\n");
    printf("  fp16: allclose(atol=1e-3, rtol=1e-2)\n");
    printf("  bf16: allclose(atol=1e-3, rtol=4e-3)\n\n");

    bool all_pass = true;

    all_pass &= test_fp32(2, 64, 256, 4);
    all_pass &= test_fp32(4, 32, 128, 8);
    all_pass &= test_fp16(2, 64, 256, 4);
    all_pass &= test_fp16(4, 32, 128, 8);
    all_pass &= test_bf16(2, 64, 256, 4);
    all_pass &= test_bf16(4, 32, 128, 8);

    printf("=== Final Result: %s ===\n", all_pass ? "ALL PASS" : "SOME FAILED");

    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return all_pass ? 0 : 1;
}
