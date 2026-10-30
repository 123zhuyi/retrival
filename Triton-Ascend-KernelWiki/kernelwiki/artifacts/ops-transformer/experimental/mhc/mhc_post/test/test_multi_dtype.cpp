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
 * mhc_post Multi-DType Test (fp32/fp16/bf16)
 *
 * Precision criteria:
 *   fp32: bit-exact (ULP = 0)
 *   fp16: allclose(atol=1e-4, rtol=1e-3)
 *   bf16: allclose(atol=1e-3, rtol=4e-3)
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

extern "C" void mhc_post_do_fp32(
    uint32_t blockDim, void* stream,
    uint8_t* branch_output, uint8_t* h_post, uint8_t* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
);

extern "C" void mhc_post_do_fp16(
    uint32_t blockDim, void* stream,
    uint8_t* branch_output, uint8_t* h_post, uint8_t* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
);

extern "C" void mhc_post_do_bf16(
    uint32_t blockDim, void* stream,
    uint8_t* branch_output, uint8_t* h_post_fp32, uint8_t* output,
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
    const float* branch_output, const float* h_post, float* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
) {
    int64_t batch_elements = seq_len * dim;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < num_streams; ++s) {
            float weight = h_post[s];
            int64_t out_idx = (b * num_streams + s) * batch_elements;
            int64_t in_idx = b * batch_elements;
            for (int64_t i = 0; i < batch_elements; ++i)
                output[out_idx + i] = branch_output[in_idx + i] * weight;
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
    printf("FP32 Test: [%ld, %ld, %ld] x %ld\n", batch, seq_len, dim, num_streams);

    int64_t input_size = batch * seq_len * dim;
    int64_t weight_size = num_streams;
    int64_t output_size = batch * num_streams * seq_len * dim;

    std::vector<float> h_branch(input_size), h_weight(weight_size);
    std::vector<float> h_output(output_size), h_ref(output_size);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_pos(0.1f, 1.1f);
    for (int64_t i = 0; i < input_size; ++i)
        h_branch[i] = dist(rng);
    float sum = 0.0f;
    for (int64_t i = 0; i < weight_size; ++i) {
        h_weight[i] = dist_pos(rng);
        sum += h_weight[i];
    }
    for (int64_t i = 0; i < weight_size; ++i) h_weight[i] /= sum;

    cpu_reference_fp32(h_branch.data(), h_weight.data(), h_ref.data(),
                       batch, seq_len, dim, num_streams);

    void *d_branch, *d_weight, *d_output;
    CHECK_ACL(aclrtMalloc(&d_branch, input_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_weight, weight_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_output, output_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(d_branch, input_size * sizeof(float), h_branch.data(),
                          input_size * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_weight, weight_size * sizeof(float), h_weight.data(),
                          weight_size * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));
    mhc_post_do_fp32(batch * num_streams, stream,
                     (uint8_t*)d_branch, (uint8_t*)d_weight, (uint8_t*)d_output,
                     batch, seq_len, dim, num_streams);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(h_output.data(), output_size * sizeof(float),
                          d_output, output_size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

    int64_t mismatch = 0;
    for (int64_t i = 0; i < output_size; ++i) {
        uint32_t a, b;
        a = bit_copy<uint32_t>(h_output[i]);
        b = bit_copy<uint32_t>(h_ref[i]);
        if (a != b) mismatch++;
    }

    CHECK_ACL(aclrtFree(d_branch));
    CHECK_ACL(aclrtFree(d_weight));
    CHECK_ACL(aclrtFree(d_output));
    CHECK_ACL(aclrtDestroyStream(stream));

    bool pass = (mismatch == 0);
    printf("  bit_exact=%s (%ld/%ld mismatch)  %s\n\n",
           pass ? "yes" : "no", mismatch, output_size, pass ? "PASS" : "FAIL");
    return pass;
}

bool test_fp16(int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams) {
    printf("FP16 Test: [%ld, %ld, %ld] x %ld\n", batch, seq_len, dim, num_streams);

    int64_t input_size = batch * seq_len * dim;
    int64_t weight_size = num_streams;
    int64_t output_size = batch * num_streams * seq_len * dim;

    std::vector<float> h_branch_f(input_size), h_weight_f(weight_size);
    std::vector<uint16_t> h_branch(input_size), h_weight(weight_size), h_output(output_size);
    std::vector<float> h_ref(output_size), h_npu_f(output_size);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_pos(0.1f, 1.1f);
    for (int64_t i = 0; i < input_size; ++i) {
        h_branch_f[i] = dist(rng);
        h_branch[i] = float_to_half(h_branch_f[i]);
        h_branch_f[i] = half_to_float(h_branch[i]);
    }
    float sum = 0.0f;
    for (int64_t i = 0; i < weight_size; ++i) {
        h_weight_f[i] = dist_pos(rng);
        sum += h_weight_f[i];
    }
    for (int64_t i = 0; i < weight_size; ++i) {
        h_weight_f[i] /= sum;
        h_weight[i] = float_to_half(h_weight_f[i]);
        h_weight_f[i] = half_to_float(h_weight[i]);
    }

    cpu_reference_fp32(h_branch_f.data(), h_weight_f.data(), h_ref.data(),
                       batch, seq_len, dim, num_streams);

    void *d_branch, *d_weight, *d_output;
    CHECK_ACL(aclrtMalloc(&d_branch, input_size * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_weight, weight_size * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_output, output_size * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(d_branch, input_size * sizeof(uint16_t), h_branch.data(),
                          input_size * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_weight, weight_size * sizeof(uint16_t), h_weight.data(),
                          weight_size * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));

    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));
    mhc_post_do_fp16(batch * num_streams, stream,
                     (uint8_t*)d_branch, (uint8_t*)d_weight, (uint8_t*)d_output,
                     batch, seq_len, dim, num_streams);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(h_output.data(), output_size * sizeof(uint16_t),
                          d_output, output_size * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));

    for (int64_t i = 0; i < output_size; ++i)
        h_npu_f[i] = half_to_float(h_output[i]);

    // atol covers subnormal flush-to-zero; rtol covers rounding
    auto r = allclose(h_npu_f.data(), h_ref.data(), output_size, 1e-4f, 1e-3f);

    CHECK_ACL(aclrtFree(d_branch));
    CHECK_ACL(aclrtFree(d_weight));
    CHECK_ACL(aclrtFree(d_output));
    CHECK_ACL(aclrtDestroyStream(stream));

    printf("  max_abs=%.2e  max_rel=%.2e  fail=%ld/%ld  %s\n\n",
           r.max_abs_err, r.max_rel_err, r.fail_count, r.total, r.pass ? "PASS" : "FAIL");
    return r.pass;
}

bool test_bf16(int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams) {
    printf("BF16 Test: [%ld, %ld, %ld] x %ld\n", batch, seq_len, dim, num_streams);

    int64_t input_size = batch * seq_len * dim;
    int64_t weight_size = num_streams;
    int64_t output_size = batch * num_streams * seq_len * dim;

    std::vector<float> h_branch_f(input_size), h_weight_f(weight_size);
    std::vector<uint16_t> h_branch(input_size), h_output(output_size);
    std::vector<float> h_ref(output_size), h_npu_f(output_size);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_pos(0.1f, 1.1f);
    for (int64_t i = 0; i < input_size; ++i) {
        h_branch_f[i] = dist(rng);
        h_branch[i] = float_to_bf16(h_branch_f[i]);
        h_branch_f[i] = bf16_to_float(h_branch[i]);
    }
    float sum = 0.0f;
    for (int64_t i = 0; i < weight_size; ++i) {
        h_weight_f[i] = dist_pos(rng);
        sum += h_weight_f[i];
    }
    for (int64_t i = 0; i < weight_size; ++i)
        h_weight_f[i] /= sum;

    cpu_reference_fp32(h_branch_f.data(), h_weight_f.data(), h_ref.data(),
                       batch, seq_len, dim, num_streams);

    void *d_branch, *d_weight, *d_output;
    CHECK_ACL(aclrtMalloc(&d_branch, input_size * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_weight, weight_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_output, output_size * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(d_branch, input_size * sizeof(uint16_t), h_branch.data(),
                          input_size * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_weight, weight_size * sizeof(float), h_weight_f.data(),
                          weight_size * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));
    mhc_post_do_bf16(batch * num_streams, stream,
                     (uint8_t*)d_branch, (uint8_t*)d_weight, (uint8_t*)d_output,
                     batch, seq_len, dim, num_streams);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(h_output.data(), output_size * sizeof(uint16_t),
                          d_output, output_size * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));

    for (int64_t i = 0; i < output_size; ++i)
        h_npu_f[i] = bf16_to_float(h_output[i]);

    // bf16 has 8-bit mantissa (vs fp16 10-bit), needs wider tolerance
    auto r = allclose(h_npu_f.data(), h_ref.data(), output_size, 1e-3f, 4e-3f);

    CHECK_ACL(aclrtFree(d_branch));
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

    printf("=== mhc_post Multi-DType Test ===\n");
    printf("Precision criteria:\n");
    printf("  fp32: bit-exact (ULP = 0)\n");
    printf("  fp16: allclose(atol=1e-4, rtol=1e-3)\n");
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
