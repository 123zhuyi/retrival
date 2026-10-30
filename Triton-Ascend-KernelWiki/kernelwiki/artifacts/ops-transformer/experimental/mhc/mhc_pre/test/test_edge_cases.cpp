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
 * mhc_pre Edge Cases Test
 * Tests boundary conditions: non-aligned dims, small batches, etc.
 */

#include "acl/acl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <random>

extern "C" void mhc_pre_do_fp32(
    uint32_t blockDim, void* stream,
    uint8_t* input, uint8_t* h_pre, uint8_t* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
);

#define CHECK_ACL(x) do { \
    aclError err = (x); \
    if (err != ACL_SUCCESS) { \
        printf("ACL Error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

void cpu_reference_fp32(
    const float* input, const float* h_pre, float* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
) {
    int64_t batch_elements = seq_len * dim;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t i = 0; i < batch_elements; ++i) {
            float sum = 0.0f;
            for (int64_t s = 0; s < num_streams; ++s) {
                int64_t in_idx = (b * num_streams + s) * batch_elements + i;
                sum += input[in_idx] * h_pre[s];
            }
            output[b * batch_elements + i] = sum;
        }
    }
}

bool test_case(const char* name, int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams) {
    printf("Test: %s [B=%ld, S=%ld, D=%ld, N=%ld]\n", name, batch, seq_len, dim, num_streams);

    int64_t input_size = batch * num_streams * seq_len * dim;
    int64_t weight_size = num_streams;
    int64_t output_size = batch * seq_len * dim;

    std::vector<float> h_input(input_size), h_weight(weight_size);
    std::vector<float> h_ref(output_size), h_npu(output_size);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_pos(0.1f, 1.1f);
    for (int64_t i = 0; i < input_size; ++i)
        h_input[i] = dist(rng);
    float sum = 0.0f;
    for (int64_t i = 0; i < weight_size; ++i) {
        h_weight[i] = dist_pos(rng);
        sum += h_weight[i];
    }
    for (int64_t i = 0; i < weight_size; ++i)
        h_weight[i] /= sum;

    cpu_reference_fp32(h_input.data(), h_weight.data(), h_ref.data(),
                       batch, seq_len, dim, num_streams);

    void *d_input, *d_weight, *d_output;
    CHECK_ACL(aclrtMalloc(&d_input, input_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_weight, weight_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_output, output_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(d_input, input_size * sizeof(float), h_input.data(),
                          input_size * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_weight, weight_size * sizeof(float), h_weight.data(),
                          weight_size * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));
    mhc_pre_do_fp32(0, stream,
                    (uint8_t*)d_input, (uint8_t*)d_weight, (uint8_t*)d_output,
                    batch, seq_len, dim, num_streams);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(h_npu.data(), output_size * sizeof(float),
                          d_output, output_size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

    bool pass = true;
    float max_err = 0.0f;
    for (int64_t i = 0; i < output_size; ++i) {
        float err = std::abs(h_npu[i] - h_ref[i]);
        max_err = std::max(max_err, err);
        if (err > 1e-6f) pass = false;
    }

    CHECK_ACL(aclrtFree(d_input));
    CHECK_ACL(aclrtFree(d_weight));
    CHECK_ACL(aclrtFree(d_output));
    CHECK_ACL(aclrtDestroyStream(stream));

    printf("  max_err=%.2e  %s\n\n", max_err, pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));

    printf("=== mhc_pre Edge Cases Test ===\n\n");

    bool all_pass = true;

    all_pass &= test_case("dim=1", 2, 8, 1, 4);
    all_pass &= test_case("dim=7 (non-aligned)", 2, 8, 7, 4);
    all_pass &= test_case("dim=15 (non-aligned)", 2, 8, 15, 4);
    all_pass &= test_case("batch=1", 1, 64, 128, 4);
    all_pass &= test_case("seq=1", 4, 1, 256, 4);
    all_pass &= test_case("num_streams=1", 4, 32, 128, 1);
    all_pass &= test_case("num_streams=2", 4, 32, 128, 2);
    all_pass &= test_case("large batch", 32, 16, 64, 4);
    all_pass &= test_case("large dim", 2, 8, 2048, 4);
    all_pass &= test_case("large seq", 2, 512, 64, 4);

    printf("=== Final Result: %s ===\n", all_pass ? "ALL PASS" : "SOME FAILED");

    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return all_pass ? 0 : 1;
}
