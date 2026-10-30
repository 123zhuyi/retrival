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
 * mhc_post Performance Benchmark
 */

#include "acl/acl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>

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

struct PerfResult {
    double mean_us;
    double std_us;
    double min_us;
    double max_us;
    double bandwidth_gbps;
};

template <typename Func>
PerfResult measure_perf(Func kernel_func, int64_t total_bytes, int warmup = 3, int repeat = 10) {
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));

    for (int i = 0; i < warmup; ++i) {
        kernel_func(stream);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    std::vector<double> times(repeat);
    for (int i = 0; i < repeat; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        kernel_func(stream);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto end = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::micro>(end - start).count();
    }

    CHECK_ACL(aclrtDestroyStream(stream));

    double sum = 0, sum_sq = 0;
    double min_t = times[0], max_t = times[0];
    for (double t : times) {
        sum += t;
        sum_sq += t * t;
        min_t = std::min(min_t, t);
        max_t = std::max(max_t, t);
    }

    double mean = sum / repeat;
    double variance = sum_sq / repeat - mean * mean;
    double stddev = std::sqrt(variance > 0 ? variance : 0);

    double bandwidth = (double)total_bytes / (mean * 1e-6) / 1e9;

    return {mean, stddev, min_t, max_t, bandwidth};
}

void run_perf_test(const char* dtype, int64_t batch, int64_t seq_len, int64_t dim,
                   int64_t num_streams, int data_elem_size, int weight_elem_size,
                   void (*kernel)(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*,
                                  int64_t, int64_t, int64_t, int64_t)) {
    int64_t input_size = batch * seq_len * dim;
    int64_t weight_size = num_streams;
    int64_t output_size = batch * num_streams * seq_len * dim;

    void *d_branch, *d_weight, *d_output;
    CHECK_ACL(aclrtMalloc(&d_branch, input_size * data_elem_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_weight, weight_size * weight_elem_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_output, output_size * data_elem_size, ACL_MEM_MALLOC_HUGE_FIRST));

    uint32_t blockDim = 0;  // use heuristic default

    int64_t total_bytes = (input_size + output_size) * data_elem_size + weight_size * weight_elem_size;

    auto kernel_func = [&](aclrtStream stream) {
        kernel(blockDim, stream,
               (uint8_t*)d_branch, (uint8_t*)d_weight, (uint8_t*)d_output,
               batch, seq_len, dim, num_streams);
    };

    PerfResult result = measure_perf(kernel_func, total_bytes);

    printf("| %s | %ld | %ld | %ld | %ld | %.1f | %.1f | %.1f | %.2f |\n",
           dtype, batch, seq_len, dim, num_streams,
           result.mean_us, result.std_us, result.min_us, result.bandwidth_gbps);

    CHECK_ACL(aclrtFree(d_branch));
    CHECK_ACL(aclrtFree(d_weight));
    CHECK_ACL(aclrtFree(d_output));
}

int main() {
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));

    printf("=== mhc_post Performance Benchmark ===\n\n");
    printf("| dtype | batch | seq | dim | streams | mean(us) | std(us) | min(us) | BW(GB/s) |\n");
    printf("|-------|-------|-----|-----|---------|----------|---------|---------|----------|\n");

    struct TestCase { int64_t b, s, d, n; };
    TestCase cases[] = {
        {1, 128, 256, 4},
        {2, 128, 256, 4},
        {4, 128, 256, 4},
        {8, 128, 256, 4},
        {4, 256, 512, 4},
        {4, 512, 1024, 4},
        {8, 128, 2048, 4},
        {16, 64, 4096, 4},
    };

    for (auto& tc : cases) {
        run_perf_test("fp32", tc.b, tc.s, tc.d, tc.n, 4, 4, mhc_post_do_fp32);
    }

    printf("\n");
    for (auto& tc : cases) {
        run_perf_test("fp16", tc.b, tc.s, tc.d, tc.n, 2, 2, mhc_post_do_fp16);
    }

    printf("\n");
    for (auto& tc : cases) {
        run_perf_test("bf16", tc.b, tc.s, tc.d, tc.n, 2, 4, mhc_post_do_bf16);  // bf16 data, fp32 weight
    }

    printf("\n=== Benchmark Complete ===\n");

    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());

    return 0;
}
