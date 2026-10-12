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
 * mhc_pre Performance Benchmark
 */

#include "acl/acl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

extern "C" void mhc_pre_do_fp32(
    uint32_t blockDim, void* stream,
    uint8_t* input, uint8_t* h_pre, uint8_t* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
);

extern "C" void mhc_pre_do_fp16(
    uint32_t blockDim, void* stream,
    uint8_t* input, uint8_t* h_pre, uint8_t* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
);

extern "C" void mhc_pre_do_bf16(
    uint32_t blockDim, void* stream,
    uint8_t* input, uint8_t* h_pre_fp32, uint8_t* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
);

#define CHECK_ACL(x) do { \
    aclError err = (x); \
    if (err != ACL_SUCCESS) { \
        printf("ACL Error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

struct BenchResult {
    double median_us;
    double min_us;
    double max_us;
    double gbps;
};

BenchResult benchmark_fp32(int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams,
                           int warmup = 10, int iters = 100) {
    int64_t input_size = batch * num_streams * seq_len * dim;
    int64_t weight_size = num_streams;
    int64_t output_size = batch * seq_len * dim;

    void *d_input, *d_weight, *d_output;
    CHECK_ACL(aclrtMalloc(&d_input, input_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_weight, weight_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_output, output_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));

    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));

    for (int i = 0; i < warmup; ++i) {
        mhc_pre_do_fp32(0, stream, (uint8_t*)d_input, (uint8_t*)d_weight,
                        (uint8_t*)d_output, batch, seq_len, dim, num_streams);
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::vector<double> times(iters);
    for (int i = 0; i < iters; ++i) {
        aclrtEvent start, end;
        CHECK_ACL(aclrtCreateEvent(&start));
        CHECK_ACL(aclrtCreateEvent(&end));
        CHECK_ACL(aclrtRecordEvent(start, stream));
        mhc_pre_do_fp32(0, stream, (uint8_t*)d_input, (uint8_t*)d_weight,
                        (uint8_t*)d_output, batch, seq_len, dim, num_streams);
        CHECK_ACL(aclrtRecordEvent(end, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        float ms;
        CHECK_ACL(aclrtEventElapsedTime(&ms, start, end));
        times[i] = ms * 1000.0;
        CHECK_ACL(aclrtDestroyEvent(start));
        CHECK_ACL(aclrtDestroyEvent(end));
    }

    CHECK_ACL(aclrtFree(d_input));
    CHECK_ACL(aclrtFree(d_weight));
    CHECK_ACL(aclrtFree(d_output));
    CHECK_ACL(aclrtDestroyStream(stream));

    std::sort(times.begin(), times.end());
    double median = times[iters / 2];
    double bytes = (input_size + output_size) * sizeof(float) + weight_size * sizeof(float);
    double gbps = bytes / (median * 1e-6) / 1e9;

    return {median, times[0], times[iters - 1], gbps};
}

void run_benchmark(const char* name, int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams) {
    auto r = benchmark_fp32(batch, seq_len, dim, num_streams);
    printf("%-20s B=%2ld S=%4ld D=%4ld N=%ld  median=%.1f us  [%.1f-%.1f]  %.1f GB/s\n",
           name, batch, seq_len, dim, num_streams, r.median_us, r.min_us, r.max_us, r.gbps);
}

int main() {
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));

    printf("=== mhc_pre Performance Benchmark (fp32) ===\n\n");

    run_benchmark("small", 4, 64, 256, 4);
    run_benchmark("medium", 8, 128, 512, 4);
    run_benchmark("large", 16, 256, 1024, 4);
    run_benchmark("xlarge", 32, 512, 2048, 4);
    run_benchmark("streams=8", 8, 128, 512, 8);
    run_benchmark("streams=2", 8, 128, 512, 2);

    printf("\n=== Benchmark Complete ===\n");

    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return 0;
}
