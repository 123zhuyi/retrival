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
 * mhc_pre blockDim Sweep for Tuning
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

#define CHECK_ACL(x) do { \
    aclError err = (x); \
    if (err != ACL_SUCCESS) { \
        printf("ACL Error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

double benchmark_with_blockdim(uint32_t blockDim, int64_t batch, int64_t seq_len,
                                int64_t dim, int64_t num_streams,
                                void* d_input, void* d_weight, void* d_output,
                                aclrtStream stream, int iters = 50) {
    for (int i = 0; i < 5; ++i) {
        mhc_pre_do_fp32(blockDim, stream, (uint8_t*)d_input, (uint8_t*)d_weight,
                        (uint8_t*)d_output, batch, seq_len, dim, num_streams);
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::vector<double> times(iters);
    for (int i = 0; i < iters; ++i) {
        aclrtEvent start, end;
        CHECK_ACL(aclrtCreateEvent(&start));
        CHECK_ACL(aclrtCreateEvent(&end));
        CHECK_ACL(aclrtRecordEvent(start, stream));
        mhc_pre_do_fp32(blockDim, stream, (uint8_t*)d_input, (uint8_t*)d_weight,
                        (uint8_t*)d_output, batch, seq_len, dim, num_streams);
        CHECK_ACL(aclrtRecordEvent(end, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        float ms;
        CHECK_ACL(aclrtEventElapsedTime(&ms, start, end));
        times[i] = ms * 1000.0;
        CHECK_ACL(aclrtDestroyEvent(start));
        CHECK_ACL(aclrtDestroyEvent(end));
    }
    std::sort(times.begin(), times.end());
    return times[iters / 2];
}

void sweep_blockdim(int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams) {
    printf("\nSweeping blockDim for [B=%ld, S=%ld, D=%ld, N=%ld]\n", batch, seq_len, dim, num_streams);
    printf("Max logical blocks = %ld\n", batch);

    int64_t input_size = batch * num_streams * seq_len * dim;
    int64_t weight_size = num_streams;
    int64_t output_size = batch * seq_len * dim;

    void *d_input, *d_weight, *d_output;
    CHECK_ACL(aclrtMalloc(&d_input, input_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_weight, weight_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_output, output_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));

    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint32_t max_blocks = static_cast<uint32_t>(batch);
    double best_time = 1e9;
    uint32_t best_blockdim = 0;

    printf("\n%10s %12s %12s\n", "blockDim", "median(us)", "rel");
    printf("%10s %12s %12s\n", "--------", "----------", "---");

    std::vector<uint32_t> test_dims;
    for (uint32_t bd = 1; bd <= max_blocks && bd <= 64; bd *= 2)
        test_dims.push_back(bd);
    if (max_blocks > 0 && std::find(test_dims.begin(), test_dims.end(), max_blocks) == test_dims.end())
        test_dims.push_back(max_blocks);
    std::sort(test_dims.begin(), test_dims.end());

    for (uint32_t bd : test_dims) {
        double t = benchmark_with_blockdim(bd, batch, seq_len, dim, num_streams,
                                           d_input, d_weight, d_output, stream);
        if (t < best_time) {
            best_time = t;
            best_blockdim = bd;
        }
        printf("%10u %12.1f %11.2fx\n", bd, t, t / best_time);
    }

    printf("\nBest: blockDim=%u (%.1f us)\n", best_blockdim, best_time);

    CHECK_ACL(aclrtFree(d_input));
    CHECK_ACL(aclrtFree(d_weight));
    CHECK_ACL(aclrtFree(d_output));
    CHECK_ACL(aclrtDestroyStream(stream));
}

int main() {
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));

    printf("=== mhc_pre blockDim Sweep ===\n");

    sweep_blockdim(8, 128, 512, 4);
    sweep_blockdim(16, 256, 1024, 4);
    sweep_blockdim(32, 128, 512, 4);

    printf("\n=== Sweep Complete ===\n");

    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return 0;
}
