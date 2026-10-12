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
 * mhc_post blockDim Sweep - Find optimal blockDim
 * Usage: ./perf_sweep [batch seq dim streams] [dtype]
 *   dtype: fp32, fp16, bf16, all (default: all)
 */

#include "acl/acl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>

extern "C" void mhc_post_do_fp32(
    uint32_t blockDim, void* stream, uint8_t* in, uint8_t* h, uint8_t* out,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams);

extern "C" void mhc_post_do_fp16(
    uint32_t blockDim, void* stream, uint8_t* in, uint8_t* h, uint8_t* out,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams);

extern "C" void mhc_post_do_bf16(
    uint32_t blockDim, void* stream, uint8_t* in, uint8_t* h_fp32, uint8_t* out,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams);

#define CHECK_ACL(x) do { \
    aclError err = (x); \
    if (err != ACL_SUCCESS) { \
        printf("ACL Error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

using KernelFunc = void(*)(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, int64_t, int64_t, int64_t, int64_t);

double measure_time(KernelFunc kernel, uint32_t blockDim,
                    void* d_in, void* d_w, void* d_out,
                    int64_t b, int64_t s, int64_t d, int64_t n,
                    int warmup = 5, int repeat = 20) {
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));

    for (int i = 0; i < warmup; ++i) {
        kernel(blockDim, stream, (uint8_t*)d_in, (uint8_t*)d_w, (uint8_t*)d_out, b, s, d, n);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    std::vector<double> times(repeat);
    for (int i = 0; i < repeat; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        kernel(blockDim, stream, (uint8_t*)d_in, (uint8_t*)d_w, (uint8_t*)d_out, b, s, d, n);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto end = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::micro>(end - start).count();
    }

    CHECK_ACL(aclrtDestroyStream(stream));
    std::sort(times.begin(), times.end());
    return times[repeat / 2];  // median
}

void sweep(const char* dtype, KernelFunc kernel,
           int64_t b, int64_t s, int64_t d, int64_t n,
           int data_sz, int weight_sz) {
    printf("\n[%s] batch=%ld seq=%ld dim=%ld streams=%ld\n", dtype, b, s, d, n);

    int64_t in_sz = b * s * d;
    int64_t out_sz = b * n * s * d;

    void *d_in, *d_w, *d_out;
    CHECK_ACL(aclrtMalloc(&d_in, in_sz * data_sz, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_w, n * weight_sz, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_out, out_sz * data_sz, ACL_MEM_MALLOC_HUGE_FIRST));

    uint32_t total_tasks = b * n;
    uint32_t heuristic_bd = total_tasks;

    printf("  total_tasks=%u, heuristic=%u\n", total_tasks, heuristic_bd);
    printf("  | blockDim | time(us) | vs_max  | note\n");
    printf("  |----------|----------|---------|-----\n");

    // Test maxBlk first
    double maxblk_time = measure_time(kernel, total_tasks, d_in, d_w, d_out, b, s, d, n);
    printf("  | %8u | %8.1f | %7s | maxBlk\n", total_tasks, maxblk_time, "1.00x");

    double best_time = maxblk_time;
    uint32_t best_bd = total_tasks;

    // Test heuristic if different
    if (heuristic_bd != total_tasks) {
        double h_time = measure_time(kernel, 0, d_in, d_w, d_out, b, s, d, n);
        double hr = h_time / maxblk_time;
        printf("  | %8u | %8.1f | %6.2fx | heuristic%s\n",
               heuristic_bd, h_time, hr, (h_time < best_time) ? " *" : "");
        if (h_time < best_time) { best_time = h_time; best_bd = heuristic_bd; }
    }

    // Test other candidates
    std::vector<uint32_t> candidates;
    for (uint32_t bd = 1; bd <= total_tasks && bd <= 64; ++bd) {
        if (total_tasks % bd == 0) candidates.push_back(bd);
    }
    for (uint32_t bd : {1u, 2u, 4u, 8u, 16u, 32u, 48u}) {
        if (bd < total_tasks && std::find(candidates.begin(), candidates.end(), bd) == candidates.end())
            candidates.push_back(bd);
    }
    std::sort(candidates.begin(), candidates.end());

    for (uint32_t bd : candidates) {
        if (bd == total_tasks || bd == heuristic_bd) continue;
        double t = measure_time(kernel, bd, d_in, d_w, d_out, b, s, d, n);
        double ratio = t / maxblk_time;
        const char* mark = (t < best_time) ? " *" : "";
        printf("  | %8u | %8.1f | %6.2fx |%s\n", bd, t, ratio, mark);
        if (t < best_time) { best_time = t; best_bd = bd; }
    }

    int64_t total_bytes = (in_sz + out_sz) * data_sz + n * weight_sz;
    double bw = total_bytes / best_time / 1e3;  // GB/s
    printf("  >> Best: blockDim=%u (%.1f us, %.0f GB/s)\n", best_bd, best_time, bw);

    CHECK_ACL(aclrtFree(d_in));
    CHECK_ACL(aclrtFree(d_w));
    CHECK_ACL(aclrtFree(d_out));
}

void print_usage(const char* prog) {
    printf("Usage: %s [batch seq dim streams] [dtype]\n", prog);
    printf("  dtype: fp32, fp16, bf16, all (default: all)\n");
    printf("Examples:\n");
    printf("  %s                    # run default test cases\n", prog);
    printf("  %s 16 64 4096 4       # custom shape, all dtypes\n", prog);
    printf("  %s 16 64 4096 4 fp16  # custom shape, fp16 only\n", prog);
}

int main(int argc, char** argv) {
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));

    printf("=== mhc_post blockDim Sweep ===\n");
    printf("Default: blockDim = batch * streams (maxBlk)\n");

    if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        print_usage(argv[0]);
        return 0;
    }

    struct TC { int64_t b, s, d, n; };
    std::vector<TC> cases;
    const char* dtype = "all";

    if (argc >= 5) {
        // Custom shape from args
        TC tc = {atol(argv[1]), atol(argv[2]), atol(argv[3]), atol(argv[4])};
        cases.push_back(tc);
        if (argc >= 6) dtype = argv[5];
    } else {
        // Default test cases
        cases = {
            {4, 128, 256, 4},   // total=16, small
            {8, 128, 256, 4},   // total=32
            {4, 512, 1024, 4},  // total=16, large data
            {16, 64, 4096, 4},  // total=64, very large dim
        };
    }

    for (auto& tc : cases) {
        printf("\n========================================\n");
        if (strcmp(dtype, "all") == 0 || strcmp(dtype, "fp32") == 0)
            sweep("fp32", mhc_post_do_fp32, tc.b, tc.s, tc.d, tc.n, 4, 4);
        if (strcmp(dtype, "all") == 0 || strcmp(dtype, "fp16") == 0)
            sweep("fp16", mhc_post_do_fp16, tc.b, tc.s, tc.d, tc.n, 2, 2);
        if (strcmp(dtype, "all") == 0 || strcmp(dtype, "bf16") == 0)
            sweep("bf16", mhc_post_do_bf16, tc.b, tc.s, tc.d, tc.n, 2, 4);
    }

    printf("\n=== Sweep Complete ===\n");

    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return 0;
}
