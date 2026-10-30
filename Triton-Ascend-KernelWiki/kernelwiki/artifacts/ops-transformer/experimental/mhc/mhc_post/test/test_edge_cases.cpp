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
 * mhc_post Edge Cases Test (fp32/fp16/bf16)
 * Tests unaligned dim, extreme batch/seq/streams, minimal cases
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

uint16_t float_to_half(float f) {
    uint32_t x = bit_copy<uint32_t>(f);
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
    return bit_copy<float>(result);
}

uint16_t float_to_bf16(float f) {
    uint32_t x = bit_copy<uint32_t>(f);
    return (uint16_t)(x >> 16);
}

float bf16_to_float(uint16_t h) {
    uint32_t x = (uint32_t)h << 16;
    return bit_copy<float>(x);
}

void cpu_ref(const float* in, const float* h, float* out,
             int64_t batch, int64_t seq, int64_t dim, int64_t streams) {
    int64_t elems = seq * dim;
    for (int64_t b = 0; b < batch; ++b)
        for (int64_t s = 0; s < streams; ++s)
            for (int64_t i = 0; i < elems; ++i)
                out[(b * streams + s) * elems + i] = in[b * elems + i] * h[s];
}

bool allclose(const float* a, const float* b, int64_t n, float atol, float rtol) {
    for (int64_t i = 0; i < n; ++i) {
        float err = std::abs(a[i] - b[i]);
        if (err > atol + rtol * std::abs(b[i])) return false;
    }
    return true;
}

bool bit_exact(const float* a, const float* b, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        uint32_t x, y;
        x = bit_copy<uint32_t>(a[i]); y = bit_copy<uint32_t>(b[i]);
        if (x != y) return false;
    }
    return true;
}

bool test_fp32(const char* name, int64_t batch, int64_t seq, int64_t dim, int64_t streams) {
    int64_t in_sz = batch * seq * dim;
    int64_t out_sz = batch * streams * seq * dim;
    std::vector<float> h_in(in_sz), h_w(streams), h_out(out_sz), h_ref(out_sz);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_pos(0.1f, 1.1f);
    for (int64_t i = 0; i < in_sz; ++i) h_in[i] = dist(rng);
    float sum = 0;
    for (int64_t i = 0; i < streams; ++i) { h_w[i] = dist_pos(rng); sum += h_w[i]; }
    for (int64_t i = 0; i < streams; ++i) h_w[i] /= sum;
    cpu_ref(h_in.data(), h_w.data(), h_ref.data(), batch, seq, dim, streams);

    void *d_in, *d_w, *d_out;
    CHECK_ACL(aclrtMalloc(&d_in, in_sz * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_w, streams * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_out, out_sz * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(d_in, in_sz * 4, h_in.data(), in_sz * 4, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_w, streams * 4, h_w.data(), streams * 4, ACL_MEMCPY_HOST_TO_DEVICE));

    aclrtStream stream; CHECK_ACL(aclrtCreateStream(&stream));
    mhc_post_do_fp32(batch * streams, stream, (uint8_t*)d_in, (uint8_t*)d_w, (uint8_t*)d_out, batch, seq, dim, streams);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(h_out.data(), out_sz * 4, d_out, out_sz * 4, ACL_MEMCPY_DEVICE_TO_HOST));

    bool pass = bit_exact(h_out.data(), h_ref.data(), out_sz);
    printf("  [fp32] %s: %s\n", name, pass ? "PASS" : "FAIL");

    CHECK_ACL(aclrtFree(d_in)); CHECK_ACL(aclrtFree(d_w)); CHECK_ACL(aclrtFree(d_out));
    CHECK_ACL(aclrtDestroyStream(stream));
    return pass;
}

bool test_fp16(const char* name, int64_t batch, int64_t seq, int64_t dim, int64_t streams) {
    int64_t in_sz = batch * seq * dim;
    int64_t out_sz = batch * streams * seq * dim;
    std::vector<float> h_in_f(in_sz), h_w_f(streams), h_ref(out_sz), h_out_f(out_sz);
    std::vector<uint16_t> h_in(in_sz), h_w(streams), h_out(out_sz);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_pos(0.1f, 1.1f);
    for (int64_t i = 0; i < in_sz; ++i) {
        h_in_f[i] = dist(rng);
        h_in[i] = float_to_half(h_in_f[i]);
        h_in_f[i] = half_to_float(h_in[i]);
    }
    float sum = 0;
    for (int64_t i = 0; i < streams; ++i) { h_w_f[i] = dist_pos(rng); sum += h_w_f[i]; }
    for (int64_t i = 0; i < streams; ++i) {
        h_w_f[i] /= sum;
        h_w[i] = float_to_half(h_w_f[i]);
        h_w_f[i] = half_to_float(h_w[i]);
    }
    cpu_ref(h_in_f.data(), h_w_f.data(), h_ref.data(), batch, seq, dim, streams);

    void *d_in, *d_w, *d_out;
    CHECK_ACL(aclrtMalloc(&d_in, in_sz * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_w, streams * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_out, out_sz * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(d_in, in_sz * 2, h_in.data(), in_sz * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_w, streams * 2, h_w.data(), streams * 2, ACL_MEMCPY_HOST_TO_DEVICE));

    aclrtStream stream; CHECK_ACL(aclrtCreateStream(&stream));
    mhc_post_do_fp16(batch * streams, stream, (uint8_t*)d_in, (uint8_t*)d_w, (uint8_t*)d_out, batch, seq, dim, streams);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(h_out.data(), out_sz * 2, d_out, out_sz * 2, ACL_MEMCPY_DEVICE_TO_HOST));

    for (int64_t i = 0; i < out_sz; ++i) h_out_f[i] = half_to_float(h_out[i]);
    bool pass = allclose(h_out_f.data(), h_ref.data(), out_sz, 1e-4f, 1e-3f);
    printf("  [fp16] %s: %s\n", name, pass ? "PASS" : "FAIL");

    CHECK_ACL(aclrtFree(d_in)); CHECK_ACL(aclrtFree(d_w)); CHECK_ACL(aclrtFree(d_out));
    CHECK_ACL(aclrtDestroyStream(stream));
    return pass;
}

bool test_bf16(const char* name, int64_t batch, int64_t seq, int64_t dim, int64_t streams) {
    int64_t in_sz = batch * seq * dim;
    int64_t out_sz = batch * streams * seq * dim;
    std::vector<float> h_in_f(in_sz), h_w_f(streams), h_ref(out_sz), h_out_f(out_sz);
    std::vector<uint16_t> h_in(in_sz), h_out(out_sz);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_pos(0.1f, 1.1f);
    for (int64_t i = 0; i < in_sz; ++i) {
        h_in_f[i] = dist(rng);
        h_in[i] = float_to_bf16(h_in_f[i]);
        h_in_f[i] = bf16_to_float(h_in[i]);
    }
    float sum = 0;
    for (int64_t i = 0; i < streams; ++i) { h_w_f[i] = dist_pos(rng); sum += h_w_f[i]; }
    for (int64_t i = 0; i < streams; ++i) h_w_f[i] /= sum;
    cpu_ref(h_in_f.data(), h_w_f.data(), h_ref.data(), batch, seq, dim, streams);

    void *d_in, *d_w, *d_out;
    CHECK_ACL(aclrtMalloc(&d_in, in_sz * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_w, streams * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&d_out, out_sz * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(d_in, in_sz * 2, h_in.data(), in_sz * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(d_w, streams * 4, h_w_f.data(), streams * 4, ACL_MEMCPY_HOST_TO_DEVICE));

    aclrtStream stream; CHECK_ACL(aclrtCreateStream(&stream));
    mhc_post_do_bf16(batch * streams, stream, (uint8_t*)d_in, (uint8_t*)d_w, (uint8_t*)d_out, batch, seq, dim, streams);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(h_out.data(), out_sz * 2, d_out, out_sz * 2, ACL_MEMCPY_DEVICE_TO_HOST));

    for (int64_t i = 0; i < out_sz; ++i) h_out_f[i] = bf16_to_float(h_out[i]);
    bool pass = allclose(h_out_f.data(), h_ref.data(), out_sz, 1e-3f, 4e-3f);
    printf("  [bf16] %s: %s\n", name, pass ? "PASS" : "FAIL");

    CHECK_ACL(aclrtFree(d_in)); CHECK_ACL(aclrtFree(d_w)); CHECK_ACL(aclrtFree(d_out));
    CHECK_ACL(aclrtDestroyStream(stream));
    return pass;
}

bool test_case(const char* name, int64_t b, int64_t s, int64_t d, int64_t n) {
    printf("\n[%s] batch=%ld seq=%ld dim=%ld streams=%ld\n", name, b, s, d, n);
    bool pass = true;
    pass &= test_fp32(name, b, s, d, n);
    pass &= test_fp16(name, b, s, d, n);
    pass &= test_bf16(name, b, s, d, n);
    return pass;
}

int main() {
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));

    printf("=== mhc_post Edge Cases Test ===\n");
    printf("Precision: fp32=bit-exact, fp16=allclose(1e-4,1e-3), bf16=allclose(1e-3,4e-3)\n");

    bool all_pass = true;

    // Unaligned dim
    all_pass &= test_case("dim=7", 2, 16, 7, 4);
    all_pass &= test_case("dim=13", 2, 16, 13, 4);
    all_pass &= test_case("dim=63", 2, 16, 63, 4);
    all_pass &= test_case("dim=127", 2, 16, 127, 4);
    all_pass &= test_case("dim=1", 2, 16, 1, 4);

    // Large batch
    all_pass &= test_case("batch=64", 64, 8, 64, 4);

    // Edge seq/streams
    all_pass &= test_case("seq=1", 4, 1, 128, 4);
    all_pass &= test_case("streams=1", 4, 32, 64, 1);

    // Minimal
    all_pass &= test_case("minimal", 1, 1, 1, 1);
    all_pass &= test_case("combo", 1, 1, 7, 1);

    printf("\n=== Final: %s ===\n", all_pass ? "ALL PASS" : "SOME FAILED");

    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return all_pass ? 0 : 1;
}
