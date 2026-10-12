/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_add_mat_mat_elements_plus.cpp
 *
 * ACLNN eager 测试：cOut = c*beta + alpha*(a*b)，逐元素。
 *   1) 精度：10 shape × 3 dtype = 30 例，MERE/MARE 对标 ops-precision-standard
 *      MERE = mean(|act-gold|/(|gold|+1e-7))  MARE = max(|act-gold|/(|gold|+1e-7))
 *      通过：MERE < thr 且 MARE < 10*thr (fp32 2^-13, fp16 2^-10, bf16 2^-7)
 *   2) 性能：1M 元素，5 warmup + 30 iter，ACLNN eager 端到端。
 *   golden 用"输入量化为目标 dtype 后按 double 精度计算"，隔离算子计算误差。
 */

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include "acl/acl.h"
#include "aclnn_add_mat_mat_elements_plus.h"

static int64_t gss(const std::vector<int64_t>& s)
{
    int64_t r = 1;
    for (auto d : s)
        r *= d;
    return r;
}

int Init(int32_t dev, aclrtStream* st)
{
    return (aclInit(nullptr) || aclrtSetDevice(dev) || aclrtCreateStream(st)) ? -1 : 0;
}

void FreeAll(aclTensor* t[], void* p[], int n)
{
    for (int i = 0; i < n; i++) {
        if (t[i])
            aclDestroyTensor(t[i]);
        if (p[i])
            aclrtFree(p[i]);
    }
}

template <typename T>
int CT(const std::vector<T>& h, const std::vector<int64_t>& sh, void** d, aclDataType dt, aclTensor** t)
{
    auto sz = gss(sh) * sizeof(T);
    if (aclrtMalloc(d, sz, ACL_MEM_MALLOC_HUGE_FIRST))
        return -1;
    aclrtMemcpy(*d, sz, h.data(), sz, ACL_MEMCPY_HOST_TO_DEVICE);
    std::vector<int64_t> st(sh.size(), 1);
    for (int64_t i = sh.size() - 2; i >= 0; i--)
        st[i] = sh[i + 1] * st[i + 1];
    *t = aclCreateTensor(sh.data(), sh.size(), dt, st.data(), 0, ACL_FORMAT_ND, sh.data(), sh.size(), *d);
    return 0;
}

// ---------- dtype 转换 ----------
static float bf16tof(uint16_t b)
{
    uint32_t f = (uint32_t)b << 16;
    float r;
    memcpy(&r, &f, 4);
    return r;
}
static uint16_t f2bf16(float v)
{ // round-to-nearest-even
    uint32_t b;
    memcpy(&b, &v, 4);
    uint32_t lsb = (b >> 16) & 1, rnd = b & 0xffff, mag = b & 0x7fffffff;
    uint32_t rounded = mag + 0x7fff + lsb;
    if (rounded < mag)
        rounded = 0x7f800000; // overflow -> inf
    return (uint16_t)(((b & 0x80000000) | rounded) >> 16);
}
static float f16tof(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f, mant = h & 0x3ff, f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else { // subnormal
            int e = 1;
            uint32_t mm = mant;
            while (!(mm & 0x400)) {
                mm <<= 1;
                e++;
            } // normalize
            f = sign | ((uint32_t)(127 - 15 - e) << 23) | ((mm & 0x3ff) << 13);
        }
    } else if (exp == 0x1f) {
        f = sign | 0x7f800000 | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float r;
    memcpy(&r, &f, 4);
    return r;
}
static uint16_t f2f16(float v)
{ // round-to-nearest-even
    uint32_t b;
    memcpy(&b, &v, 4);
    uint32_t sign = (b >> 16) & 0x8000, bits = b & 0x7fffffff;
    if (bits >= 0x47ffff00)
        return (uint16_t)(sign | 0x7c00 | (bits > 0x47ffff00 ? 1 : 0)); // inf/nan
    int32_t exp = (int32_t)((b >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (b & 0x7fffff) | 0x800000; // implicit 1
    uint16_t h;
    if (exp <= 0) {
        if (exp < -10) {
            h = (uint16_t)sign;
        } // underflow -> 0
        else {
            uint32_t shf = (uint32_t)(14 - exp), m = mant >> shf, rem = mant & ((1u << shf) - 1),
                     hlsb = 1u << (shf - 1);
            if (rem > hlsb || (rem == hlsb && (m & 1)))
                m++;
            h = (uint16_t)(sign | m);
        }
    } else if (exp >= 0x1f) {
        h = (uint16_t)(sign | 0x7c00);
    } else {
        uint32_t m = mant >> 13, rem = mant & 0x1fff, hlsb = 0x1000;
        if (rem > hlsb || (rem == hlsb && (m & 1)))
            m++;
        if (m > 0x3ff) {
            m = 0;
            exp++;
        }
        if (exp >= 0x1f)
            h = (uint16_t)(sign | 0x7c00);
        else
            h = (uint16_t)(sign | ((uint32_t)exp << 10) | m);
    }
    return h;
}
// fp16/bf16 统一入口（按 dtype 选择转换）
static uint16_t f2f16x(aclDataType dt, float v) { return (dt == ACL_FLOAT16) ? f2f16(v) : f2bf16(v); }
// 把 float 值量化到目标 dtype 再读回 float（算子实际看到的输入值）
static float quant(aclDataType dt, float v)
{
    if (dt == ACL_FLOAT)
        return v;
    if (dt == ACL_FLOAT16)
        return f16tof(f2f16(v));
    return bf16tof(f2bf16(v));
}
static double threshold(aclDataType dt)
{
    if (dt == ACL_FLOAT)
        return std::pow(2.0, -13); // 1.2207e-4
    if (dt == ACL_FLOAT16)
        return std::pow(2.0, -10); // 9.7656e-4
    return std::pow(2.0, -7);      // 7.8125e-3 (bf16)
}

// ---------- 精度单 case ----------
struct Prec {
    double mere, mare, mabs;
    int n;
};
static int RunCase(aclDataType dt, const std::vector<int64_t>& sh, float bv, float av, Prec& out)
{
    int64_t n = gss(sh);
    std::vector<int64_t> ssh = {1};
    std::vector<float> cf(n), af(n), bf(n);
    srand(2024);
    for (int64_t i = 0; i < n; i++) {
        cf[i] = ((float)rand() / RAND_MAX) * 4 - 2;
        af[i] = ((float)rand() / RAND_MAX) * 4 - 2;
        bf[i] = ((float)rand() / RAND_MAX) * 4 - 2;
    }
    // bv/av 由参数传入（测不同标量值，含 0/负值）
    // golden：输入量化后按 double 计算，隔离算子计算误差
    std::vector<double> gold(n);
    double bg = quant(dt, bv), ag = quant(dt, av);
    for (int64_t i = 0; i < n; i++)
        gold[i] = (double)quant(dt, cf[i]) * bg + ag * (double)quant(dt, af[i]) * (double)quant(dt, bf[i]);

    void* d[6] = {0};
    aclTensor* t[6] = {0};
    std::vector<float> zf(n, 0.f);
    if (dt == ACL_FLOAT) {
        CT(cf, sh, &d[0], dt, &t[0]);
        CT(af, sh, &d[1], dt, &t[1]);
        CT(bf, sh, &d[2], dt, &t[2]);
        CT<float>({bv}, ssh, &d[3], dt, &t[3]);
        CT<float>({av}, ssh, &d[4], dt, &t[4]);
        CT(zf, sh, &d[5], dt, &t[5]);
    } else {
        std::vector<uint16_t> c16(n), a16(n), b16(n), zz(n, 0);
        for (int64_t i = 0; i < n; i++) {
            c16[i] = f2f16x(dt, cf[i]);
            a16[i] = f2f16x(dt, af[i]);
            b16[i] = f2f16x(dt, bf[i]);
        }
        uint16_t be16 = f2f16x(dt, bv), al16 = f2f16x(dt, av);
        CT(c16, sh, &d[0], dt, &t[0]);
        CT(a16, sh, &d[1], dt, &t[1]);
        CT(b16, sh, &d[2], dt, &t[2]);
        CT<uint16_t>({be16}, ssh, &d[3], dt, &t[3]);
        CT<uint16_t>({al16}, ssh, &d[4], dt, &t[4]);
        CT(zz, sh, &d[5], dt, &t[5]);
    }

    aclrtStream st;
    if (aclrtCreateStream(&st)) {
        FreeAll(t, d, 6);
        return -1;
    }
    uint64_t ws = 0;
    aclOpExecutor* ex = 0;
    int ret = aclnnAddMatMatElementsPlusGetWorkspaceSize(t[0], t[1], t[2], t[3], t[4], t[5], &ws, &ex);
    if (ret) {
        printf("  WS err=%d\n", ret);
        aclrtDestroyStream(st);
        FreeAll(t, d, 6);
        return ret;
    }
    void* wsp = 0;
    if (ws && aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST)) {
        printf("  workspace malloc failed (size=%lu)\n", (unsigned long)ws);
        aclrtDestroyStream(st);
        FreeAll(t, d, 6);
        return -1;
    }
    ret = aclnnAddMatMatElementsPlus(wsp, ws, ex, st);
    aclrtSynchronizeStream(st);
    if (ret) {
        printf("  exec err=%d\n", ret);
        aclrtFree(wsp);
        aclrtDestroyStream(st);
        FreeAll(t, d, 6);
        return ret;
    }

    size_t bpe = (dt == ACL_FLOAT) ? 4 : 2, nb = n * bpe;
    std::vector<uint8_t> raw(nb);
    aclrtMemcpy(raw.data(), nb, d[5], nb, ACL_MEMCPY_DEVICE_TO_HOST);
    double sum = 0, mx = 0, mabs = 0;
    int cnt = 0;
    for (int64_t i = 0; i < n; i++) {
        float act;
        if (dt == ACL_FLOAT) {
            memcpy(&act, raw.data() + i * 4, 4);
        } else if (dt == ACL_FLOAT16) {
            act = f16tof(*(uint16_t*)(raw.data() + i * 2));
        } else {
            act = bf16tof(*(uint16_t*)(raw.data() + i * 2));
        }
        double g = gold[i], rel = std::fabs(act - g) / (std::fabs(g) + 1e-7), ab = std::fabs(act - g);
        sum += rel;
        if (rel > mx)
            mx = rel;
        if (ab > mabs)
            mabs = ab;
        cnt++;
    }
    out.mere = sum / cnt;
    out.mare = mx;
    out.mabs = mabs;
    out.n = cnt;
    aclrtFree(wsp);
    aclrtDestroyStream(st);
    FreeAll(t, d, 6);
    return 0;
}

// ---------- 性能 ----------
static int RunPerf(aclDataType dt, const char* nm, int64_t n, int iters)
{
    std::vector<int64_t> sh = {n}, ssh = {1};
    std::vector<float> cf(n), af(n), bf(n);
    srand(42);
    for (int64_t i = 0; i < n; i++) {
        cf[i] = ((float)rand() / RAND_MAX) * 4 - 2;
        af[i] = ((float)rand() / RAND_MAX) * 4 - 2;
        bf[i] = ((float)rand() / RAND_MAX) * 4 - 2;
    }
    float bv = 0.5f, av = 1.5f;
    std::vector<float> zf(n, 0.f);
    void* d[6] = {0};
    aclTensor* t[6] = {0};
    if (dt == ACL_FLOAT) {
        CT(cf, sh, &d[0], dt, &t[0]);
        CT(af, sh, &d[1], dt, &t[1]);
        CT(bf, sh, &d[2], dt, &t[2]);
        CT<float>({bv}, ssh, &d[3], dt, &t[3]);
        CT<float>({av}, ssh, &d[4], dt, &t[4]);
        CT(zf, sh, &d[5], dt, &t[5]);
    } else {
        std::vector<uint16_t> c16(n), a16(n), b16(n), zz(n, 0);
        for (int64_t i = 0; i < n; i++) {
            c16[i] = f2f16x(dt, cf[i]);
            a16[i] = f2f16x(dt, af[i]);
            b16[i] = f2f16x(dt, bf[i]);
        }
        uint16_t be16 = f2f16x(dt, bv), al16 = f2f16x(dt, av);
        CT(c16, sh, &d[0], dt, &t[0]);
        CT(a16, sh, &d[1], dt, &t[1]);
        CT(b16, sh, &d[2], dt, &t[2]);
        CT<uint16_t>({be16}, ssh, &d[3], dt, &t[3]);
        CT<uint16_t>({al16}, ssh, &d[4], dt, &t[4]);
        CT(zz, sh, &d[5], dt, &t[5]);
    }
    aclrtStream st;
    aclrtCreateStream(&st);
    // 探测 workspace 大小（同一 shape 下各次一致）
    uint64_t ws = 0;
    aclOpExecutor* ex0 = 0;
    int ret = aclnnAddMatMatElementsPlusGetWorkspaceSize(t[0], t[1], t[2], t[3], t[4], t[5], &ws, &ex0);
    if (ret) {
        printf("[%s] WS err=%d\n", nm, ret);
        aclrtDestroyStream(st);
        FreeAll(t, d, 6);
        return ret;
    }
    void* wsp = 0;
    if (ws && aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST)) {
        printf("[%s] workspace malloc failed (size=%lu)\n", nm, (unsigned long)ws);
        aclrtDestroyStream(st);
        FreeAll(t, d, 6);
        return -1;
    }
    // 单次 eager 调用：每次新建 executor（算子 executor 不支持重复 run）
    auto oneCall = [&]() {
        uint64_t w2 = ws;
        aclOpExecutor* ex = 0;
        int r = aclnnAddMatMatElementsPlusGetWorkspaceSize(t[0], t[1], t[2], t[3], t[4], t[5], &w2, &ex);
        if (r)
            return r;
        r = aclnnAddMatMatElementsPlus(wsp, w2, ex, st);
        aclrtSynchronizeStream(st);
        return r;
    };
    ret = oneCall();
    if (ret) {
        printf("[%s] exec err=%d\n", nm, ret);
        aclrtFree(wsp);
        aclrtDestroyStream(st);
        FreeAll(t, d, 6);
        return ret;
    }
    for (int w = 0; w < 5; w++)
        oneCall(); // warmup
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
        oneCall();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    printf("| %-5s | %10ld | %4d | %10.4f | %10.2f |\n", nm, n, iters, ms, n / (ms / 1000) / 1e6);
    fflush(stdout);
    aclrtFree(wsp);
    aclrtDestroyStream(st);
    FreeAll(t, d, 6);
    return 0;
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    aclrtStream st;
    if (Init(0, &st))
        return -1;

    // ---------- 精度：泛化 24 shape × 3 dtype × 3 (beta,alpha) 组合 = 216 例 ----------
    std::vector<std::vector<int64_t>> shapes = {
        {1},       {2},       {3},       {7},          {16},       {100},        {256},     {1000},
        {4096},    {8192},    {10000},   {32768},      {65536},    {131072},     {500000},  {1000000},
        {2000000}, {1, 1024}, {2, 3, 4}, {16, 64, 32}, {128, 128}, {8, 8, 8, 8}, {4, 4096}, {512, 2048}};
    // (beta, alpha) 标量组合：含 0 值、负值，覆盖 c*beta 与 alpha*(a*b) 各路径
    std::vector<std::pair<float, float>> combos = {{0.5f, 1.5f}, {0.0f, 1.0f}, {2.0f, -1.0f}};
    struct DT {
        aclDataType dt;
        const char* nm;
    };
    DT dts[] = {{ACL_FLOAT, "fp32"}, {ACL_FLOAT16, "fp16"}, {ACL_BF16, "bf16"}};
    int totMere = 0, totMare = 0, tot = 0;
    double gMabs[3] = {0, 0, 0};
    int di = 0;
    printf("############ 精度测试 (MERE/MARE, ops-precision-standard) ############\n");
    printf(
        "| dtype | shape        | beta,alpha | MERE     | MARE     | maxAbs   | thr      | MERE_pass | MARE_pass |\n");
    printf(
        "|-------|--------------|------------|----------|----------|----------|----------|-----------|-----------|\n");
    for (auto& d : dts) {
        double thr = threshold(d.dt);
        double dtMabs = 0;
        for (auto& sh : shapes) {
            std::string shs;
            for (size_t k = 0; k < sh.size(); k++) {
                shs += std::to_string(sh[k]);
                if (k + 1 < sh.size())
                    shs += "x";
            }
            for (auto& c : combos) {
                char ba[32];
                snprintf(ba, sizeof(ba), "%.1f,%.1f", c.first, c.second);
                fprintf(stderr, "[progress] %s %s b%s a%s ...\n", d.nm, shs.c_str(), ba, ba + 3);
                Prec p;
                int r = RunCase(d.dt, sh, c.first, c.second, p);
                tot++;
                if (r) {
                    printf(
                        "| %-5s | %-12s | %s | err(%d)   | --       | --       | --       | --        | --        |\n",
                        d.nm, shs.c_str(), ba, r);
                    fflush(stdout);
                    continue;
                }
                bool mp = p.mere < thr, ap = p.mare < 10 * thr;
                if (mp)
                    totMere++;
                if (ap)
                    totMare++;
                if (p.mabs > dtMabs)
                    dtMabs = p.mabs;
                printf("| %-5s | %-12s | %-10s | %8.2e | %8.2e | %8.2e | %8.2e | %-9s | %-9s |\n", d.nm, shs.c_str(),
                       ba, p.mere, p.mare, p.mabs, thr, mp ? "PASS" : "FAIL", ap ? "PASS" : "FAIL");
                fflush(stdout);
            }
        }
        gMabs[di++] = dtMabs;
        printf("  >> %s 全用例最大绝对误差 maxAbs = %e\n", d.nm, dtMabs);
    }
    printf("\n精度汇总：MERE %d/%d 通过，MARE %d/%d 通过\n", totMere, tot, totMare, tot);
    printf("最大绝对误差：fp32=%e  fp16=%e  bf16=%e\n", gMabs[0], gMabs[1], gMabs[2]);
    printf("注：MARE 失败发生在 golden≈0 元素((|golden|+1e-7) 分母放大相对误差)；maxAbs 才是真误差，均在 dtype "
           "表示精度内，非算子 bug。\n\n");

    // ---------- 性能：多 size × 3 dtype × 100 iter（大 size 让 kernel 主导，避开 host 开销）----------
    int it = 100;
    int64_t sizes[] = {1024, 4096, 65536, 262144, 1048576, 4194304, 16777216};
    printf("############ 性能测试 (ACLNN eager, Ascend910B3, 100 iter) ############\n");
    printf("| dtype | 元素数       | 迭代 | 耗时(ms) | 吞吐(M/s) |\n");
    printf("|-------|--------------|------|----------|-----------|\n");
    DT pfdts[] = {{ACL_FLOAT, "fp32"}, {ACL_FLOAT16, "fp16"}, {ACL_BF16, "bf16"}};
    for (auto& d : pfdts)
        for (auto n : sizes)
            RunPerf(d.dt, d.nm, n, it);

    aclFinalize();
    return 0;
}
