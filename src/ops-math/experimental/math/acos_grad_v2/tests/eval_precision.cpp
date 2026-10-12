/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file eval_precision.cpp
 * @brief AcosGradV2 精度评测（ascendc-evaluation 方法论：多用例 NPU vs CPU-golden，atol/rtol）
 *
 * golden = fp64: z = -dy / sqrt(1 - y^2)   （|y|>1 → NaN，跳过）
 * 容差: FP32 1e-5 / FP16 1e-3 / BF16 1e-2
 * 用例: 3 dtype × 11 shape = 33 例（1D-4D、小/中/大、对齐/非对齐）
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "../op_api/aclnn_acos_grad_v2.h"

enum class DT { FP32, FP16, BF16 };
static int64_t SS(const std::vector<int64_t>& s)
{
    int64_t n = 1;
    for (auto d : s)
        n *= d;
    return n;
}
static void Init(int d, aclrtStream* st)
{
    aclInit(nullptr);
    aclrtSetDevice(d);
    aclrtCreateStream(st);
}

static void EncF(float v, uint8_t* o) { std::memcpy(o, &v, 4); }
static void EncH(float v, uint8_t* o)
{
    uint32_t x;
    std::memcpy(&x, &v, 4);
    uint16_t sg = (x >> 31) & 1;
    int32_t e = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    uint16_t h = e <= 0 ? (sg << 15) : (e >= 31 ? ((sg << 15) | (0x1f << 10)) : ((sg << 15) | (e << 10) | (m >> 13)));
    std::memcpy(o, &h, 2);
}
static float DecH(const uint8_t* i)
{
    uint16_t h;
    std::memcpy(&h, i, 2);
    uint32_t sg = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0)
        return m == 0 ? (sg ? -0.0f : 0.0f) : (sg ? -1 : 1) * (m / 1024.0f / 1024.0f);
    if (e == 31)
        return m ? NAN : (sg ? -INFINITY : INFINITY);
    float v = (1.0f + m / 1024.0f) * std::pow(2.0f, (int)e - 15);
    return sg ? -v : v;
}
static void EncB(float v, uint8_t* o)
{
    uint32_t x;
    std::memcpy(&x, &v, 4);
    uint16_t b = (uint16_t)((x + 0x7FFFU + ((x >> 16) & 1)) >> 16);
    std::memcpy(o, &b, 2);
}
static float DecB(const uint8_t* i)
{
    uint16_t b;
    std::memcpy(&b, i, 2);
    uint32_t x = (uint32_t)b << 16;
    float v;
    std::memcpy(&v, &x, 4);
    return v;
}
static void Enc(float v, DT dt, uint8_t* o)
{
    if (dt == DT::FP32)
        EncF(v, o);
    else if (dt == DT::FP16)
        EncH(v, o);
    else
        EncB(v, o);
}
static float Dec(const uint8_t* i, DT dt)
{
    if (dt == DT::FP32) {
        float v;
        std::memcpy(&v, i, 4);
        return v;
    }
    return dt == DT::FP16 ? DecH(i) : DecB(i);
}
static size_t Dsz(DT dt) { return dt == DT::FP32 ? 4 : 2; }
static aclDataType AD(DT dt) { return dt == DT::FP32 ? ACL_FLOAT : (dt == DT::FP16 ? ACL_FLOAT16 : ACL_BF16); }
static const char* DN(DT dt) { return dt == DT::FP32 ? "FP32" : (dt == DT::FP16 ? "FP16" : "BF16"); }

int MkT(const std::vector<uint8_t>& b, const std::vector<int64_t>& sh, void** d, aclDataType t, aclTensor** tt)
{
    aclrtMalloc(d, b.size(), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(*d, b.size(), b.data(), b.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    std::vector<int64_t> st(sh.size(), 1);
    for (int64_t i = sh.size() - 2; i >= 0; i--)
        st[i] = sh[i + 1] * st[i + 1];
    *tt = aclCreateTensor(sh.data(), sh.size(), t, st.data(), 0, ACL_FORMAT_ND, sh.data(), sh.size(), *d);
    return 0;
}

struct R {
    DT dt;
    int total;
    int pass;
    bool ok;
    double mx;
};
static R Run(const std::vector<int64_t>& sh, DT dt, float ylo, float yhi, float atol, float rtol, aclrtStream st)
{
    int64_t N = SS(sh);
    size_t ds = Dsz(dt);
    std::vector<uint8_t> yb(N * ds), dyb(N * ds);
    for (int64_t i = 0; i < N; i++) {
        float yv = ylo + (yhi - ylo) * (float)i / (float)N, dv = -1.0f + 2.0f * (float)i / (float)N;
        Enc(yv, dt, yb.data() + i * ds);
        Enc(dv, dt, dyb.data() + i * ds);
    }
    aclTensor* yT = nullptr;
    void* yD = nullptr;
    MkT(yb, sh, &yD, AD(dt), &yT);
    aclTensor* dyT = nullptr;
    void* dyD = nullptr;
    MkT(dyb, sh, &dyD, AD(dt), &dyT);
    std::vector<uint8_t> zb(N * ds, 0);
    aclTensor* zT = nullptr;
    void* zD = nullptr;
    MkT(zb, sh, &zD, AD(dt), &zT);
    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    int ret = aclnnAcosGradV2GetWorkspaceSize(yT, dyT, zT, &ws, &ex);
    void* wa = nullptr;
    if (ws > 0)
        aclrtMalloc(&wa, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret == 0)
        ret = aclnnAcosGradV2(wa, ws, ex, st);
    aclrtSynchronizeStream(st);
    std::vector<uint8_t> rb(N * ds, 0);
    aclrtMemcpy(rb.data(), rb.size(), zD, rb.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    int pass = 0, dom = 0;
    double mx = 0;
    for (int64_t i = 0; i < N; i++) {
        float yq = Dec(yb.data() + i * ds, dt), dq = Dec(dyb.data() + i * ds, dt), rs = Dec(rb.data() + i * ds, dt);
        double omy = 1.0 - (double)yq * (double)yq;
        if (omy <= 0)
            continue;
        dom++;
        double g = (double)dq * (-1.0 / std::sqrt(omy));
        if (std::isnan(rs) || std::isinf(rs))
            continue;
        double err = std::fabs((double)rs - g), tol = atol + rtol * std::fabs(g),
               re = std::fabs(g) > 0 ? err / std::fabs(g) : 0;
        if (re > mx)
            mx = re;
        if (err <= tol)
            pass++;
    }
    R r;
    r.dt = dt;
    r.total = dom > 0 ? dom : (int)N;
    r.pass = pass;
    r.mx = mx;
    r.ok = (ret == 0) && (pass == r.total);
    aclDestroyTensor(yT);
    aclDestroyTensor(dyT);
    aclDestroyTensor(zT);
    aclrtFree(yD);
    aclrtFree(dyD);
    aclrtFree(zD);
    if (ws > 0)
        aclrtFree(wa);
    return r;
}

int main()
{
    int dev = 0;
    aclrtStream st;
    Init(dev, &st);
    std::vector<std::vector<int64_t>> sh = {{8},        {255},     {1024},        {4096},       {17, 31},    {32, 256},
                                            {64, 1024}, {3, 5, 7}, {2, 4, 8, 16}, {1, 3, 1, 5}, {1000, 1000}};
    struct T {
        DT dt;
        float a, r;
    } ts[] = {{DT::FP32, 1e-5f, 1e-5f}, {DT::FP16, 1e-3f, 1e-3f}, {DT::BF16, 1e-2f, 1e-2f}};
    printf("dtype,elem,max_relerr,atol/rtol,status\n");
    int ap = 0, ac = 0;
    int pd[3] = {0, 0, 0}, cd[3] = {0, 0, 0};
    for (auto& t : ts)
        for (auto& s : sh) {
            auto r = Run(s, t.dt, -0.99f, 0.99f, t.a, t.r, st);
            printf("%s,%d,%.4e,%.0e,%s\n", DN(t.dt), r.total, r.mx, t.a, r.ok ? "PASS" : "FAIL");
            cd[(int)t.dt]++;
            if (r.ok) {
                pd[(int)t.dt]++;
                ap++;
            }
            ac++;
        }
    printf("\n=== 总例数 %d, 通过 %d ===\nFP32 %d/%d FP16 %d/%d BF16 %d/%d\n结论: %s\n", ac, ap, pd[0], cd[0], pd[1],
           cd[1], pd[2], cd[2], ap == ac ? "PRECISION PASS" : "PRECISION FAIL");
    aclrtDestroyStream(st);
    aclrtResetDevice(dev);
    aclFinalize();
    return ap == ac ? 0 : 1;
}
