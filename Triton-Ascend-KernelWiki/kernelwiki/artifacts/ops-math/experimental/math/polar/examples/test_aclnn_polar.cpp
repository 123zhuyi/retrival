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
 * Polar 算子独立 aclnn 调用测试（精度 + 性能）
 *
 * 参考官方范式：cann/ops-math math/sin/examples/test_aclnn_sin.cpp（两段式 aclnn 标准调用）
 * 适配 Polar：2×float32 输入(input, angle) + complex64 输出(out=broadcast(input,angle))
 *
 * 编译（在已安装自定义 Polar 包的 NPU 环境，路径按实际 CANN/vendor 调整）：
 *   source /home/ma-user/Ascend/cann-8.5.0/set_env.sh
 *   OPP=$ASCEND_HOME_PATH/opp/vendors/customize
 *   g++ -O2 -std=c++17 test_aclnn_polar.cpp -o test_aclnn_polar \
 *       -I$ASCEND_HOME_PATH/include -I$OPP/op_api/include \
 *       -L$ASCEND_HOME_PATH/lib64 -L$OPP/op_api/lib \
 *       -lascendcl -lnnopbase -lcust_opapi
 * 运行：
 *   export LD_LIBRARY_PATH=$OPP/op_api/lib:$LD_LIBRARY_PATH
 *   ./test_aclnn_polar                       # 精度 + 自带计时
 *   msprof --application="./test_aclnn_polar" # 设备侧性能采集
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include "acl/acl.h"
#include "aclnn_polar.h" // 自定义算子生成的 aclnn 头（custom vendor op_api/include）

#define CHECK_RET(cond, expr) \
    do {                      \
        if (!(cond)) {        \
            expr;             \
        }                     \
    } while (0)
#define LOG(msg, ...)               \
    do {                            \
        printf(msg, ##__VA_ARGS__); \
    } while (0)

struct C64 {
    float re;
    float im;
}; // 与 complex64 内存布局一致（8 字节）

static int64_t ShapeSize(const std::vector<int64_t>& s)
{
    int64_t n = 1;
    for (auto v : s)
        n *= v;
    return n;
}

static int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclInit failed %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclrtSetDevice failed %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclrtCreateStream failed %d\n", ret); return ret);
    return 0;
}

template <typename T>
static int CreateAclTensor(const std::vector<T>& host, const std::vector<int64_t>& shape, void** devAddr,
                           aclDataType dt, aclTensor** tensor)
{
    auto bytes = ShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(devAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclrtMalloc failed %d\n", ret); return ret);
    ret = aclrtMemcpy(*devAddr, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclrtMemcpy H2D failed %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = (int64_t)shape.size() - 2; i >= 0; --i)
        strides[i] = shape[i + 1] * strides[i + 1];
    *tensor = aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *devAddr);
    return 0;
}

// numpy 右对齐广播 out shape
static std::vector<int64_t> BroadcastShape(const std::vector<int64_t>& a, const std::vector<int64_t>& b)
{
    size_t r = std::max(a.size(), b.size());
    std::vector<int64_t> o(r, 1);
    for (size_t i = 0; i < r; ++i) {
        int64_t da = (i + a.size() >= r) ? a[i + a.size() - r] : 1;
        int64_t db = (i + b.size() >= r) ? b[i + b.size() - r] : 1;
        o[i] = std::max(da, db);
    }
    return o;
}

// 一组用例：构造 input/angle，调用 aclnnPolar（warmup + 计时重复），校验 + 报告
static int RunCase(const char* name, aclrtStream stream, const std::vector<int64_t>& inShape,
                   const std::vector<int64_t>& anShape, int repeat = 100)
{
    auto outShape = BroadcastShape(inShape, anShape);
    int64_t inN = ShapeSize(inShape), anN = ShapeSize(anShape), outN = ShapeSize(outShape);

    std::vector<float> inH(inN), anH(anN);
    for (int64_t i = 0; i < inN; ++i)
        inH[i] = 0.5f + (float)(i % 17);        // abs ≥ 0 区间
    for (int64_t i = 0; i < anN; ++i)
        anH[i] = -3.1415926f + (float)(i % 11); // 角度
    std::vector<C64> outH(outN, {0.f, 0.f});

    void *inD = nullptr, *anD = nullptr, *outD = nullptr;
    aclTensor *in = nullptr, *an = nullptr, *out = nullptr;
    CHECK_RET(CreateAclTensor(inH, inShape, &inD, ACL_FLOAT, &in) == 0, return -1);
    CHECK_RET(CreateAclTensor(anH, anShape, &anD, ACL_FLOAT, &an) == 0, return -1);
    CHECK_RET(CreateAclTensor(outH, outShape, &outD, ACL_COMPLEX64, &out) == 0, return -1);

    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnPolarGetWorkspaceSize(in, an, out, &wsSize, &exec);
    CHECK_RET(ret == ACL_SUCCESS, LOG("[%s] GetWorkspaceSize failed %d\n", name, ret); return ret);
    void* wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG("[%s] ws malloc failed %d\n", name, ret); return ret);
    }

    // warmup
    for (int i = 0; i < 10; ++i) {
        ret = aclnnPolar(wsAddr, wsSize, exec, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG("[%s] aclnnPolar failed %d\n", name, ret); return ret);
    }
    aclrtSynchronizeStream(stream);

    // 计时（壁钟，msprof 另采设备时）
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; ++i) {
        ret = aclnnPolar(wsAddr, wsSize, exec, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG("[%s] aclnnPolar failed %d\n", name, ret); return ret);
    }
    aclrtSynchronizeStream(stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / repeat;

    // 读回 + CPU 参考校验（polar = abs*(cos+ i sin)，按广播规则）
    aclrtMemcpy(outH.data(), outN * sizeof(C64), outD, outN * sizeof(C64), ACL_MEMCPY_DEVICE_TO_HOST);
    int64_t err = 0;
    for (int64_t g = 0; g < outN; ++g) {
        // 由 g 反解广播下的 input/angle 元素（行主序，size==1 维不进位）
        int64_t inIdx = 0, anIdx = 0, accIn = 1, accAn = 1, rem = g;
        for (int64_t d = (int64_t)outShape.size() - 1; d >= 0; --d) {
            int64_t c = rem % outShape[d];
            rem /= outShape[d];
            int64_t di = (d + (int64_t)inShape.size() >= (int64_t)outShape.size()) ?
                             inShape[d + inShape.size() - outShape.size()] :
                             1;
            int64_t da = (d + (int64_t)anShape.size() >= (int64_t)outShape.size()) ?
                             anShape[d + anShape.size() - outShape.size()] :
                             1;
            if (di != 1)
                inIdx += (c % di) * accIn;
            if (da != 1)
                anIdx += (c % da) * accAn;
            accIn *= di;
            accAn *= da;
        }
        float a = inH[inIdx % inN], th = anH[anIdx % anN];
        float er = a * std::cos(th), ei = a * std::sin(th);
        if (std::fabs(er - outH[g].re) > 1e-4f || std::fabs(ei - outH[g].im) > 1e-4f)
            ++err;
    }
    LOG("[%-22s] in=%lldel an=%lldel out=%lldel  avg=%.2f us/call  err=%lld/%lld  %s\n", name, (long long)inN,
        (long long)anN, (long long)outN, us, (long long)err, (long long)outN, (err <= outN * 1e-4 ? "PASS" : "FAIL"));

    aclDestroyTensor(in);
    aclDestroyTensor(an);
    aclDestroyTensor(out);
    aclrtFree(inD);
    aclrtFree(anD);
    aclrtFree(outD);
    if (wsSize > 0)
        aclrtFree(wsAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG("Init acl failed %d\n", ret); return ret);

    // 多组用例：同 shape（小/大-性能）+ 广播（多种模式）
    RunCase("same-small [2,6,10]", stream, {2, 6, 10}, {2, 6, 10});
    RunCase("same-large [4096,4096]", stream, {4096, 4096}, {4096, 4096}, 50);
    RunCase("bcast [4,1,8]x[4,5,8]", stream, {4, 1, 8}, {4, 5, 8});
    RunCase("bcast scalar [1]x[3,4,5]", stream, {1}, {3, 4, 5});
    RunCase("bcast [8,1]x[1,7]", stream, {8, 1}, {1, 7});

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
