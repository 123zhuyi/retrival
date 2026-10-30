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
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

// ============================================================================
// slogdet 算子 aclnn 调用示例（experimental/math/slogdet，原生 AscendC 实现，fp32）
//
// 演示 aclnnSlogdet 两段式接口（GetWorkspaceSize -> Execute）在真机 NPU 上计算
// batch 方阵的行列式符号 signOut 与行列式绝对值的自然对数 logOut，对标
// torch.linalg.slogdet。示例同时给出 CPU 带部分主元 LU 的 golden（与 torch /
// numpy 等价），将 NPU 输出与 golden 逐 batch 比对（sign 精确、log rtol/atol=1e-4），
// 并覆盖一个奇异矩阵（det=0 → logOut=-inf、signOut=0）以演示退化语义。
//
// 链接说明（关键）：示例 CMake 用 `-Wl,--no-as-needed ${CUSTOM_OP_LIBRARY}` 将自定义
//   libcust_opapi.so 置最前，保证运行时命中本算子的自定义 AscendC kernel（Slogdet），
//   而非上游 math/slogdet 转发壳（LogMatrixDeterminant）。
// ============================================================================

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "acl/acl.h"
#include "aclnn_slogdet_native.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// fp32 浮点社区精度标准（双万分之一）：log 输出 rtol/atol = 1e-4。
static constexpr double kRtol = 1.0e-4;
static constexpr double kAtol = 1.0e-4;

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 空 batch（标量输出 []）也分配 1 元素占位
    size_t allocSize = size == 0 ? sizeof(T) : size;
    // 调用 aclrtMalloc 申请 device 侧内存
    auto ret = aclrtMalloc(deviceAddr, allocSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    if (size > 0) {
        // 调用 aclrtMemcpy 将 host 侧数据拷贝到 device 侧内存
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    }

    // 计算连续 tensor 的 strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用 aclCreateTensor 接口创建 aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

// ============================================================================
// CPU Golden：带部分主元（partial pivoting）的 LU 分解 → (sign, log|det|)。
//   对单个 n×n 行主序方阵计算 slogdet，与 torch.linalg.slogdet / numpy 等价：
//     - 部分主元：第 k 步在第 k 列 k..n-1 行选 |max| 作主元并交换（记录置换奇偶性）。
//     - log 域累加：logOut = Σ log|U_ii|（规避连乘上溢/下溢）。
//     - 奇异（主元绝对值 <= 相对阈值）：logOut=-inf、signOut=0。
// ============================================================================
static void ComputeSlogdetGoldenOne(const float* mat, int64_t n, float& outSign, float& outLog)
{
    std::vector<double> M(static_cast<size_t>(n * n));
    double maxAbs = 0.0;
    for (int64_t i = 0; i < n * n; ++i) {
        double v = static_cast<double>(mat[i]);
        double a = std::abs(v);
        if (a > maxAbs) maxAbs = a;
        M[static_cast<size_t>(i)] = v;
    }
    // LAPACK 风格相对奇异阈值 n·FLT_EPSILON(fp32)·maxAbs，与 NPU kernel 同量级
    constexpr double kFltEpsF32 = 1.1920929e-7;
    double singularEps = static_cast<double>(n) * kFltEpsF32 * maxAbs;

    double sign = 1.0;
    double logabs = 0.0;
    for (int64_t k = 0; k < n; ++k) {
        int64_t piv = k;
        double maxv = std::abs(M[static_cast<size_t>(k * n + k)]);
        for (int64_t i = k + 1; i < n; ++i) {
            double cur = std::abs(M[static_cast<size_t>(i * n + k)]);
            if (cur > maxv) {
                maxv = cur;
                piv = i;
            }
        }
        double pivot = M[static_cast<size_t>(piv * n + k)];
        if (std::abs(pivot) <= singularEps) {
            outSign = 0.0f;
            outLog = -std::numeric_limits<float>::infinity();
            return;
        }
        if (piv != k) {
            for (int64_t j = 0; j < n; ++j) {
                std::swap(M[static_cast<size_t>(k * n + j)], M[static_cast<size_t>(piv * n + j)]);
            }
            sign = -sign;
        }
        if (pivot < 0.0) sign = -sign;
        logabs += std::log(std::abs(pivot));
        for (int64_t i = k + 1; i < n; ++i) {
            double f = M[static_cast<size_t>(i * n + k)] / pivot;
            for (int64_t j = k; j < n; ++j) {
                M[static_cast<size_t>(i * n + j)] -= f * M[static_cast<size_t>(k * n + j)];
            }
        }
    }
    outSign = static_cast<float>(sign);
    outLog = static_cast<float>(logabs);
}

// 逐 batch 比对（sign 精确 {-1,0,+1}；log 含 -inf 特殊值 + rtol/atol）
static bool VerifyAgainstGolden(
    const std::vector<float>& self, int64_t batchCount, int64_t n, const std::vector<float>& signActual,
    const std::vector<float>& logActual)
{
    bool pass = true;
    const int64_t matStride = n * n;
    for (int64_t b = 0; b < batchCount; ++b) {
        float gs = 0.0f, gl = 0.0f;
        ComputeSlogdetGoldenOne(self.data() + b * matStride, n, gs, gl);
        float as = signActual[static_cast<size_t>(b)];
        float al = logActual[static_cast<size_t>(b)];

        bool signOk = (gs == as);  // 0.0 == -0.0 为 true
        bool logOk;
        if (std::isinf(gl) && gl < 0.0f) {
            logOk = std::isinf(al) && al < 0.0f;  // 奇异 -inf 精确匹配
        } else {
            logOk = std::abs(static_cast<double>(al) - gl) <= (kAtol + kRtol * std::abs(static_cast<double>(gl)));
        }
        LOG_PRINT(
            "  batch[%ld]: sign golden=%+.0f actual=%+.0f (%s) | log golden=%-12.6g actual=%-12.6g (%s)\n", (long)b,
            (double)gs, (double)as, signOk ? "OK" : "MISMATCH", (double)gl, (double)al, logOk ? "OK" : "MISMATCH");
        if (!signOk || !logOk) pass = false;
    }
    return pass;
}

int main()
{
    // 1. device/stream 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    //    self: [3, 2, 2] —— 3 个 2x2 方阵，覆盖正/负行列式与奇异（det=0）三种语义：
    //      [[2,0],[0,3]] → det=+6, sign=+1, log=ln6
    //      [[0,1],[1,0]] → det=-1, sign=-1, log=0（行交换置换）
    //      [[1,2],[2,4]] → det=0,  sign=0,  log=-inf（秩亏，演示退化语义）
    //    输出 signOut/logOut: [3]（self 去掉最后两维的 batch 形状）。
    std::vector<int64_t> selfShape = {3, 2, 2};
    std::vector<int64_t> outShape = {3};  // batch 形状
    int64_t n = selfShape.back();
    int64_t batchCount = GetShapeSize(outShape);

    std::vector<float> selfHostData = {
        2, 0, 0, 3,  // det=+6
        0, 1, 1, 0,  // det=-1
        1, 2, 2, 4,  // det=0（奇异）
    };
    std::vector<float> signOutHostData(static_cast<size_t>(batchCount), 0.0f);
    std::vector<float> logOutHostData(static_cast<size_t>(batchCount), 0.0f);

    void* selfDeviceAddr = nullptr;
    void* signOutDeviceAddr = nullptr;
    void* logOutDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* signOut = nullptr;
    aclTensor* logOut = nullptr;

    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(signOutHostData, outShape, &signOutDeviceAddr, aclDataType::ACL_FLOAT, &signOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(logOutHostData, outShape, &logOutDeviceAddr, aclDataType::ACL_FLOAT, &logOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用 aclnnSlogdet 第一段接口（参数顺序：self, signOut, logOut）
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnSlogdetGetWorkspaceSize(self, signOut, logOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSlogdetGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 4. 根据 workspaceSize 申请 device 内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 5. 调用 aclnnSlogdet 第二段接口执行计算
    ret = aclnnSlogdet(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSlogdet failed. ERROR: %d\n", ret); return ret);

    // 6. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. 将 device 侧输出拷回 host 并打印
    std::vector<float> signResult(static_cast<size_t>(batchCount), 0.0f);
    std::vector<float> logResult(static_cast<size_t>(batchCount), 0.0f);
    ret = aclrtMemcpy(
        signResult.data(), signResult.size() * sizeof(float), signOutDeviceAddr, batchCount * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy signOut failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(
        logResult.data(), logResult.size() * sizeof(float), logOutDeviceAddr, batchCount * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy logOut failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("\n==== aclnnSlogdet NPU 输出（self=[3,2,2], fp32）====\n");
    for (int64_t i = 0; i < batchCount; i++) {
        LOG_PRINT("  signOut[%ld] = %+.4f,  logOut[%ld] = %.6f\n", i, signResult[i], i, logResult[i]);
    }

    // 8. 与 CPU 带部分主元 LU golden 逐 batch 比对（sign 精确 + log rtol/atol=1e-4）
    LOG_PRINT("\n==== 与 CPU LU golden 比对（对标 torch.linalg.slogdet）====\n");
    bool pass = VerifyAgainstGolden(selfHostData, batchCount, n, signResult, logResult);
    LOG_PRINT("\n比对结果: %s\n", pass ? "PASS" : "FAIL");

    // 9. 释放资源
    aclDestroyTensor(self);
    aclDestroyTensor(signOut);
    aclDestroyTensor(logOut);
    aclrtFree(selfDeviceAddr);
    aclrtFree(signOutDeviceAddr);
    aclrtFree(logOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return pass ? 0 : 1;
}
