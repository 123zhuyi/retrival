/**
 * @file test_aclnn_bitwise_not.cpp
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * ============================================================================
 * BitwiseNot (experimental/math/bitwise_not) eager 调用示例 —— 单输入按位/逻辑取反。
 *
 * 调用路径:
 *   本示例走 aclnn 两段式接口:
 *       aclnnBitwiseNotGetWorkspaceSize(self, out, &workspaceSize, &executor);
 *       aclnnBitwiseNot(workspace, workspaceSize, executor, stream);
 *
 * 算子语义（详见 docs/aclnnBitwiseNot.md）:
 *   - 整型 (int8/int16/int32/int64/uint8): 按位取反 out = ~self
 *     （对有符号等价 -self-1，与 numpy.invert / torch.bitwise_not 严格一致）
 *   - bool: 逻辑非 out = (self == 0) ? 1 : 0   (0<->1，非裸位翻转)
 *   单输入单输出，out.shape == self.shape，out.dtype == self.dtype，ND，无 broadcast，无属性。
 *
 * 精度判据: bitwise exact —— atol=0, rtol=0，逐元素严格相等（== numpy.invert）。
 *
 * 运行说明:
 *   两段式 aclnn 入口直接命中自定义算子包安装的预编译 binary kernel，无需在线 JIT，
 *   故不调用 aclSetCompileopt（避免引入 libacl_op_compiler 依赖、破坏仓库统一
 *   build.sh --run_example 链接约定）；编译运行流程由配套 run.sh 自动封装。
 * ============================================================================
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "acl/acl.h"
// 自动生成的本算子两段式 aclnn 入口（OpType BitwiseNot）。
#include "aclnn_bitwise_not.h"

#define CHECK_RET(cond, return_expr)         \
    do {                                     \
        if (!(cond)) {                       \
            return_expr;                     \
        }                                    \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

static int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

// device/stream 初始化。两段式 aclnn 入口直接命中自定义算子包安装的预编译 binary kernel，
// 无需在线 JIT，故不再调用 aclSetCompileopt。
static int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

// 构造连续 ND aclTensor（host 数据 -> device 内存 -> aclCreateTensor）。
template <typename T>
static int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                           aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));
    if (size <= 0) {
        size = 1;  // 空 tensor 占位，避免 0 字节申请
    }
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    if (!hostData.empty()) {
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), hostData.size() * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    }

    // 连续 tensor 的 strides。
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);
    return 0;
}

// 经两段式 aclnnBitwiseNot 调本算子 native kernel：
//   host self -> device -> aclnnBitwiseNotGetWorkspaceSize / aclnnBitwiseNot -> device -> outHost。
template <typename T>
static int RunBitwiseNotOnNpu(aclrtStream stream, aclDataType dtype, const std::vector<int64_t> &shape,
                                const std::vector<T> &selfHost, std::vector<T> &outHost)
{
    void *selfDev = nullptr;
    void *outDev = nullptr;
    aclTensor *self = nullptr;
    aclTensor *out = nullptr;
    void *workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    int rc = -1;

    int ret = CreateAclTensor(selfHost, shape, &selfDev, dtype, &self);
    CHECK_RET(ret == 0, goto cleanup);
    ret = CreateAclTensor(outHost, shape, &outDev, dtype, &out);
    CHECK_RET(ret == 0, goto cleanup);

    {
        // 第一段：计算 workspace + 执行器。
        aclError aclRet = aclnnBitwiseNotGetWorkspaceSize(self, out, &workspaceSize, &executor);
        CHECK_RET(aclRet == ACL_SUCCESS,
                  LOG_PRINT("aclnnBitwiseNotGetWorkspaceSize failed. ERROR: %d msg=%s\n", aclRet,
                            aclGetRecentErrMsg());
                  goto cleanup);

        if (workspaceSize > 0) {
            aclRet = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", aclRet); goto cleanup);
        }

        // 第二段：执行计算。
        aclRet = aclnnBitwiseNot(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(aclRet == ACL_SUCCESS,
                  LOG_PRINT("aclnnBitwiseNot failed. ERROR: %d msg=%s\n", aclRet, aclGetRecentErrMsg());
                  goto cleanup);
    }

    {
        aclError aclRet = aclrtSynchronizeStream(stream);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); goto cleanup);
    }

    if (!outHost.empty()) {
        aclError aclRet = aclrtMemcpy(outHost.data(), outHost.size() * sizeof(T), outDev, outHost.size() * sizeof(T),
                                      ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", aclRet); goto cleanup);
    }
    rc = 0;

cleanup:
    if (self != nullptr) aclDestroyTensor(self);
    if (out != nullptr) aclDestroyTensor(out);
    if (selfDev != nullptr) aclrtFree(selfDev);
    if (outDev != nullptr) aclrtFree(outDev);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    return rc;
}

int main()
{
    // 1. device / stream 初始化。
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    int initRet = Init(deviceId, &stream);
    CHECK_RET(initRet == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", initRet); return initRet);

    bool allPass = true;

    // ---- 示例 1: int32 按位取反（out = ~self），含负数边界。----
    {
        std::vector<int64_t> shape = {2, 4};
        std::vector<int32_t> self = {0, 1, -1, 2, 7, -8, 2147483647, -2147483648};
        std::vector<int32_t> actual(self.size(), 0);
        std::vector<int32_t> golden(self.size(), 0);
        for (size_t i = 0; i < self.size(); ++i) {
            golden[i] = static_cast<int32_t>(~self[i]);  // CPU golden == numpy.invert
        }

        int ret = RunBitwiseNotOnNpu(stream, ACL_INT32, shape, self, actual);
        CHECK_RET(ret == 0, LOG_PRINT("[FAIL] int32 case run failed\n"); allPass = false);

        if (ret == 0) {
            bool ok = (memcmp(actual.data(), golden.data(), self.size() * sizeof(int32_t)) == 0);
            LOG_PRINT("[%s] int32  ~self bitwise-exact\n", ok ? "PASS" : "FAIL");
            for (size_t i = 0; i < self.size(); ++i) {
                LOG_PRINT("    self[%zu]=%d  out=%d  golden=%d\n", i, self[i], actual[i], golden[i]);
            }
            allPass = allPass && ok;
        }
    }

    // ---- 示例 2: int8 边界（~(-128)=127, ~0=-1, ~127=-128）。----
    {
        std::vector<int64_t> shape = {5};
        std::vector<int8_t> self = {-128, -1, 0, 1, 127};
        std::vector<int8_t> actual(self.size(), 0);
        std::vector<int8_t> golden(self.size(), 0);
        for (size_t i = 0; i < self.size(); ++i) {
            golden[i] = static_cast<int8_t>(~self[i]);
        }
        int ret = RunBitwiseNotOnNpu(stream, ACL_INT8, shape, self, actual);
        CHECK_RET(ret == 0, LOG_PRINT("[FAIL] int8 case run failed\n"); allPass = false);
        if (ret == 0) {
            bool ok = (memcmp(actual.data(), golden.data(), self.size() * sizeof(int8_t)) == 0);
            LOG_PRINT("[%s] int8   ~self bitwise-exact (boundary ~(-128)=%d)\n", ok ? "PASS" : "FAIL", actual[0]);
            allPass = allPass && ok;
        }
    }

    // ---- 示例 3: uint8 边界（~0=255, ~255=0）。----
    {
        std::vector<int64_t> shape = {4};
        std::vector<uint8_t> self = {0, 1, 128, 255};
        std::vector<uint8_t> actual(self.size(), 0);
        std::vector<uint8_t> golden(self.size(), 0);
        for (size_t i = 0; i < self.size(); ++i) {
            golden[i] = static_cast<uint8_t>(~self[i]);
        }
        int ret = RunBitwiseNotOnNpu(stream, ACL_UINT8, shape, self, actual);
        CHECK_RET(ret == 0, LOG_PRINT("[FAIL] uint8 case run failed\n"); allPass = false);
        if (ret == 0) {
            bool ok = (memcmp(actual.data(), golden.data(), self.size() * sizeof(uint8_t)) == 0);
            LOG_PRINT("[%s] uint8  ~self bitwise-exact (~0=%u ~255=%u)\n", ok ? "PASS" : "FAIL", actual[0], actual[3]);
            allPass = allPass && ok;
        }
    }

    // ---- 示例 4: bool 逻辑非（0<->1, 非裸位翻转）。----
    {
        std::vector<int64_t> shape = {4};
        std::vector<uint8_t> self = {0, 1, 0, 1};  // bool 以 uint8 0/1 存储
        std::vector<uint8_t> actual(self.size(), 0);
        std::vector<uint8_t> golden(self.size(), 0);
        for (size_t i = 0; i < self.size(); ++i) {
            golden[i] = (self[i] == 0) ? 1 : 0;  // 逻辑非
        }
        int ret = RunBitwiseNotOnNpu(stream, ACL_BOOL, shape, self, actual);
        CHECK_RET(ret == 0, LOG_PRINT("[FAIL] bool case run failed\n"); allPass = false);
        if (ret == 0) {
            bool ok = (memcmp(actual.data(), golden.data(), self.size() * sizeof(uint8_t)) == 0);
            LOG_PRINT("[%s] bool   !self (0<->1) bitwise-exact\n", ok ? "PASS" : "FAIL");
            allPass = allPass && ok;
        }
    }

    LOG_PRINT("==== BitwiseNot eager example: %s ====\n", allPass ? "ALL PASS" : "SOME FAILED");

    // 资源释放 / 去初始化。
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return allPass ? 0 : 1;
}
