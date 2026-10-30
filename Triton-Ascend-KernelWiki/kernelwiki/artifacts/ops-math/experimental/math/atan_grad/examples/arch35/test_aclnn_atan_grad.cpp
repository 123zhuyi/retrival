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
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file test_aclnn_atan_grad.cpp
 * @brief aclnnAtanGrad 调用示例
 *
 * 功能：通过 ACLNN 两段式接口调用 AtanGrad 算子，计算 atan 函数的输入梯度。
 * 公式：dx_i = dy_i * (1 / (1 + x_i^2))
 *
 * 编译与运行步骤：
 *   1. 编译算子包：cd ops/atan_grad && bash build.sh --soc=ascend910b
 *   2. 安装算子包：（参照 build.sh 输出的安装指令）
 *   3. 编译本示例：
 *        g++ -std=c++17 -o test_aclnn_atan_grad test_aclnn_atan_grad.cpp \
 *            -I${ASCEND_TOOLKIT_HOME}/include \
 *            -L${ASCEND_TOOLKIT_HOME}/lib64 \
 *            -lacl_op_compiler -lascendcl \
 *            -Wl,-rpath,${ASCEND_TOOLKIT_HOME}/lib64
 *   4. 运行：./test_aclnn_atan_grad
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <numeric>

#include "acl/acl.h"
#include "aclnn_atan_grad.h"

// ============================================================================
// 辅助宏
// ============================================================================

#define CHECK_ACL(expr)                                                                                            \
    do {                                                                                                           \
        auto _ret = (expr);                                                                                        \
        if (_ret != ACL_SUCCESS) {                                                                                 \
            std::cerr << "[ERROR] " #expr " failed, ret=" << _ret << " at " __FILE__ ":" << __LINE__ << std::endl; \
            return -1;                                                                                             \
        }                                                                                                          \
    } while (0)

#define CHECK_ACLNN(expr)                                                                                          \
    do {                                                                                                           \
        auto _ret = (expr);                                                                                        \
        if (_ret != ACL_SUCCESS) {                                                                                 \
            std::cerr << "[ERROR] " #expr " failed, ret=" << _ret << " at " __FILE__ ":" << __LINE__ << std::endl; \
            return -1;                                                                                             \
        }                                                                                                          \
    } while (0)

// ============================================================================
// 辅助函数
// ============================================================================

/** 计算 shape 的总元素数（0 维 shape 视为 scalar，返回 1） */
static int64_t ShapeSize(const std::vector<int64_t>& shape)
{
    if (shape.empty())
        return 1;
    int64_t size = 1;
    for (auto d : shape)
        size *= d;
    return size;
}

/** 计算连续 Tensor 的 strides（行优先，单位：元素数） */
static std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape)
{
    if (shape.empty())
        return {};
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

/**
 * 在 Device 侧分配内存，并将 hostData 复制到 Device。
 * 返回 Device 指针，调用方负责 aclrtFree。
 */
static int AllocAndCopy(const void* hostData, size_t byteSize, void** devPtr)
{
    size_t allocSize = (byteSize > 0) ? byteSize : 1;
    CHECK_ACL(aclrtMalloc(devPtr, allocSize, ACL_MEM_MALLOC_HUGE_FIRST));
    if (byteSize > 0) {
        CHECK_ACL(aclrtMemcpy(*devPtr, byteSize, hostData, byteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    return 0;
}

/**
 * 创建 aclTensor。
 * @param devPtr   Device 内存地址（已分配）
 * @param shape    Tensor 的逻辑维度
 * @param dataType aclDataType 枚举
 * @return 非 nullptr 表示成功，调用方负责 aclDestroyTensor
 */
static aclTensor* CreateTensor(void* devPtr, const std::vector<int64_t>& shape, aclDataType dataType)
{
    auto strides = ComputeStrides(shape);
    return aclCreateTensor(
        shape.data(), static_cast<uint64_t>(shape.size()), dataType, strides.data(),
        0, // storageOffset
        aclFormat::ACL_FORMAT_ND, shape.data(), static_cast<uint64_t>(shape.size()), devPtr);
}

// ============================================================================
// CPU Golden：用于结果验证
// ============================================================================

static void ComputeGolden(const float* x, const float* dy, float* dx, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        double xd = static_cast<double>(x[i]);
        double dyd = static_cast<double>(dy[i]);
        dx[i] = static_cast<float>(dyd * (1.0 / (1.0 + xd * xd)));
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char* argv[])
{
    // ------------------------------------------------------------------
    // 0. 测试数据准备（fp32，shape=[4, 8]）
    // ------------------------------------------------------------------
    const std::vector<int64_t> shape = {4, 8};
    const int64_t numElements = ShapeSize(shape);
    const size_t byteSize = static_cast<size_t>(numElements) * sizeof(float);

    // 输入 x（前向输入）：[-2, -1, 0, ..., 1, 2] 线性区间
    std::vector<float> x_host(numElements);
    for (int64_t i = 0; i < numElements; ++i) {
        x_host[i] = -2.0f + 4.0f * static_cast<float>(i) / static_cast<float>(numElements - 1);
    }

    // 输入 dy（上游梯度）：全 1
    std::vector<float> dy_host(numElements, 1.0f);

    // 输出 dx（初始化为 0）
    std::vector<float> dx_host(numElements, 0.0f);

    // ------------------------------------------------------------------
    // 1. ACL 初始化
    // ------------------------------------------------------------------
    std::cout << "[INFO] 初始化 ACL..." << std::endl;
    CHECK_ACL(aclInit(nullptr));

    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    // ------------------------------------------------------------------
    // 2. 在 Device 侧分配内存并上传数据
    // ------------------------------------------------------------------
    std::cout << "[INFO] 分配 Device 内存并上传数据..." << std::endl;

    void* x_dev = nullptr;
    void* dy_dev = nullptr;
    void* dx_dev = nullptr;

    if (AllocAndCopy(x_host.data(), byteSize, &x_dev) != 0)
        goto cleanup;
    if (AllocAndCopy(dy_host.data(), byteSize, &dy_dev) != 0)
        goto cleanup;
    if (AllocAndCopy(dx_host.data(), byteSize, &dx_dev) != 0)
        goto cleanup;

    {
        // ------------------------------------------------------------------
        // 3. 创建 aclTensor
        // ------------------------------------------------------------------
        std::cout << "[INFO] 创建 aclTensor..." << std::endl;

        aclTensor* x_tensor = CreateTensor(x_dev, shape, ACL_FLOAT);
        aclTensor* dy_tensor = CreateTensor(dy_dev, shape, ACL_FLOAT);
        aclTensor* dx_tensor = CreateTensor(dx_dev, shape, ACL_FLOAT);

        if (!x_tensor || !dy_tensor || !dx_tensor) {
            std::cerr << "[ERROR] aclCreateTensor 失败" << std::endl;
            if (x_tensor)
                aclDestroyTensor(x_tensor);
            if (dy_tensor)
                aclDestroyTensor(dy_tensor);
            if (dx_tensor)
                aclDestroyTensor(dx_tensor);
            goto cleanup;
        }

        // ------------------------------------------------------------------
        // 4. 第一段接口：计算 workspaceSize，获取 executor
        // ------------------------------------------------------------------
        std::cout << "[INFO] 调用 aclnnAtanGradGetWorkspaceSize..." << std::endl;

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;

        CHECK_ACLNN(aclnnAtanGradGetWorkspaceSize(x_tensor, dy_tensor, dx_tensor, &workspaceSize, &executor));

        std::cout << "[INFO] workspaceSize = " << workspaceSize << " bytes" << std::endl;

        // ------------------------------------------------------------------
        // 5. 按需分配 workspace（逐元素算子通常为 0）
        // ------------------------------------------------------------------
        void* workspace = nullptr;
        if (workspaceSize > 0) {
            CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }

        // ------------------------------------------------------------------
        // 6. 第二段接口：执行算子
        // ------------------------------------------------------------------
        std::cout << "[INFO] 调用 aclnnAtanGrad..." << std::endl;

        CHECK_ACLNN(aclnnAtanGrad(workspace, workspaceSize, executor, stream));

        // ------------------------------------------------------------------
        // 7. 等待流执行完毕
        // ------------------------------------------------------------------
        CHECK_ACL(aclrtSynchronizeStream(stream));

        // ------------------------------------------------------------------
        // 8. 将结果从 Device 复制回 Host
        // ------------------------------------------------------------------
        std::cout << "[INFO] 复制结果到 Host..." << std::endl;

        CHECK_ACL(aclrtMemcpy(dx_host.data(), byteSize, dx_dev, byteSize, ACL_MEMCPY_DEVICE_TO_HOST));

        // ------------------------------------------------------------------
        // 9. 精度验证（与 CPU Golden 比较）
        // ------------------------------------------------------------------
        std::cout << "[INFO] 精度验证..." << std::endl;

        std::vector<float> golden(numElements);
        ComputeGolden(x_host.data(), dy_host.data(), golden.data(), static_cast<size_t>(numElements));

        double maxRelErr = 0.0;
        for (int64_t i = 0; i < numElements; ++i) {
            double g = static_cast<double>(golden[i]);
            double a = static_cast<double>(dx_host[i]);
            double relErr = std::abs(g - a) / (std::abs(g) + 1e-7);
            if (relErr > maxRelErr)
                maxRelErr = relErr;
        }

        // fp32 精度阈值：MARE < 10 * 2^-13 ≈ 0.00122
        const double fp32_mare_thresh = 10.0 * std::pow(2.0, -13);
        bool passed = (maxRelErr < fp32_mare_thresh);

        // 打印前几个结果
        std::cout << "\n[INFO] 部分输出（前 8 个元素）:" << std::endl;
        std::cout << "  idx |    x     |   dy   |  dx(NPU) |  dx(CPU) |  rel_err" << std::endl;
        std::cout << "  ----|----------|--------|----------|----------|----------" << std::endl;
        for (int i = 0; i < 8 && i < numElements; ++i) {
            double relErr = std::abs((double)golden[i] - (double)dx_host[i]) / (std::abs((double)golden[i]) + 1e-7);
            printf(
                "  %3d | %8.4f | %6.3f | %8.5f | %8.5f | %.2e\n", i, x_host[i], dy_host[i], dx_host[i], golden[i],
                relErr);
        }

        std::cout << "\n[INFO] 最大相对误差 (MARE) = " << maxRelErr << " (阈值=" << fp32_mare_thresh << ")"
                  << std::endl;
        std::cout << "[INFO] 验证结果: " << (passed ? "PASS" : "FAIL") << std::endl;

        // ------------------------------------------------------------------
        // 清理资源
        // ------------------------------------------------------------------
        if (workspace)
            aclrtFree(workspace);
        aclDestroyTensor(x_tensor);
        aclDestroyTensor(dy_tensor);
        aclDestroyTensor(dx_tensor);

        if (!passed) {
            goto cleanup;
        }
    }

cleanup:
    if (x_dev)
        aclrtFree(x_dev);
    if (dy_dev)
        aclrtFree(dy_dev);
    if (dx_dev)
        aclrtFree(dx_dev);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::cout << "[INFO] 示例运行完毕。" << std::endl;
    return 0;
}
