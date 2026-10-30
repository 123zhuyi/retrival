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
 * @file test_aclnn_eltwise.cpp
 * @brief Eltwise 算子 ACLNN 调用示例
 *
 */

#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnn_eltwise.h"

#define CHECK_ACL(expr)                                                                                          \
    do {                                                                                                         \
        auto _ret = (expr);                                                                                      \
        if (_ret != ACL_SUCCESS) {                                                                               \
            std::cerr << "ACL Error: " << #expr << " returned " << _ret << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                                                              \
            goto cleanup;                                                                                        \
        }                                                                                                        \
    } while (0)

int main()
{
    // ========================================================================
    // 1. 参数设置
    //    3 个输入张量，shape: [2, 4] = 8 个元素
    //    mode=1 (SUM): out = 0.5*x0 + 1.5*x1 + 2.0*x2
    // ========================================================================
    constexpr int64_t NUM_INPUTS = 3;
    constexpr int64_t ROWS = 2;
    constexpr int64_t COLS = 4;
    constexpr int64_t ELEM_COUNT = ROWS * COLS;
    const int64_t shape[] = {ROWS, COLS};
    const int64_t strides[] = {COLS, 1};
    constexpr int64_t ndim = 2;
    constexpr int64_t mode = 1; // SUM

    // 加权系数
    float coeffValues[NUM_INPUTS] = {0.5f, 1.5f, 2.0f};

    // 输入数据
    float hostInput0[ELEM_COUNT] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float hostInput1[ELEM_COUNT] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float hostInput2[ELEM_COUNT] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    float* hostInputs[NUM_INPUTS] = {hostInput0, hostInput1, hostInput2};

    // ========================================================================
    // 2. ACL 初始化
    // ========================================================================
    int ret = 1;
    aclrtStream stream = nullptr;
    void* devInput[NUM_INPUTS] = {nullptr};
    void* devOutput = nullptr;
    void* workspace = nullptr;
    aclTensor* inputTensors[NUM_INPUTS] = {nullptr};
    aclTensorList* inputList = nullptr;
    aclTensor* outTensor = nullptr;
    aclFloatArray* coeffArr = nullptr;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    CHECK_ACL(aclrtCreateStream(&stream));

    // ========================================================================
    // 3. 设备内存分配 & 数据拷贝
    // ========================================================================
    {
        size_t dataBytes = ELEM_COUNT * sizeof(float);

        // 为每个输入分配设备内存并拷贝数据
        for (int i = 0; i < NUM_INPUTS; ++i) {
            CHECK_ACL(aclrtMalloc(&devInput[i], dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
            CHECK_ACL(aclrtMemcpy(devInput[i], dataBytes, hostInputs[i], dataBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        // 分配输出内存
        CHECK_ACL(aclrtMalloc(&devOutput, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMemset(devOutput, dataBytes, 0, dataBytes));

        // ====================================================================
        // 4. 创建 aclTensor 和 aclTensorList
        // ====================================================================
        for (int i = 0; i < NUM_INPUTS; ++i) {
            inputTensors[i] =
                aclCreateTensor(shape, ndim, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, ndim, devInput[i]);
        }
        inputList = aclCreateTensorList(inputTensors, NUM_INPUTS);

        outTensor = aclCreateTensor(shape, ndim, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, ndim, devOutput);

        // 创建 coeff 数组
        coeffArr = aclCreateFloatArray(coeffValues, NUM_INPUTS);

        // ====================================================================
        // 5. 调用 aclnnEltwise（两段式接口）
        // ====================================================================
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;

        CHECK_ACL(aclnnEltwiseGetWorkspaceSize(inputList, mode, coeffArr, outTensor, &workspaceSize, &executor));
        if (workspaceSize > 0) {
            CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }

        CHECK_ACL(aclnnEltwise(workspace, workspaceSize, executor, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));

        // ====================================================================
        // 6. 读取输出 & 精度验证
        // ====================================================================
        float hostOutput[ELEM_COUNT] = {};
        CHECK_ACL(aclrtMemcpy(hostOutput, dataBytes, devOutput, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST));

        std::cout << "Eltwise Example (mode=1 SUM, shape: [2,4], dtype: float32)" << std::endl;
        std::cout << "  out[i] = 0.5*x0[i] + 1.5*x1[i] + 2.0*x2[i]" << std::endl;
        std::cout << "---------------------------------------------------------------" << std::endl;
        printf("  %4s | %11s | %11s | %11s | %9s\n", "Idx", "NPU Output", "Expected", "Diff", "Status");
        std::cout << "---------------------------------------------------------------" << std::endl;

        int passCount = 0;
        constexpr float rtol = 1e-4f;
        constexpr float atol = 1e-4f;

        for (int i = 0; i < ELEM_COUNT; ++i) {
            // CPU reference: out = c0*x0 + c1*x1 + c2*x2
            float expected =
                coeffValues[0] * hostInput0[i] + coeffValues[1] * hostInput1[i] + coeffValues[2] * hostInput2[i];
            float diff = std::fabs(hostOutput[i] - expected);
            float threshold = atol + rtol * std::fabs(expected);
            bool pass = (diff <= threshold);
            passCount += pass ? 1 : 0;

            printf(
                "  [%d,%d] | %11.5f | %11.5f | %9.2e | %s\n", (int)(i / COLS), (int)(i % COLS), hostOutput[i], expected,
                diff, pass ? "PASS" : "FAIL");
        }

        std::cout << "---------------------------------------------------------------" << std::endl;
        std::cout << "Result: " << passCount << "/" << ELEM_COUNT << " passed";
        if (passCount == ELEM_COUNT) {
            std::cout << " -- ALL PASS" << std::endl;
            ret = 0;
        } else {
            std::cout << " -- FAILED" << std::endl;
            ret = 1;
        }
    }

    // ========================================================================
    // 7. 资源释放
    // ========================================================================
cleanup:
    if (coeffArr)
        aclDestroyFloatArray(coeffArr);
    if (inputList)
        aclDestroyTensorList(inputList);
    for (int i = 0; i < NUM_INPUTS; ++i) {
        if (inputTensors[i])
            aclDestroyTensor(inputTensors[i]);
    }
    if (outTensor)
        aclDestroyTensor(outTensor);
    if (workspace)
        aclrtFree(workspace);
    for (int i = 0; i < NUM_INPUTS; ++i) {
        if (devInput[i])
            aclrtFree(devInput[i]);
    }
    if (devOutput)
        aclrtFree(devOutput);
    if (stream)
        aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return ret;
}
