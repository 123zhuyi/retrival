/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_strided_slice_grad.h"

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
    // 固定写法，AscendCL 初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 申请 device 侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 将 host 数据拷贝到 device
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // 计算连续 tensor 的 strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // 创建 aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1.（固定写法）device/stream 初始化，参考 AscendCL 对外接口列表
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    //
    // 场景说明：
    //   正向 StridedSlice 从 shape=[3,4,5] 的张量中切出 [0:2, 1:3, 2:4]（步长全1），
    //   正向输出 shape = [2, 2, 2]。
    //   本反向算子将 dy（shape=[2,2,2]）中的梯度散射回 output（shape=[3,4,5]），
    //   未被切片覆盖的位置填 0。
    //
    // 参数：
    //   shape   = [3, 4, 5]   — 原始正向输入的 shape，即 output 的 shape
    //   begin   = [0, 1, 2]
    //   end     = [2, 3, 4]
    //   strides = [1, 1, 1]
    //   mask 全 0

    std::vector<int64_t> shapeData = {3, 4, 5};
    std::vector<int64_t> beginData = {0, 1, 2};
    std::vector<int64_t> endData = {2, 3, 4};
    std::vector<int64_t> stridesData = {1, 1, 1};

    std::vector<int64_t> dyShape = {2, 2, 2};
    std::vector<int64_t> outputShape = {3, 4, 5};

    // dy 梯度数据（float，全 1.0）
    std::vector<float> dyHostData(GetShapeSize(dyShape), 1.0f);
    // output 初始化为 0
    std::vector<float> outputHostData(GetShapeSize(outputShape), 0.0f);

    void* dyDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* output = nullptr;

    // 创建 dy aclTensor
    ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 output aclTensor
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建值依赖输入（aclIntArray）
    aclIntArray* shape = aclCreateIntArray(shapeData.data(), shapeData.size());
    aclIntArray* begin = aclCreateIntArray(beginData.data(), beginData.size());
    aclIntArray* end = aclCreateIntArray(endData.data(), endData.size());
    aclIntArray* strides = aclCreateIntArray(stridesData.data(), stridesData.size());
    CHECK_RET(shape != nullptr && begin != nullptr && end != nullptr && strides != nullptr,
              LOG_PRINT("aclCreateIntArray failed.\n");
              return -1);

    // mask 参数（全 0 表示不启用任何特殊切片语义）
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;

    // 3. 调用 CANN 算子库 API（两段式接口）
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 调用 aclnnStridedSliceGrad 第一段接口
    ret = aclnnStridedSliceGradGetWorkspaceSize(shape, begin, end, strides, dy, beginMask, endMask, ellipsisMask,
                                                newAxisMask, shrinkAxisMask, output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStridedSliceGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的 workspaceSize 申请 device 内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用 aclnnStridedSliceGrad 第二段接口
    ret = aclnnStridedSliceGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStridedSliceGrad failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 将结果从 device 侧拷贝回 host 侧并打印
    auto size = GetShapeSize(outputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outputDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放 aclIntArray 和 aclTensor
    aclDestroyIntArray(shape);
    aclDestroyIntArray(begin);
    aclDestroyIntArray(end);
    aclDestroyIntArray(strides);
    aclDestroyTensor(dy);
    aclDestroyTensor(output);

    // 7. 释放 Device 资源
    aclrtFree(dyDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
