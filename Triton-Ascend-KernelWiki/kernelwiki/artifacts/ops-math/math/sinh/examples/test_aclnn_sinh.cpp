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
#include "aclnnop/aclnn_sinh.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int ExecuteSinh(aclTensor* self, aclTensor* out, aclrtStream stream) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    int ret = aclnnSinhGetWorkspaceSize(self, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) return ret;

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) return ret;
    }

    ret = aclnnSinh(workspaceAddr, workspaceSize, executor, stream);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    return ret;
}

int ExecuteInplaceSinh(aclTensor* self, aclrtStream stream) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    int ret = aclnnInplaceSinhGetWorkspaceSize(self, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) return ret;

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) return ret;
    }

    ret = aclnnInplaceSinh(workspaceAddr, workspaceSize, executor, stream);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    return ret;
}

int PrintResults(void* outDeviceAddr, std::vector<int64_t>& outShape,
                    void* selfDeviceAddr, std::vector<int64_t>& selfShape) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    int ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) return ret;

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    auto inplaceSize = GetShapeSize(selfShape);
    std::vector<float> inplaceResultData(inplaceSize, 0);
    ret = aclrtMemcpy(inplaceResultData.data(), inplaceResultData.size() * sizeof(inplaceResultData[0]),
                      selfDeviceAddr, inplaceSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) return ret;

    for (int64_t i = 0; i < inplaceSize; i++) {
        LOG_PRINT("inplaceResult[%ld] is: %f\n", i, inplaceResultData[i]);
    }

    return ACL_SUCCESS;
}

int CreateInputOutputTensors(aclTensor** self, aclTensor** out,
                             void** selfDeviceAddr, void** outDeviceAddr) {
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> outShape = {4, 2};
    std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};

    // 创建self aclTensor
    int ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_FLOAT, self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, outDeviceAddr, aclDataType::ACL_FLOAT, out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    return ACL_SUCCESS;
}

void CleanupResources(aclTensor* self, aclTensor* out,
                      void* selfDeviceAddr, void* outDeviceAddr,
                      aclrtStream stream, int32_t deviceId) {
    // 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(self);
    aclDestroyTensor(out);

    // 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main() {
    // 1. Device/stream initialization
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct input and output tensors
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;

    ret = CreateInputOutputTensors(&self, &out, &selfDeviceAddr, &outDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create tensors failed. ERROR: %d\n", ret);
              CleanupResources(self, out, selfDeviceAddr, outDeviceAddr, stream, deviceId); return ret);

    // 3. Execute sinh operations
    ret = ExecuteSinh(self, out, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExecuteSinh failed. ERROR: %d\n", ret);
              CleanupResources(self, out, selfDeviceAddr, outDeviceAddr, stream, deviceId); return ret);

    ret = ExecuteInplaceSinh(self, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExecuteInplaceSinh failed. ERROR: %d\n", ret);
              CleanupResources(self, out, selfDeviceAddr, outDeviceAddr, stream, deviceId); return ret);

    // 4. Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              CleanupResources(self, out, selfDeviceAddr, outDeviceAddr, stream, deviceId); return ret);

    // 5. Print results
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> outShape = {4, 2};
    ret = PrintResults(outDeviceAddr, outShape, selfDeviceAddr, selfShape);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("PrintResults failed. ERROR: %d\n", ret);
              CleanupResources(self, out, selfDeviceAddr, outDeviceAddr, stream, deviceId); return ret);

    // 6. Cleanup resources
    CleanupResources(self, out, selfDeviceAddr, outDeviceAddr, stream, deviceId);

    return 0;
}
