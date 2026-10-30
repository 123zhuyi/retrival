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
#include "aclnnop/aclnn_svd.h"

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

struct SVDTensors {
  void* inputDeviceAddr = nullptr;
  void* uDeviceAddr = nullptr;
  void* sigmaDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* u = nullptr;
  aclTensor* sigma = nullptr;
  aclTensor* v = nullptr;
  std::vector<int64_t> uShape = {2, 2};
  std::vector<int64_t> sigmaShape = {2};
  std::vector<int64_t> vShape = {3, 3};
};

struct SVDWorkspace {
  void* addr = nullptr;
  uint64_t size = 0;
};

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

int SetupAndExecuteSVD(aclrtStream stream, SVDTensors& tensors, SVDWorkspace& workspace) {
  auto ret = 0;

  //构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> inputShape = {2, 3};
  std::vector<float> inputHostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> uHostData = {0, 0, 0, 0};
  std::vector<float> sigmaHostData = {0, 0};
  std::vector<float> vHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  bool computeUV = true;
  bool fullMatrices = true;

  // 创建input aclTensor
  ret = CreateAclTensor(inputHostData, inputShape, &tensors.inputDeviceAddr, aclDataType::ACL_FLOAT, &tensors.input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建u aclTensor
  ret = CreateAclTensor(uHostData, tensors.uShape, &tensors.uDeviceAddr, aclDataType::ACL_FLOAT, &tensors.u);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建sigma aclTensor
  ret = CreateAclTensor(sigmaHostData, tensors.sigmaShape, &tensors.sigmaDeviceAddr, aclDataType::ACL_FLOAT, &tensors.sigma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建v aclTensor
  ret = CreateAclTensor(vHostData, tensors.vShape, &tensors.vDeviceAddr, aclDataType::ACL_FLOAT, &tensors.v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // 调用CANN算子库API，需要修改为具体的Api名称
  aclOpExecutor* executor;
  // 调用aclnnSvdGetWorkspaceSize第一段接口
  ret = aclnnSvdGetWorkspaceSize(tensors.input, computeUV, fullMatrices, tensors.sigma, tensors.u, tensors.v, 
                                 &workspace.size, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSvdGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  
  // 根据第一段接口计算出的workspaceSize申请device内存
  workspace.addr = nullptr;
  if (workspace.size > 0) {
    ret = aclrtMalloc(&workspace.addr, workspace.size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  
  // 调用aclnnSvd第二段接口
  ret = aclnnSvd(workspace.addr, workspace.size, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRightShift failed. ERROR: %d\n", ret); return ret);

  // 同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  return 0;
}

int ProcessAndCleanupSVD(SVDTensors& tensors, SVDWorkspace& workspace) {
  auto ret = 0;
  
  // 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto uSize = GetShapeSize(tensors.uShape);
  std::vector<float> uData(uSize, 0);
  ret = aclrtMemcpy(uData.data(), uData.size() * sizeof(uData[0]), tensors.uDeviceAddr,
                    uSize * sizeof(uData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outTensor U from device to host failed. ERROR: %d\n", ret); return ret);
  
  for (int64_t i = 0; i < uSize; i++) {
    LOG_PRINT("u[%ld] is: %f\n", i, uData[i]);
  }

  auto sigmaSize = GetShapeSize(tensors.sigmaShape);
  std::vector<float> sigmaData(sigmaSize, 0);
  ret = aclrtMemcpy(sigmaData.data(), sigmaData.size() * sizeof(sigmaData[0]), tensors.sigmaDeviceAddr,
                    sigmaSize * sizeof(sigmaData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outTensor sigma from device to host failed. ERROR: %d\n", ret); return ret);
  
  for (int64_t i = 0; i < sigmaSize; i++) {
    LOG_PRINT("sigma[%ld] is: %f\n", i, sigmaData[i]);
  }
  
  auto vSize = GetShapeSize(tensors.vShape);
  std::vector<float> vData(vSize, 0);
  ret = aclrtMemcpy(vData.data(), vData.size() * sizeof(vData[0]), tensors.vDeviceAddr,
                    vSize * sizeof(vData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outTensor V from device to host failed. ERROR: %d\n", ret); return ret);
  
  for (int64_t i = 0; i < vSize; i++) {
    LOG_PRINT("v[%ld] is: %f\n", i, vData[i]);
  }  
  
  // 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(tensors.input);
  aclDestroyTensor(tensors.u);
  aclDestroyTensor(tensors.sigma);
  aclDestroyTensor(tensors.v);

  // 释放device 资源
  aclrtFree(tensors.inputDeviceAddr);
  aclrtFree(tensors.uDeviceAddr);
  aclrtFree(tensors.sigmaDeviceAddr);
  aclrtFree(tensors.vDeviceAddr);
  if (workspace.size > 0) {
    aclrtFree(workspace.addr);
  }
  
  return 0;
}

int ExecuteSVDOperator(aclrtStream stream) {
  SVDTensors tensors;
  SVDWorkspace workspace;
  
  auto ret = SetupAndExecuteSVD(stream, tensors, workspace);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  ret = ProcessAndCleanupSVD(tensors, workspace);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  return 0;
}


int main() {
  // device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 执行GtScalar操作
  ret = ExecuteSVDOperator(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExecuteGtScalarOperator failed. ERROR: %d\n", ret); return ret);

  // 重置设备和终结ACL
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}