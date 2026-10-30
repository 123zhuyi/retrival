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
#include "aclnnop/aclnn_hans_encode.h"
#include "aclnnop/aclnn_hans_decode.h"

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
  // 固定写法，初始化
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

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API文档
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<float> inputHost(65536, 0);
  std::vector<float> mantissaHost(49152, 0);
  std::vector<float> fixedHost(16384, 0);
  std::vector<float> varHost(16384, 0);
  std::vector<int32_t> pdfHost(256, 0);
  std::vector<float> recoverHost(65536, 0);
  bool statistic = true;
  bool reshuff = false;
  int64_t outHostAddr = -1;
  int64_t outHostLength = 0;

  void* inputAddr = nullptr;
  void* outMantissaAddr = nullptr;
  void* outFixedAddr = nullptr;
  void* outVarAddr = nullptr;
  void* pdfAddr = nullptr;
  void* recoverAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* outMantissa = nullptr;
  aclTensor* outFixed = nullptr;
  aclTensor* pdf = nullptr;
  aclTensor* outVar = nullptr;
  aclTensor* recover = nullptr;
  // 创建out aclTensor
  ret = CreateAclTensor(inputHost, {1, 65536}, &inputAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(mantissaHost, {1, 49152}, &outMantissaAddr, aclDataType::ACL_FLOAT, &outMantissa);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(fixedHost, {1, 16384}, &outFixedAddr, aclDataType::ACL_FLOAT, &outFixed);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(varHost, {1, 16384}, &outVarAddr, aclDataType::ACL_FLOAT, &outVar);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(pdfHost, {1, 256}, &pdfAddr, aclDataType::ACL_INT32, &pdf);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(recoverHost, {1, 65536}, &recoverAddr, aclDataType::ACL_FLOAT, &recover);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnHansEncode第一段接口
  ret = aclnnHansEncodeGetWorkspaceSize(input, pdf, statistic, reshuff, outMantissa, outFixed, outVar, &workspaceSize,
                                        &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHansEncodeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > static_cast<uint64_t>(0)) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnHansEncode第二段接口
  ret = aclnnHansEncode(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHansEncode failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = 16384 * sizeof(float);
  std::vector<float> resultData(16384, 0);
  ret = aclrtMemcpy(resultData.data(), size, outFixedAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < 128; i++) {
    int32_t intVal = *reinterpret_cast<int32_t*>(&resultData[i]);
    LOG_PRINT("result header[%ld] is: %d\n", i, intVal);
  }

  uint64_t workspaceSizeDecode = 0;
  aclOpExecutor* executorDecode;
  // 调用aclnnHansDecode第一段接口
  ret = aclnnHansDecodeGetWorkspaceSize(outMantissa, outFixed, outVar, pdf, reshuff, recover, &workspaceSizeDecode,
                                        &executorDecode);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHansDecodeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddrDecode = nullptr;
  if (workspaceSizeDecode > 0) {
    ret = aclrtMalloc(&workspaceAddrDecode, workspaceSizeDecode, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnHansEncode第二段接口
  ret = aclnnHansDecode(workspaceAddrDecode, workspaceSizeDecode, executorDecode, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHansDecode failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  std::vector<float> recoverData(65536, 0);
  ret = aclrtMemcpy(recoverData.data(), 65536 * sizeof(recoverData[0]), recoverAddr, 65536 * sizeof(recoverData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < 256; i++) {
    LOG_PRINT("reco[%ld] is: %f org is: %f\n", i, recoverData[i], inputHost[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(input);
  aclDestroyTensor(outMantissa);
  aclDestroyTensor(outFixed);
  aclDestroyTensor(outVar);
  aclDestroyTensor(pdf);
  aclDestroyTensor(recover);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(inputAddr);
  aclrtFree(outMantissaAddr);
  aclrtFree(outFixedAddr);
  aclrtFree(outVarAddr);
  aclrtFree(pdfAddr);
  aclrtFree(recoverAddr);
  if (workspaceSize > static_cast<uint64_t>(0)) {
    aclrtFree(workspaceAddr);
  }
  if (workspaceSizeDecode > 0) {
    aclrtFree(workspaceAddrDecode);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}