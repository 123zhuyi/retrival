/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_mla_prolog_nq_bsh.cpp
 * \brief
 */

#include <iostream>
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mla_prolog.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)      \
  do {                               \
    printf(message, ##__VA_ARGS__);  \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
int CreateAclTensorND(const std::vector<T>& shape, void** deviceAddr, void** hostAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMallocHost申请host侧内存
    ret = aclrtMallocHost(hostAddr, size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);
    memset(*hostAddr, 0, size);
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensorNZ(const std::vector<T>& shape, void** deviceAddr, void** hostAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMallocHost申请host侧内存
    ret = aclrtMallocHost(hostAddr, size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);
    memset(*hostAddr, 0, size);
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                              shape.data(), shape.size(), *deviceAddr);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    return 0;
}

int TransToNZShape(std::vector<int64_t> &shapeND, size_t typeSize) {
    if (typeSize == static_cast<size_t>(0)) {
      return 0;
    }
    int64_t h = shapeND[0];
    int64_t w = shapeND[1];
    int64_t h0 = static_cast<int64_t>(16);
    int64_t w0 = static_cast<int64_t>(32) / static_cast<int64_t>(typeSize);
    int64_t h1 = h / h0;
    int64_t w1 = w / w0;
    shapeND[0] = w1;
    shapeND[1] = h1;
    shapeND.emplace_back(h0);
    shapeND.emplace_back(w0);
    return 0;
}

int main() {
    // 1. 固定写法，device/stream初始化, 参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> tokenXShape = {8, 1, 7168};            // B,S,He
    std::vector<int64_t> weightDqShape = {7168, 1536};          // He,Hcq
    std::vector<int64_t> weightUqQrShape = {1536, 6144};        // Hcq,N*(D+Dr)
    std::vector<int64_t> weightUkShape = {32, 128, 512};        // N,D,Hckv
    std::vector<int64_t> weightDkvKrShape = {7168, 576};        // He,Hckv+Dr
    std::vector<int64_t> rmsnormGammaCqShape = {1536};          // Hcq
    std::vector<int64_t> rmsnormGammaCkvShape = {512};          // Hckv
    std::vector<int64_t> ropeSinShape = {8, 1, 64};             // B,S,Dr
    std::vector<int64_t> ropeCosShape = {8, 1, 64};             // B,S,Dr
    std::vector<int64_t> cacheIndexShape = {8, 1};              // B,S
    std::vector<int64_t> kvCacheShape = {16, 128, 1, 512};      // BolckNum,BlockSize,Nkv,Hckv
    std::vector<int64_t> krCacheShape = {16, 128, 1, 64};       // BolckNum,BlockSize,Nkv,Dr
    std::vector<int64_t> queryShape = {8, 1, 32, 512};          // B,S,N,Hckv
    std::vector<int64_t> queryRopeShape = {8, 1, 32, 64};       // B,S,N,Dr
    double rmsnormEpsilonCq = 1e-5;
    double rmsnormEpsilonCkv = 1e-5;
    char cacheMode[] = "PA_BSND";

    void* tokenXDeviceAddr = nullptr;
    void* weightDqDeviceAddr = nullptr;
    void* weightUqQrDeviceAddr = nullptr;
    void* weightUkDeviceAddr = nullptr;
    void* weightDkvKrDeviceAddr = nullptr;
    void* rmsnormGammaCqDeviceAddr = nullptr;
    void* rmsnormGammaCkvDeviceAddr = nullptr;
    void* ropeSinDeviceAddr = nullptr;
    void* ropeCosDeviceAddr = nullptr;
    void* cacheIndexDeviceAddr = nullptr;
    void* kvCacheDeviceAddr = nullptr;
    void* krCacheDeviceAddr = nullptr;
    void* queryDeviceAddr = nullptr;
    void* queryRopeDeviceAddr = nullptr;

    void* tokenXHostAddr = nullptr;
    void* weightDqHostAddr = nullptr;
    void* weightUqQrHostAddr = nullptr;
    void* weightUkHostAddr = nullptr;
    void* weightDkvKrHostAddr = nullptr;
    void* rmsnormGammaCqHostAddr = nullptr;
    void* rmsnormGammaCkvHostAddr = nullptr;
    void* ropeSinHostAddr = nullptr;
    void* ropeCosHostAddr = nullptr;
    void* cacheIndexHostAddr = nullptr;
    void* kvCacheHostAddr = nullptr;
    void* krCacheHostAddr = nullptr;
    void* queryHostAddr = nullptr;
    void* queryRopeHostAddr = nullptr;

    aclTensor* tokenX = nullptr;
    aclTensor* weightDq = nullptr;
    aclTensor* weightUqQr = nullptr;
    aclTensor* weightUk = nullptr;
    aclTensor* weightDkvKr = nullptr;
    aclTensor* rmsnormGammaCq = nullptr;
    aclTensor* rmsnormGammaCkv = nullptr;
    aclTensor* ropeSin = nullptr;
    aclTensor* ropeCos = nullptr;
    aclTensor* cacheIndex = nullptr;
    aclTensor* kvCache = nullptr;
    aclTensor* krCache = nullptr;
    aclTensor* query = nullptr;
    aclTensor* queryRope = nullptr;

    // 转换三个NZ格式变量的shape
    constexpr size_t EXAMPLE_INT8_SIZE = sizeof(int8_t);
    constexpr size_t EXAMPLE_BFLOAT16_SIZE = sizeof(int16_t);
    ret = TransToNZShape(weightDqShape, EXAMPLE_BFLOAT16_SIZE);
    CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
    ret = TransToNZShape(weightUqQrShape, EXAMPLE_BFLOAT16_SIZE);
    CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
    ret = TransToNZShape(weightDkvKrShape, EXAMPLE_BFLOAT16_SIZE);
    CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);

    // 创建tokenX aclTensor
    ret = CreateAclTensorND(tokenXShape, &tokenXDeviceAddr, &tokenXHostAddr, aclDataType::ACL_BF16, &tokenX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weightDq aclTensor
    ret = CreateAclTensorNZ(weightDqShape, &weightDqDeviceAddr, &weightDqHostAddr, aclDataType::ACL_BF16, &weightDq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weightUqQr aclTensor
    ret = CreateAclTensorNZ(weightUqQrShape, &weightUqQrDeviceAddr, &weightUqQrHostAddr, aclDataType::ACL_BF16, &weightUqQr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weightUk aclTensor
    ret = CreateAclTensorND(weightUkShape, &weightUkDeviceAddr, &weightUkHostAddr, aclDataType::ACL_BF16, &weightUk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weightDkvKr aclTensor
    ret = CreateAclTensorNZ(weightDkvKrShape, &weightDkvKrDeviceAddr, &weightDkvKrHostAddr, aclDataType::ACL_BF16, &weightDkvKr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建ropeSin aclTensor
    ret = CreateAclTensorND(ropeSinShape, &ropeSinDeviceAddr, &ropeSinHostAddr, aclDataType::ACL_BF16, &ropeSin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建ropeCos aclTensor
    ret = CreateAclTensorND(ropeCosShape, &ropeCosDeviceAddr, &ropeCosHostAddr, aclDataType::ACL_BF16, &ropeCos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rmsnormGammaCq aclTensor
    ret = CreateAclTensorND(rmsnormGammaCqShape, &rmsnormGammaCqDeviceAddr, &rmsnormGammaCqHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rmsnormGammaCkv aclTensor
    ret = CreateAclTensorND(rmsnormGammaCkvShape, &rmsnormGammaCkvDeviceAddr, &rmsnormGammaCkvHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建cacheIndex aclTensor
    ret = CreateAclTensorND(cacheIndexShape, &cacheIndexDeviceAddr, &cacheIndexHostAddr, aclDataType::ACL_INT64, &cacheIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建kvCache aclTensor
    ret = CreateAclTensorND(kvCacheShape, &kvCacheDeviceAddr, &kvCacheHostAddr, aclDataType::ACL_BF16, &kvCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建krCache aclTensor
    ret = CreateAclTensorND(krCacheShape, &krCacheDeviceAddr, &krCacheHostAddr, aclDataType::ACL_BF16, &krCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建query aclTensor
    ret = CreateAclTensorND(queryShape, &queryDeviceAddr, &queryHostAddr, aclDataType::ACL_BF16, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建queryRope aclTensor
    ret = CreateAclTensorND(queryRopeShape, &queryRopeDeviceAddr, &queryRopeHostAddr, aclDataType::ACL_BF16, &queryRope);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // 调用aclnnMlaProlog第一段接口
    ret = aclnnMlaPrologGetWorkspaceSize(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, cacheIndex, kvCache, krCache, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, rmsnormEpsilonCq, rmsnormEpsilonCkv, cacheMode, query, queryRope, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnMlaProlog第二段接口
    ret = aclnnMlaProlog(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaProlog failed. ERROR: %d\n", ret); return ret);

    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(queryShape);
    auto copySize = size * aclDataTypeSize(aclDataType::ACL_BF16);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), copySize, queryDeviceAddr, copySize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
    }
    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(tokenX);
    aclDestroyTensor(weightDq);
    aclDestroyTensor(weightUqQr);
    aclDestroyTensor(weightUk);
    aclDestroyTensor(weightDkvKr);
    aclDestroyTensor(rmsnormGammaCq);
    aclDestroyTensor(rmsnormGammaCkv);
    aclDestroyTensor(ropeSin);
    aclDestroyTensor(ropeCos);
    aclDestroyTensor(cacheIndex);
    aclDestroyTensor(kvCache);
    aclDestroyTensor(krCache);
    aclDestroyTensor(query);
    aclDestroyTensor(queryRope);

    // 7. 释放device 资源
    aclrtFree(tokenXDeviceAddr);
    aclrtFree(weightDqDeviceAddr);
    aclrtFree(weightUqQrDeviceAddr);
    aclrtFree(weightUkDeviceAddr);
    aclrtFree(weightDkvKrDeviceAddr);
    aclrtFree(rmsnormGammaCqDeviceAddr);
    aclrtFree(rmsnormGammaCkvDeviceAddr);
    aclrtFree(ropeSinDeviceAddr);
    aclrtFree(ropeCosDeviceAddr);
    aclrtFree(cacheIndexDeviceAddr);
    aclrtFree(kvCacheDeviceAddr);
    aclrtFree(krCacheDeviceAddr);
    aclrtFree(queryDeviceAddr);
    aclrtFree(queryRopeDeviceAddr);

    // 8. 释放host 资源
    aclrtFreeHost(tokenXHostAddr);
    aclrtFreeHost(weightDqHostAddr);
    aclrtFreeHost(weightUqQrHostAddr);
    aclrtFreeHost(weightUkHostAddr);
    aclrtFreeHost(weightDkvKrHostAddr);
    aclrtFreeHost(rmsnormGammaCqHostAddr);
    aclrtFreeHost(rmsnormGammaCkvHostAddr);
    aclrtFreeHost(ropeSinHostAddr);
    aclrtFreeHost(ropeCosHostAddr);
    aclrtFreeHost(cacheIndexHostAddr);
    aclrtFreeHost(kvCacheHostAddr);
    aclrtFreeHost(krCacheHostAddr);
    aclrtFreeHost(queryHostAddr);
    aclrtFreeHost(queryRopeHostAddr);

    if (workspaceSize > static_cast<uint64_t>(0)) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
