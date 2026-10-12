/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_min_dim.h"

static std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

template <typename T>
static bool RunCase(aclrtStream stream, const char* name, aclDataType dtype, const std::vector<int64_t>& shape,
                    const std::vector<int64_t>& storageShape, const std::vector<int64_t>& strides,
                    const std::vector<T>& input, int64_t dim, bool keepDims, const std::vector<int64_t>& outShape,
                    const std::vector<T>& expectedValues, const std::vector<int32_t>& expectedIndices)
{
    void* xDev = nullptr;
    void* valueDev = nullptr;
    void* indexDev = nullptr;
    void* workspace = nullptr;
    aclTensor* x = nullptr;
    aclTensor* values = nullptr;
    aclTensor* indices = nullptr;

    auto cleanup = [&]() {
        if (x != nullptr)
            aclDestroyTensor(x);
        if (values != nullptr)
            aclDestroyTensor(values);
        if (indices != nullptr)
            aclDestroyTensor(indices);
        if (workspace != nullptr)
            aclrtFree(workspace);
        if (xDev != nullptr)
            aclrtFree(xDev);
        if (valueDev != nullptr)
            aclrtFree(valueDev);
        if (indexDev != nullptr)
            aclrtFree(indexDev);
    };
    auto fail = [&](const char* step, aclError ret) {
        std::printf("[FAIL] %s: %s returned %d\n", name, step, static_cast<int>(ret));
        cleanup();
        return false;
    };

    aclError ret = aclrtMalloc(&xDev, input.size() * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
        return fail("aclrtMalloc(x)", ret);
    ret = aclrtMalloc(&valueDev, expectedValues.size() * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
        return fail("aclrtMalloc(values)", ret);
    ret = aclrtMalloc(&indexDev, expectedIndices.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
        return fail("aclrtMalloc(indices)", ret);
    ret = aclrtMemcpy(xDev, input.size() * sizeof(T), input.data(), input.size() * sizeof(T),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS)
        return fail("aclrtMemcpy(input)", ret);

    auto outStrides = ContiguousStrides(outShape);
    x = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, storageShape.data(),
                        storageShape.size(), xDev);
    values = aclCreateTensor(outShape.data(), outShape.size(), dtype, outStrides.data(), 0, ACL_FORMAT_ND,
                             outShape.data(), outShape.size(), valueDev);
    indices = aclCreateTensor(outShape.data(), outShape.size(), ACL_INT32, outStrides.data(), 0, ACL_FORMAT_ND,
                              outShape.data(), outShape.size(), indexDev);
    if (x == nullptr || values == nullptr || indices == nullptr) {
        std::printf("[FAIL] %s: aclCreateTensor returned null\n", name);
        cleanup();
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnMinDimGetWorkspaceSize(x, dim, keepDims, values, indices, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS)
        return fail("aclnnMinDimGetWorkspaceSize", ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS)
            return fail("aclrtMalloc(workspace)", ret);
    }
    ret = aclnnMinDim(workspace, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS)
        return fail("aclnnMinDim", ret);
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS)
        return fail("aclrtSynchronizeStream", ret);

    std::vector<T> actualValues(expectedValues.size());
    std::vector<int32_t> actualIndices(expectedIndices.size());
    ret = aclrtMemcpy(actualValues.data(), actualValues.size() * sizeof(T), valueDev, actualValues.size() * sizeof(T),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS)
        return fail("aclrtMemcpy(values)", ret);
    ret = aclrtMemcpy(actualIndices.data(), actualIndices.size() * sizeof(int32_t), indexDev,
                      actualIndices.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS)
        return fail("aclrtMemcpy(indices)", ret);

    const bool valuesOk = std::memcmp(actualValues.data(), expectedValues.data(), expectedValues.size() * sizeof(T)) ==
                          0;
    const bool indicesOk = actualIndices == expectedIndices;
    cleanup();
    std::printf("[%s] %s\n", valuesOk && indicesOk ? "PASS" : "FAIL", name);
    return valuesOk && indicesOk;
}

int main()
{
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS)
        return ret;
    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS)
        return ret;
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS)
        return ret;

    bool ok = true;
    ok &= RunCase<float>(stream, "fp32 ties", ACL_FLOAT, {3, 4}, {3, 4}, {4, 1}, {5, 2, 8, 2, 1, 9, 3, 7, 4, 4, 0, 6},
                         1, false, {3}, {2, 1, 0}, {1, 0, 2});
    ok &= RunCase<uint16_t>(stream, "fp16 negative dim keepdim", ACL_FLOAT16, {2, 3}, {2, 3}, {3, 1},
                            {0x4200, 0x3c00, 0x3c00, 0x4000, 0x4400, 0x0000}, -1, true, {2, 1}, {0x3c00, 0x0000},
                            {1, 2});
    ok &= RunCase<uint16_t>(stream, "bf16 first dimension", ACL_BF16, {2, 3}, {2, 3}, {3, 1},
                            {0x40a0, 0x4000, 0x40e0, 0x3f80, 0x4080, 0x4040}, 0, false, {3}, {0x3f80, 0x4000, 0x4040},
                            {1, 0, 1});
    ok &= RunCase<int16_t>(stream, "int16", ACL_INT16, {2, 4}, {2, 4}, {4, 1}, {7, -2, -2, 5, -8, 3, 9, 9}, 1, false,
                           {2}, {-2, -8}, {1, 0});
    ok &= RunCase<float>(stream, "non-contiguous fp32 input", ACL_FLOAT, {2, 3}, {2, 4}, {4, 1},
                         {6, 1, 4, 99, 2, 8, 2, 99}, 1, false, {2}, {1, 2}, {1, 0});

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    std::printf("ArgMinWithValue examples: %s\n", ok ? "ALL PASS" : "FAILED");
    return ok ? 0 : 1;
}
