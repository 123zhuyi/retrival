/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "random/stateless_random_uniform_v3/op_api/stateless_random_uniform_v3.h"
#include "aclnn/aclnn_base.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include <iostream>

using namespace op;

int main()
{
    int32_t deviceId = 0;
    auto ret = aclrtSetDevice(deviceId);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "aclrtSetDevice failed, ret: " << ret << std::endl;
        return -1;
    }

    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "aclrtCreateStream failed, ret: " << ret << std::endl;
        return -1;
    }

    std::vector<int64_t> shape = {1000};
    auto self = op::CreateTensor(shape, DataType::DT_FLOAT);
    if (self == nullptr) {
        std::cerr << "CreateTensor failed" << std::endl;
        return -1;
    }

    uint64_t seed = 12345;
    uint64_t offset = 0;
    float from = 10.0f;
    float to = 20.0f;

    auto executor = CREATE_EXECUTOR();
    if (executor.get() == nullptr) {
        std::cerr << "CREATE_EXECUTOR failed" << std::endl;
        return -1;
    }

    auto result = l0op::StatelessRandomUniformV3(self, seed, offset, from, to, executor.get());
    if (result == nullptr) {
        std::cerr << "StatelessRandomUniformV3 failed" << std::endl;
        return -1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executorPtr = nullptr;
    auto workspaceRet = executor->GetWorkspaceSize(workspaceSize, &executorPtr);
    if (workspaceRet != ACLNN_SUCCESS) {
        std::cerr << "GetWorkspaceSize failed: " << workspaceRet << std::endl;
        return -1;
    }

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize);
        if (ret != ACL_ERROR_NONE) {
            std::cerr << "aclrtMalloc failed, ret: " << ret << std::endl;
            return -1;
        }
    }

    auto runRet = executor->Run(workspace, workspaceSize, executorPtr, stream);
    if (runRet != ACLNN_SUCCESS) {
        std::cerr << "Run failed: " << runRet << std::endl;
        return -1;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "aclrtSynchronizeStream failed, ret: " << ret << std::endl;
        return -1;
    }

    std::cout << "StatelessRandomUniformV3 test passed!" << std::endl;

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);

    return 0;
}
