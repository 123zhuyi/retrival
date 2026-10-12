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
 * \file update_context.cpp
 * \brief
 */

#include <torch/extension.h>
#include "hccl_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;

constexpr static uint8_t COMM_ENGINE_AIV = 4;
constexpr uint32_t OP_TYPE_ALL_TO_ALLV = 8;
constexpr uint32_t EP_OP_TYPE = OP_TYPE_ALL_TO_ALLV;
const std::string EP_ALG_CONFIG = "AlltoAll=level0:fullmesh;level1:pairwise";

struct CommContext {
    uint64_t epRankId = 0;
    uint64_t kfcContextAddr = 0;
    uint64_t epHcclBuffer_[1024];
};

static int32_t CreatMc2Context(HcclComm &comm, int64_t worldSize, int64_t &cclBufferSize, CommContext &mc2Context)
{
    uint32_t ctxIndex = 0;
    uint32_t rankId;
    (void)HcclGetRankIdFunc(comm, &rankId);
    mc2Context.epRankId = rankId;
    for (uint64_t remoteRankId = 0; remoteRankId < worldSize; remoteRankId++) {
        void *remoteAddr = nullptr;
        uint64_t commSize = 0;
        HcclResult ret;
        if (rankId == remoteRankId) {
            ret = static_cast<HcclResult>(HcclGetHcclBufferFunc(comm, &remoteAddr, &commSize));
            cclBufferSize = commSize;
        } else {
            ret = static_cast<HcclResult>(HcclGetRemoteIpcHcclBufFunc(comm, remoteRankId, &remoteAddr, &commSize));
        }
        TORCH_CHECK((ret == HCCL_SUCCESS), "Get HcclBufferSize failed, ret=", ret);
        mc2Context.epHcclBuffer_[remoteRankId] = (uint64_t)remoteAddr;
    }

    return 0;
}

static int32_t CreateHcclContext(HcclComm &commHandle, void *opArgs, int64_t worldSize, const char* groupName,
    std::string algConfig, uint32_t opType)
{
    HcclResult ret = static_cast<HcclResult>(HcclKfcOpArgsSetAlgConfigFunc(opArgs, const_cast<char *>(algConfig.c_str())));  // 设置通信类型
    TORCH_CHECK(ret == 0, "HcclKfcOpArgsSetAlgConfig failed, ret:", ret);
    ret = static_cast<HcclResult>(HcclCommGetHandleWithNameFunc(groupName, &commHandle));  // 通过groupName获取groupHandle
    TORCH_CHECK(ret == 0, "HcclGetCommHandle failed, ret:", ret);
    void *opsResCtx;
    ret = static_cast<HcclResult>(HcclCreateOpResCtxFunc(commHandle, opType, opArgs, &opsResCtx)); // 创建HcclContext
    TORCH_CHECK(ret == 0, "HcclCreateOpResCtx failed, ret:", ret);

    // Get Comm world Size and rank
    uint32_t rankId = 0;
    uint32_t worldSizeHccl = 0;
    ret = static_cast<HcclResult>(HcclGetRankSizeFunc(commHandle, &worldSizeHccl)); // 获取通信域大小
    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankSize failed, ret:", ret);
    ret = static_cast<HcclResult>(HcclGetRankIdFunc(commHandle, &rankId)); // 获取卡号
    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankId failed, ret:", ret);
    if ((rankId >= worldSizeHccl) || (worldSize != worldSizeHccl)) {
        TORCH_CHECK(rankId < worldSizeHccl, "rankId:", rankId, " worldSizeHccl:", worldSizeHccl, " worldSize:", worldSize);
        return -1;
    }
    return 0;
}

static int32_t GetMc2Context(CommContext &mc2ContextHost, int64_t epWorldSize, int64_t &cclBufferSize, const char* groupEpStr)
{
    InitHcclFunctions();
    void* opArgs = nullptr;
    HcclResult ret = static_cast<HcclResult>(HcclKfcAllocOpArgsFunc(&opArgs));  // 通信配置对象创建
    TORCH_CHECK(ret == 0, "HcclKfcAllocOpArgs failed, ret:", ret);
    uint8_t commEngine = COMM_ENGINE_AIV;
    ret = static_cast<HcclResult>(HcclKfcOpArgsSetCommEngineFunc(opArgs, (uint8_t)commEngine)); // 设置通信方式
    TORCH_CHECK(ret == 0, "HcclKfcOpArgsSetCommEngine failed, ret:", ret);
    HcclComm epCommHandle;
    int32_t contextRet = CreateHcclContext(epCommHandle, opArgs, epWorldSize, groupEpStr, EP_ALG_CONFIG, EP_OP_TYPE);
    TORCH_CHECK(contextRet == 0, "CreateHcclContext failed, ret:", contextRet);
    ret = static_cast<HcclResult>(HcclKfcFreeOpArgsFunc(opArgs)); // 释放通信配置对象
    TORCH_CHECK(ret == 0, "getHcclKfcFreeOpArgs failed, ret:", ret);
    contextRet = CreatMc2Context(epCommHandle, epWorldSize, cclBufferSize, mc2ContextHost);
    TORCH_CHECK(contextRet == 0, "CreatMc2Context failed, ret:", contextRet);

    return 0;
}

/**
 * @brief ACLNN Warpper for aclnnAbs
 * @param x Input Tensor (on NPU)
 * @return Result Tensor
**/
bool UpdateContext(std::string groupEp, int64_t epWorldSize, py::object cclBufferSize, at::Tensor &contextTensor)
{
    CommContext mc2ContextHost;
    int64_t bufferSize = 0;
    int32_t ret = GetMc2Context(mc2ContextHost, epWorldSize, bufferSize, groupEp.c_str());
    TORCH_CHECK(ret == 0, "GetMc2Context failed, ret:", ret);
    cclBufferSize.attr("value") = bufferSize;
    at::Tensor hostContext = at::from_blob(&mc2ContextHost, {sizeof(CommContext) / sizeof(int32_t)}, at::kInt);
    contextTensor.copy_(hostContext);
    return true;
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("update_context", &UpdateContext, "update_context",
          py::arg("groupEp"), py::arg("epWorldSize"),
          py::arg("cclBufferSize"), py::arg("contextTensor").noconvert());
}
} // op_api