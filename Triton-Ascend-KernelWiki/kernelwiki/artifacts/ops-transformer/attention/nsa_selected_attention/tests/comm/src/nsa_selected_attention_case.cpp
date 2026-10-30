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
 * \file nsa_selected_attention_case.cpp
 * \brief NsaSelectedAttention 测试用例.
 */

#include <utility>
#include <vector>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "nsa_selected_attention_case.h"

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */
#define NSA_SELECTED_ATTENTION_KERNEL_PARAM                                                                              \
    (GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR topkIndices, GM_ADDR attenMask, GM_ADDR actualSeqQLen,         \
     GM_ADDR actualSeqKvLen, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR attentionOut, GM_ADDR workspace,          \
     GM_ADDR tiling)

typedef void(*NsaSelectedAttentionKernelFunc) NSA_SELECTED_ATTENTION_KERNEL_PARAM;

extern "C" __global__ __aicore__ void nsa_selected_attention NSA_SELECTED_ATTENTION_KERNEL_PARAM;

using namespace ops::adv::tests::nsa_selected_attention_ns;
using ops::adv::tests::utils::Case;
using ops::adv::tests::utils::Platform;
using ops::adv::tests::utils::Tensor;
using ops::adv::tests::utils::TensorIntf;

bool RunNsaSelectedAttention(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &input,
                           std::vector<TensorIntf *> &output, uint8_t *workspace, uint8_t *tilingData)
{
    auto kernelFunc = (NsaSelectedAttentionKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim,
                input[0]->GetDevData(),  // query
                input[1]->GetDevData(),  // key
                input[2]->GetDevData(),  // value
                input[3]->GetDevData(),  // topkIndices
                input[4]->GetDevData(),  // attenMask
                input[5]->GetDevData(),  // actualSeqQLen
                input[6]->GetDevData(),  // actualSeqKvLen
                output[0]->GetDevData(), // softmaxMax
                output[1]->GetDevData(), // softmaxSum
                output[2]->GetDevData(), // attentionOut
                workspace, tilingData);

    return true;
}

extern "C" ge::graphStatus NsaSelectedAttentionTilingFuncStub(gert::TilingContext *ctx)
{
    auto *nsaCase = static_cast<NsaSelectedAttentionCase *>(Case::GetCurrentCase());
    if (nsaCase != nullptr) {
        NsaSelectedAttentionCase::DoTilingParam tilingParam;
        tilingParam.ctx = ctx;
        tilingParam.ret = ge::GRAPH_SUCCESS;

        if (!nsaCase->DoOpTiling(tilingParam)) {
            return tilingParam.ret;
        }
        return nsaCase->tilingKernelFunc(ctx);
    }
    return ge::GRAPH_FAILED;
}

bool NsaSelectedAttentionCase::InitParam()
{
    if (!mParam.Init()) {
        LOG_ERR("Init Param Failed.");
        return false;
    }
    return true;
}

bool NsaSelectedAttentionCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("NsaSelectedAttention");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&mParam.query, &mParam.key, &mParam.value, &mParam.topkIndices, &mParam.attenMask,
                                 &mParam.actualSeqQLen, &mParam.actualSeqKvLen});
    rst = rst && mCtx.SetOutputs({&mParam.softmaxMax, &mParam.softmaxSum, &mParam.attentionOut});
    rst = rst && mCtx.SetAttrs({{"scale_value", mParam.scale},
                                {"head_num", mParam.n2},
                                {"sparse_mode", mParam.sparseMode},
                                {"input_layout", mParam.layout},
                                {"selected_block_size", mParam.selectedBlockSize},
                                {"selected_block_count", mParam.selectedBlockCount}});
    rst = rst && mCtx.SetKernelRunCbf(RunNsaSelectedAttention);
    rst = rst && mCtx.SetKernelMainFunc((void *)nsa_selected_attention);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("GlobalPlatform is null.");
        return false;
    }

    tilingKernelFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingNsaSelectedAttention");
    if (tilingKernelFunc == nullptr) {
        LOG_ERR("Can not get origin tiling func, NsaSelectedAttention(%p)", tilingKernelFunc);
        return false;
    }
    IMPL_OP(NsaSelectedAttention).Tiling(NsaSelectedAttentionTilingFuncStub);
    return rst;
}

bool NsaSelectedAttentionCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool NsaSelectedAttentionCase::Run()
{
    if (!mEnable) {
        return true;
    }
    if (!mOpInfo.ProcessTiling(mName)) {
        return false;
    }
    if (!mOpInfo.ProcessKernel(mName)) {
        return false;
    }
    return true;
}

bool NsaSelectedAttentionCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}

NsaSelectedAttentionCase::NsaSelectedAttentionCase()
{
    this->mName = "NsaSelectedAttentionCase";
    this->mOpInfo.mName = "NsaSelectedAttention";
}

NsaSelectedAttentionCase::NsaSelectedAttentionCase(const char *name, bool enable, const char *dbgInfo, OpInfo opInfo,
                                               Param param)
    : Case(name, enable, dbgInfo), mOpInfo(opInfo), mParam(param)
{
}
