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
 * \file lightning_indexer_grad_case.cpp
 * \brief LightningIndexerGrad 测试用例.
 */
#include "lightning_indexer_grad_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace Ops::Transformer::OpTiling;
using LightningIndexerGradKernelFunc = void(*) LIGHTNING_INDEXER_GRAD_PARAM;
using namespace ops::adv::tests::LIG;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Platform = ops::adv::tests::utils::Platform;
using LightningIndexerGradCase = ops::adv::tests::LIG::LightningIndexerGradCase;
using CaseWithSocversion = ops::adv::tests::utils::CaseWithSocversion;

namespace optiling {
ASCENDC_EXTERN_C ge::graphStatus LightningIndexerGrad(gert::TilingContext *context);
} // namespace optiling

namespace {
const size_t LIG_ACTUAL_SEQ_LENGTH_Q_INPUT_INDEX = 5UL;
const size_t LIG_ACTUAL_SEQ_LENGTH_K_INPUT_INDEX = 6UL;

ASCENDC_EXTERN_C ge::graphStatus TilingForLightningIndexerGradStub(gert::TilingContext *context)
{
    auto *lightningIndexerGradCase = static_cast<LightningIndexerGradCase *>(CaseWithSocversion::GetCurrentCase());
    if (lightningIndexerGradCase != nullptr) {
        LightningIndexerGradCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        p.actSeqLenQTensor =
            const_cast<gert::Tensor *>(context->GetOptionalInputTensor(LIG_ACTUAL_SEQ_LENGTH_Q_INPUT_INDEX));
        p.actSeqLenKTensor =
            const_cast<gert::Tensor *>(context->GetOptionalInputTensor(LIG_ACTUAL_SEQ_LENGTH_K_INPUT_INDEX));
        if (!lightningIndexerGradCase->DoOpTiling(p)) {
            return p.ret;
        }
        return lightningIndexerGradCase->mLightningIndexerGradOriginTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool RunLightningIndexerGrad(std::function<void(LIG_INPUT_DTYPE)> func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                    std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    (void)blockDim;
    // Kernel 运行
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(func, 1,
                inputs[0]->GetDevData(),
                inputs[1]->GetDevData(),
                inputs[2]->GetDevData(),
                inputs[3]->GetDevData(),
                inputs[4]->GetDevData(),
                inputs[5]->GetDevData(),
                inputs[6]->GetDevData(),
                outputs[0]->GetDevData(),
                outputs[1]->GetDevData(),
                outputs[2]->GetDevData(),
                workspace, tilingData);
    return true;
}
} // namespace

LightningIndexerGradCase::LightningIndexerGradCase(const std::function<void(LIG_INPUT_DTYPE)>& templatekeyKernelFunc)
: LightningIndexerGradCase("Undefined", true, "", OpInfoWithSocversion(), LightningIndexerGradParam(), 0)
{
    mLIGKernelFunc = templatekeyKernelFunc;
}

LightningIndexerGradCase::LightningIndexerGradCase() : LightningIndexerGradCase("Undefined", true, "", OpInfoWithSocversion(), LightningIndexerGradParam(), 0)
{
}

LightningIndexerGradCase::LightningIndexerGradCase(const char *name, bool enable, const char *dbgInfo, OpInfoWithSocversion opinfo,
    LightningIndexerGradParam param, int32_t tilingTemplatePriority)
    : CaseWithSocversion(name, enable, dbgInfo, tilingTemplatePriority), mOpInfo(std::move(opinfo)), mParam(std::move(param))
{
    mOpInfo.mName = "LightningIndexerGrad";
    mLightningIndexerGradOriginTilingFuncName = "TilingForLightningIndexerGrad";
}

bool LightningIndexerGradCase::Run()
{
    if (!mEnable) {
        return true;
    }
    if (!mOpInfo.ProcessTiling(mName, this->socVersion)) {
        return false;
    }
    if (!mOpInfo.ProcessKernel(mName)) {
        return false;
    }
    return true;
}

bool LightningIndexerGradCase::InitParam()
{
    if (!mParam.Init()) {
        return false;
    }
    return true;
}

bool LightningIndexerGradCase::InitOpInfo()
{
    bool rst = this->InitOpInfoCtx();
    if (!rst) {
        return rst;
    }

    if (!this->InitOriginTilingFunc()) {
        return false;
    }
    IMPL_OP(LightningIndexerGrad).Tiling(TilingForLightningIndexerGradStub);

    return true;
}

bool LightningIndexerGradCase::InitOpInfoCtx()
{
    bool rst = mCtx.SetOpName(mOpInfo.mName.c_str());
    rst = rst && mCtx.SetDeterministic(mOpInfo.mCtr.mDeterministic);
    rst = rst && mCtx.SetInputs({&mParam.query_, &mParam.key_, &mParam.dy_, &mParam.sparseIndices_,
        &mParam.weights_, &mParam.actualSeqLengthsQ_, &mParam.actualSeqLengthsK_});
    rst = rst && mCtx.SetOutputs({&mParam.dQuery_, &mParam.dKey_, &mParam.dWeights_});
    rst = rst && mCtx.SetAttrs({{"headNum", mParam.headNum_},
                                {"layout", mParam.layout_},
                                {"sparseMode", mParam.sparseMode_},
                                {"preTokens", mParam.preTokens_},
                                {"nextTokens", mParam.nextTokens_},
                                {"determinstic", mParam.determinstic_}});
    rst = rst && mCtx.SetKernelRunTemplateCbf(RunLightningIndexerGrad);
    rst = rst && mCtx.SetKernelTemplateMainFunc(mLIGKernelFunc);
    rst = rst && mOpInfo.SetContext(&mCtx);
    return rst;
}

bool LightningIndexerGradCase::InitOriginTilingFunc()
{
    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    mLightningIndexerGradOriginTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingForLightningIndexerGrad");
    if (mLightningIndexerGradOriginTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, LightningIndexerGrad(%p)", mLightningIndexerGradOriginTilingFunc);
        return false;
    }
    return true;
}

bool LightningIndexerGradCase::InitCurrentCasePtr()
{
    CaseWithSocversion::mCurrentCasePtr = this;
    return true;
}

bool LightningIndexerGradCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    if (mPreTilingRunCbf != nullptr) {
        mPreTilingRunCbf(tilingParam);
    }

    if (tilingParam.actSeqLenQTensor != nullptr) {
        tilingParam.actSeqLenQTensor->SetData(gert::TensorData{mParam.actualSeqLenQueryTensorData_.data()});
    }
    if (tilingParam.actSeqLenKTensor != nullptr) {
        tilingParam.actSeqLenKTensor->SetData(gert::TensorData{mParam.actualSeqLenKeyTensorData_.data()});
    }
    tilingParam.ret = Ops::Transformer::OpTiling::TilingRegistryNew::GetInstance().DoTilingImpl(tilingParam.ctx);
    return true;
}
