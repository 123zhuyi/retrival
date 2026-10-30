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
 * \file nsag_case.cpp
 * \brief NativeSelectedAttention / NativeSelectedAttentionGrad 测试用例.
 */

#include "nsag_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling_base/tiling_base.h"
using namespace Ops::Transformer::OpTiling;
/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */

#define NSAG_KERNEL_PARAM                                                                                              \
    (__gm__ uint8_t * query, __gm__ uint8_t * key, __gm__ uint8_t * value, __gm__ uint8_t * attention_out,             \
     __gm__ uint8_t * attention_out_grad, __gm__ uint8_t * softmax_max, __gm__ uint8_t * softmax_sum,                  \
     __gm__ uint8_t * topk_indices, __gm__ uint8_t * actual_seq_qlen, __gm__ uint8_t * actual_seq_kvlen,               \
     __gm__ uint8_t * atten_mask, __gm__ uint8_t * dq, __gm__ uint8_t * dk, __gm__ uint8_t * dv,                       \
     __gm__ uint8_t * workspace, __gm__ uint8_t * tiling)

typedef void(*NsagKernalFunc) NSAG_KERNEL_PARAM;

extern "C" __global__ __aicore__ void nsa_selected_attention_grad NSAG_KERNEL_PARAM;

using namespace ops::adv::tests::nsaGrad;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunNsag(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
             std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (NsagKernalFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
                inputs[3]->GetDevData(), inputs[4]->GetDevData(), inputs[5]->GetDevData(), inputs[6]->GetDevData(),
                inputs[7]->GetDevData(), inputs[8]->GetDevData(), inputs[9]->GetDevData(), inputs[10]->GetDevData(),
                outputs[0]->GetDevData(), // queryOut
                outputs[1]->GetDevData(), // valueOut
                outputs[2]->GetDevData(), // keyOut
                workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingSelectedAttentionStub(gert::TilingContext *context)
{
    auto *nsaGradCase = static_cast<NsaGradCase *>(Case::GetCurrentCase());
    if (nsaGradCase != nullptr) {
        NsaGradCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        p.actualSeqQLengthsTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(8));
        p.actualSeqKVLengthsTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(9));
        if (!nsaGradCase->DoOpTiling(p)) {
            return p.ret;
        }
        return nsaGradCase->NsagTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool NsaGradCase::InitParam()
{
    int64_t t1 = mParam.B * mParam.S1;
    int64_t t2 = mParam.B * mParam.S2;
    auto queryShape = {t1, mParam.N1, mParam.D};
    auto keyShape = {t2, mParam.N2, mParam.D};
    auto valueShape = {t2, mParam.N2, mParam.D2};
    auto attentionOutShape = {t1, mParam.N1, mParam.D2};
    auto layout = "TND";
    auto dtype = ge::DataType::DT_BF16;
    auto format = ge::FORMAT_ND;

    query = Tensor("query", queryShape, layout, dtype, format);
    key = Tensor("key", keyShape, layout, dtype, format);
    value = Tensor("value", valueShape, layout, dtype, format);
    attention_out = Tensor("attentionOut", attentionOutShape, layout, dtype, format);
    attention_out_grad = Tensor("attentionOutGrad", attentionOutShape, layout, dtype, format);
    softmax_max = Tensor("softmaxMax", {t1, mParam.N1, 8}, layout, ge::DataType::DT_FLOAT, format);
    softmax_sum = Tensor("softmaxSum", {t1, mParam.N1, 8}, layout, ge::DataType::DT_FLOAT, format);
    topk_indices =
        Tensor("topkIndices", {t1, mParam.N2, mParam.SelectedBlockCount}, layout, ge::DataType::DT_INT32, format);

    // option input
    actual_seq_qlen = Tensor("actualSeqQlen", {mParam.B}, "B", ge::DataType::DT_INT64, format);
    actual_seq_kvlen = Tensor("actualSeqKvlen", {mParam.B}, "B", ge::DataType::DT_INT64, format);
    atten_mask = Tensor("attenMask", {}, "None", ge::DataType::DT_BOOL, format);

    // output
    dq = Tensor("queryOut", queryShape, layout, dtype, format);
    dk = Tensor("keyOut", keyShape, layout, dtype, format);
    dv = Tensor("valueOut", valueShape, layout, dtype, format);


    mParam.actualSeqQData = std::vector(mParam.B, static_cast<int64_t>(0));
    mParam.actualSeqKVData = std::vector(mParam.B, static_cast<int64_t>(0));
    std::vector<int64_t> topkData(t1 * mParam.N2 * mParam.SelectedBlockCount);

    // init tensor data
    for (auto i = 0; i < t1 * mParam.N2 * mParam.SelectedBlockCount; i++) {
        topkData[i] = i;
    }
    for (auto i = 0; i < mParam.B; i++) {
        mParam.actualSeqQData[i] = i + 1;
        mParam.actualSeqKVData[i] = (i + 1) * mParam.SelectedBlockCount * mParam.SelectedBlockSize;
    }

    if (!InitTensor(actual_seq_qlen, mParam.actualSeqQData)) {
        return false;
    }
    if (!InitTensor(actual_seq_kvlen, mParam.actualSeqKVData)) {
        return false;
    }
    if (!InitTensor(topk_indices, topkData)) {
        return false;
    }

    return true;
}

bool NsaGradCase::InitOpInfo()
{
    auto *nsagKernalFunc = (void *)nsa_selected_attention_grad;

    bool rst = mCtx.SetOpName("NsaSelectedAttentionGrad");
    rst = rst && mCtx.SetDeterministic(isDeterministic);
    rst = rst && mCtx.SetInputs({&query, &key, &value, &attention_out, &attention_out_grad, &softmax_max, &softmax_sum,
                                 &topk_indices, &actual_seq_qlen, &actual_seq_kvlen, &atten_mask});
    rst = rst && mCtx.SetOutputs({&dq, &dk, &dv});
    rst = rst && mCtx.SetAttrs({{"scaleValue", mParam.scaleValue},
                                {"SelectedBlockCount", mParam.SelectedBlockCount},
                                {"SelectedBlockSize", mParam.SelectedBlockSize},
                                {"headNum", mParam.N1},
                                {"inputLayout", mParam.inputLayout},
                                {"sparseMode", mParam.sparseMode}});
    rst = rst && mCtx.SetKernelRunCbf(RunNsag);
    rst = rst && mCtx.SetTilingDataMaxSize(2280); // max tilingDataLen
    rst = rst && mCtx.SetKernelMainFunc(nsagKernalFunc);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    NsagTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingNsaSelectedAttentionGrad");
    if (NsagTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, nsag(%p)", NsagTilingFunc);
        return false;
    }
    IMPL_OP(NsaSelectedAttentionGrad).Tiling(TilingSelectedAttentionStub);
    return rst;
}

bool NsaGradCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool NsaGradCase::Run()
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

NsaGradCase::NsaGradCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, nsaGradParam param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "NsaSelectedAttentionGrad";
}

NsaGradCase::NsaGradCase()
{
}

NsaGradCase::nsaGradParam::nsaGradParam()
{
}

NsaGradCase::nsaGradParam::nsaGradParam(int64_t pB, int64_t pS1, int64_t pS2, int64_t pN1, int64_t pD, int64_t pD2,
                                        int64_t pN2, int64_t pSelectedBlockCount, int64_t pSelectedBlockSize,
                                        float pScaleValue, std::string pInputLayout, int64_t pSparseMode)
    : B(pB), S1(pS1), S2(pS2), N1(pN1), D(pD), D2(pD2), N2(pN2), SelectedBlockCount(pSelectedBlockCount),
      SelectedBlockSize(pSelectedBlockSize), scaleValue(pScaleValue), inputLayout(pInputLayout), sparseMode(pSparseMode)
{
}

bool NsaGradCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    if (tilingParam.actualSeqQLengthsTensor != nullptr && mParam.actualSeqQData.size() != 0) {
        tilingParam.actualSeqQLengthsTensor->SetData(gert::TensorData{mParam.actualSeqQData.data()});
    }
    if (tilingParam.actualSeqKVLengthsTensor != nullptr && mParam.actualSeqKVData.size() != 0) {
        tilingParam.actualSeqKVLengthsTensor->SetData(gert::TensorData{mParam.actualSeqKVData.data()});
    }
    return true;
}