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
 * \file nsa_compress_attention_case.cpp
 * \brief NsaCompressAttention 测试用例.
 */
#include "nsa_compress_attention_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
using namespace Ops::Transformer::OpTiling;
/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */

#define NSA_COMPRESS_ATTENTION_KERNEL_PARAM                                                        \
    (uint8_t *query, uint8_t *key, uint8_t *value, uint8_t *attenMask, uint8_t * actualSeqLengths, \
     uint8_t *actualSeqLengthsCmpKv, uint8_t *actualSeqLengthsSelKv, uint8_t *topkMask,            \
     uint8_t *softmaxMax, uint8_t *softmaxSum, uint8_t *attentionOut, uint8_t *topkIndicesOut,     \
     uint8_t *workspace, uint8_t *tiling)

using NsaCompressAttentionKernelFunc = void(*) NSA_COMPRESS_ATTENTION_KERNEL_PARAM;

extern "C" __global__ __aicore__ void nsa_compress_attention NSA_COMPRESS_ATTENTION_KERNEL_PARAM;

using namespace ops::adv::tests::NsaCompressAttention;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;
using NsaCompressAttentionCase = ops::adv::tests::NsaCompressAttention::NsaCompressAttentionCase;

namespace {
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;
const size_t DIM_4 = 4;
const size_t DIM_5 = 5;
const size_t DIM_6 = 6;
const size_t DIM_7 = 7;
const uint32_t MAX_TILING_DATA_SIZE = 2854;
}

bool RunNsaCompressAttention(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                         std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (NsaCompressAttentionKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim,
                inputs[0]->GetDevData(),      // query
                inputs[1]->GetDevData(),      // key
                inputs[DIM_2]->GetDevData(),  // value
                inputs[DIM_3]->GetDevData(),  // attenMask
                inputs[DIM_4]->GetDevData(),  // actualSeqLengths
                inputs[DIM_5]->GetDevData(),  // actualSeqLengthsCmpKv
                inputs[DIM_6]->GetDevData(),  // actualSeqLengthsSelKv
                inputs[DIM_7]->GetDevData(),  // topkMask
                outputs[0]->GetDevData(),     // softmaxMax
                outputs[1]->GetDevData(),     // softmaxSum
                outputs[DIM_2]->GetDevData(), // attentionOut
                outputs[DIM_3]->GetDevData(), // topkIndicesOut
                workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingForNsaCompressAttentionStub(gert::TilingContext *context)
{
    auto *nsaCompressAttentionCase = static_cast<NsaCompressAttentionCase *>(Case::GetCurrentCase());
    if (nsaCompressAttentionCase != nullptr) {
        NsaCompressAttentionCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        p.actSeqQLenTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(DIM_4));
        p.actSeqCmpKVLenTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(DIM_5));
        p.actSeqSelKVLenTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(DIM_6));
        if (!nsaCompressAttentionCase->DoOpTiling(p)) {
            return p.ret;
        }
        return nsaCompressAttentionCase->nsaCompressAttentionTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool NsaCompressAttentionCase::InitQkvAndOut()
{
    std::vector<int64_t> layoutQ;
    std::vector<int64_t> layoutK;
    std::vector<int64_t> layoutV;
    std::vector<int64_t> layoutAttenOut;
    std::string layoutSoftmaxStr;
    std::vector<int64_t> layoutSoftmax;
    std::string layoutTopkIndicesStr;
    std::vector<int64_t> layoutTopkIndices;
    switch (mParam.layoutType) {
        case LayoutType::TND:
            mParam.layout = "TND";
            layoutSoftmaxStr = "T_N1_8";
            layoutTopkIndicesStr = "N2_T_selBlkCnt";
            for (long &it : mParam.actualSeqQLenList) {
                auto pre = mParam.actualSeqQLenTensorData.empty() ? 0 : mParam.actualSeqQLenTensorData.back();
                mParam.actualSeqQLenTensorData.push_back(it + pre);
                mParam.t1 += it;
            }
            for (long &it : mParam.actualCmpSeqKvLenList) {
                auto pre = mParam.actualCmpSeqKVLenTensorData.empty() ? 0 : mParam.actualCmpSeqKVLenTensorData.back();
                mParam.actualCmpSeqKVLenTensorData.push_back(it + pre);
                mParam.t2 += it;
            }
            for (long &it : mParam.actualSelSeqKvLenList) {
                auto pre = mParam.actualSelSeqKVLenTensorData.empty() ? 0 : mParam.actualSelSeqKVLenTensorData.back();
                mParam.actualSelSeqKVLenTensorData.push_back(it + pre);
            }
            layoutQ = {mParam.n2, mParam.t1, mParam.g, mParam.d1};
            layoutK = {mParam.t2, mParam.n2, mParam.d1};
            layoutV = {mParam.t2, mParam.n2, mParam.d2};
            layoutAttenOut = {mParam.n2, mParam.t1, mParam.g, mParam.d2};
            layoutSoftmax = {mParam.t1, mParam.n2 * mParam.g, 8};
            layoutTopkIndices = {mParam.n2, mParam.t1, mParam.selBlkCnt};
            break;
        default:
            LOG_ERR("Unknown LayoutType=%d", static_cast<int32_t>(mParam.layoutType));
            return false;
    }
    query = Tensor("query", layoutQ, mParam.layout.c_str(), mParam.dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    key = Tensor("key", layoutK, mParam.layout.c_str(), mParam.dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    value = Tensor("value", layoutV, mParam.layout.c_str(), mParam.dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    attenOut = Tensor("attenOut", layoutAttenOut, mParam.layout.c_str(), mParam.dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);
    softmaxMax = Tensor("softmaxMax", layoutSoftmax, layoutSoftmaxStr.c_str(), ge::DataType::DT_FLOAT, ge::FORMAT_ND,
                        Tensor::TensorType::REQUIRED_OUTPUT);
    softmaxSum = Tensor("softmaxSum", layoutSoftmax, layoutSoftmaxStr.c_str(), ge::DataType::DT_FLOAT, ge::FORMAT_ND,
                        Tensor::TensorType::REQUIRED_OUTPUT);
    topkIndicesOut = Tensor("topkIndicesOut", layoutTopkIndices, layoutTopkIndicesStr.c_str(), ge::DataType::DT_INT32, ge::FORMAT_ND,
                            Tensor::TensorType::REQUIRED_OUTPUT);
    return true;
}

bool NsaCompressAttentionCase::InitOptInputs()
{
    std::string layoutAttenMaskStr;
    std::vector<int64_t> layoutAttenMask;
    switch (mParam.attenMaskShapeType) {
        case AttenMaskShapeType::S1_S2:
            layoutAttenMaskStr = "S1_S2";
            layoutAttenMask = {mParam.s1, mParam.s2}; //maxS1, maxS2
            break;
        default:
            layoutAttenMaskStr = "None";
            layoutAttenMask = {};
            break;
    }
    attenMask = Tensor("attenMask", layoutAttenMask, layoutAttenMaskStr.c_str(), mParam.attenMaskDtype, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    std::string layoutTopkMaskStr;
    std::vector<int64_t> layoutTopkMask;
    switch (mParam.topkMaskShapeType) {
        case TopkMaskShapeType::S1_S2:
            layoutTopkMaskStr = "S1_S2";
            layoutTopkMask = {mParam.s1, mParam.s2}; //maxS1, masS2'
            break;
        default:
            layoutTopkMaskStr = "None";
            layoutTopkMask = {};
            break;
    }
    topkMask = Tensor("topkMask", layoutTopkMask, layoutTopkMaskStr.c_str(), mParam.topkMaskDtype, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    if (!mParam.actualSeqQLenTensorData.empty()) {
        actualSeqQLen = Tensor("actualSeqQLen", {static_cast<int64_t>(mParam.actualSeqQLenTensorData.size())}, "B",
                               ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    } else {
        actualSeqQLen = Tensor("actualSeqQLen", {}, "None", ge::DataType::DT_INT64, ge::FORMAT_ND,
                               Tensor::TensorType::OPTIONAL_INPUT);
    }
    if (!mParam.actualCmpSeqKVLenTensorData.empty()) {
        actualCmpSeqKvLen = Tensor("actualCmpSeqKvLen", {static_cast<int64_t>(mParam.actualCmpSeqKVLenTensorData.size())}, "B",
                                   ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    } else {
        actualCmpSeqKvLen = Tensor("actualCmpSeqKvLen", {}, "None", ge::DataType::DT_INT64, ge::FORMAT_ND,
                                   Tensor::TensorType::OPTIONAL_INPUT);
    }
    if (!mParam.actualSelSeqKVLenTensorData.empty()) {
        actualSelSeqKvLen = Tensor("actualSelSeqKvLen", {static_cast<int64_t>(mParam.actualSelSeqKVLenTensorData.size())}, "B",
                                   ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    } else {
        actualSelSeqKvLen = Tensor("actualSelSeqKvLen", {}, "None", ge::DataType::DT_INT64, ge::FORMAT_ND,
                                   Tensor::TensorType::OPTIONAL_INPUT);
    }

    if (!InitTensor(actualSeqQLen, mParam.actualSeqQLenTensorData)) {
        return false;
    }
    if (!InitTensor(actualCmpSeqKvLen, mParam.actualCmpSeqKVLenTensorData)) {
        return false;
    }
    if (!InitTensor(actualSelSeqKvLen, mParam.actualSelSeqKVLenTensorData)) {
        return false;
    }
    return true;
}

bool NsaCompressAttentionCase::InitParam()
{
    if (!this->InitQkvAndOut() || !this->InitOptInputs()) {
        return false;
    }
    return true;
}

bool NsaCompressAttentionCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("NsaCompressAttention");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&query, &key, &value, &attenMask, &actualSeqQLen, &actualCmpSeqKvLen,
                                 &actualSelSeqKvLen, &topkMask});
    rst = rst && mCtx.SetOutputs({&softmaxMax, &softmaxSum, &attenOut, &topkIndicesOut});
    rst = rst && mCtx.SetAttrs({{"scale_value", mParam.scale},
                                {"scale_value", mParam.n2 * mParam.g},
                                {"input_layout", mParam.layout},
                                {"sparse_mode", mParam.sparseMode},
                                {"compress_block_size", mParam.cmpBlkSize},
                                {"compress_stride", mParam.cmpStride},
                                {"select_block_size", mParam.selBlkSize},
                                {"select_block_count", mParam.selBlkCnt}});
    rst = rst && mCtx.SetKernelRunCbf(RunNsaCompressAttention);
    rst = rst && mCtx.SetTilingDataMaxSize(MAX_TILING_DATA_SIZE);
    rst = rst && mCtx.SetKernelMainFunc((void *)nsa_compress_attention);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    nsaCompressAttentionTilingFunc = (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingNsaCompressAttention");
    if (nsaCompressAttentionTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, nsaCompressAttentionTilingFunc(%p)", nsaCompressAttentionTilingFunc);
        return false;
    }
    IMPL_OP(NsaCompressAttention).Tiling(TilingForNsaCompressAttentionStub);
    return rst;
}

bool NsaCompressAttentionCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool NsaCompressAttentionCase::Run()
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

NsaCompressAttentionCase::NsaCompressAttentionCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "NsaCompressAttention";
}

NsaCompressAttentionCase::NsaCompressAttentionCase()
{
}
NsaCompressAttentionCase::Param::Param(int64_t pB, int64_t pN2, int64_t pG, int64_t pS1, int64_t pS2, int64_t pD1, int64_t pD2, ge::DataType pDtype,
                                       LayoutType pLayoutType, float pScale, int64_t pSparseMode, int64_t pCmpBlkSize, int64_t pCmpStride,
                                       int64_t pSelBlkSize, int64_t pSelBlkCnt, AttenMaskShapeType pAttenMaskShapeType,
                                       TopkMaskShapeType pTopkMaskShapeType, ge::DataType pAttenMaskDtype, ge::DataType pTopkMaskDtype,
                                       std::vector<int64_t> pActualSeqQLenList, std::vector<int64_t> pActualCmpSeqKvLenList,
                                       std::vector<int64_t> pActualSelSeqKvLenList)
    : b(pB), n2(pN2), g(pG), s1(pS1), s2(pS2), d1(pD1), d2(pD2), dtype(pDtype), layoutType(pLayoutType), scale(pScale), sparseMode(pSparseMode),
      cmpBlkSize(pCmpBlkSize), cmpStride(pCmpStride), selBlkSize(pSelBlkSize), selBlkCnt(pSelBlkCnt), attenMaskShapeType(pAttenMaskShapeType),
      topkMaskShapeType(pTopkMaskShapeType), attenMaskDtype(pAttenMaskDtype), topkMaskDtype(pTopkMaskDtype), actualSeqQLenList(pActualSeqQLenList),
      actualCmpSeqKvLenList(pActualCmpSeqKvLenList), actualSelSeqKvLenList(pActualSelSeqKvLenList)
{
}


bool NsaCompressAttentionCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    if (tilingParam.actSeqQLenTensor != nullptr) {
        tilingParam.actSeqQLenTensor->SetData(gert::TensorData{mParam.actualSeqQLenTensorData.data()});
    }
    if (tilingParam.actSeqCmpKVLenTensor != nullptr) {
        tilingParam.actSeqCmpKVLenTensor->SetData(gert::TensorData{mParam.actualCmpSeqKVLenTensorData.data()});
    }
    if (tilingParam.actSeqSelKVLenTensor != nullptr) {
        tilingParam.actSeqSelKVLenTensor->SetData(gert::TensorData{mParam.actualSelSeqKVLenTensorData.data()});
    }
    /* 按优先级 Tiling */
    tilingParam.ret = Ops::Transformer::OpTiling::TilingRegistryNew::GetInstance().DoTilingImpl(tilingParam.ctx, {0});
    return true;
}