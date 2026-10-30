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
 * \file nsa_compress_grad_case.cpp
 * \brief NsaCompressGrad 测试用例.
 */
#include "nsa_compress_grad_case.h"
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

#define NSA_COMPRESS_GRAD_KERNEL_PARAM                                                                                 \
    (GM_ADDR outputGrad, GM_ADDR inputKV, GM_ADDR weight, GM_ADDR actSeqLenOptional, GM_ADDR inputGradOut,             \
     GM_ADDR weightGradOut, GM_ADDR workspace, GM_ADDR tiling)

using NsaCompressGradKernelFunc = void(*) NSA_COMPRESS_GRAD_KERNEL_PARAM;

extern "C" __global__ __aicore__ void nsa_compress_grad NSA_COMPRESS_GRAD_KERNEL_PARAM;

using namespace ops::adv::tests::NsaCompressGrad;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunNsaCompressGrad(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                         std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (NsaCompressGradKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim,
                inputs[0]->GetDevData(),  // outputGrad
                inputs[1]->GetDevData(),  // inputKV
                inputs[2]->GetDevData(),  // weight
                inputs[3]->GetDevData(),  // actSeqLenOptional
                outputs[0]->GetDevData(), // inputGradOut
                outputs[1]->GetDevData(), // weightGradOut
                workspace, tilingData); // 执行内核函数，传入输入输出数据和工作空间
    return true;
}

extern "C" ge::graphStatus TilingNsaCompressGradStub(gert::TilingContext *context)
{
    auto *nsaCompressGradCase = static_cast<NsaCompressGradCase *>(Case::GetCurrentCase());
    if (nsaCompressGradCase != nullptr) {
        NsaCompressGradCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        if (!nsaCompressGradCase->DoOpTiling(p)) {
            return p.ret;
        }
        return nsaCompressGradCase->nsaCompressGradTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool NsaCompressGradCase::InitParam()
{
    outputGrad = Tensor("outputGrad", {mParam.blockNum, mParam.headNum, mParam.headDim}, "3", mParam.optionalDataType,
     ge::FORMAT_ND);
    inputKV = Tensor("inputKV", {mParam.seqLensSum, mParam.headNum, mParam.headDim}, "3", mParam.optionalDataType,
     ge::FORMAT_ND);
    weight = Tensor("weight", {mParam.blockSize, mParam.headNum}, "2", mParam.optionalDataType,
     ge::FORMAT_ND);
    actSeqLenOptional = Tensor("actSeqLenOptional", {mParam.batchSize}, "1", mParam.actSeqLenOptionalDataType, 
     ge::FORMAT_ND);

    inputGradOut = Tensor("inputGradOut", {mParam.seqLensSum, mParam.headNum, mParam.headDim}, "3",
     mParam.optionalDataType,  ge::FORMAT_ND);
    weightGradOut = Tensor("weightGradOut", {mParam.blockSize, mParam.headNum}, "2",
     mParam.optionalDataType,  ge::FORMAT_ND);

    for (long &it : mParam.actSeqLens) {
        auto pre = mParam.actSeqLensTensorData.empty() ? 0 : mParam.actSeqLensTensorData.back();
        mParam.actSeqLensTensorData.push_back(it + pre);
    }

    if (!mParam.actSeqLensTensorData.empty()) {
        actSeqLenOptional = Tensor("actSeqLens", {static_cast<int64_t>(mParam.actSeqLensTensorData.size())}, "B",
                               mParam.actSeqLenOptionalDataType, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    } else {
        actSeqLenOptional = Tensor("actSeqLens", {}, "None", mParam.actSeqLenOptionalDataType, ge::FORMAT_ND,
                               Tensor::TensorType::OPTIONAL_INPUT);
    }
    if (!ops::adv::tests::NsaCompressGrad::NsaCompressGradCase::InitTensor(actSeqLenOptional, mParam.actSeqLensTensorData)) {
        return false;
    }

    return true;
}

bool NsaCompressGradCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("NsaCompressGrad");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&outputGrad, &inputKV, &weight, &actSeqLenOptional});
    rst = rst && mCtx.SetOutputs({&inputGradOut, &weightGradOut});
    rst = rst && mCtx.SetAttrs({{"compressBlockSize", mParam.blockSize},
                                {"compressStride", mParam.blockStride},
                                {"actSeqLenType", mParam.seqLenType},
                                {"layoutOptional", mParam.layout}});

    rst = rst && mCtx.SetKernelRunCbf(RunNsaCompressGrad);
    rst = rst && mCtx.SetKernelMainFunc((void *)nsa_compress_grad);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    nsaCompressGradTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingNsaCompressGrad");
    if (nsaCompressGradTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, nsaCompressGrad(%p)", nsaCompressGradTilingFunc);
        return false;
    }
    IMPL_OP(NsaCompressGrad).Tiling(TilingNsaCompressGradStub);
    return rst;
}

bool NsaCompressGradCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool NsaCompressGradCase::Run()
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

NsaCompressGradCase::NsaCompressGradCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre,
                                           Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "NsaCompressGrad";
}

NsaCompressGradCase::NsaCompressGradCase()
{
}
NsaCompressGradCase::Param::Param()
{
}
NsaCompressGradCase::Param::Param(int64_t pHeadNum, int64_t pHeadDim, int64_t pBlockSize, int64_t pBlockStride,
                                  int64_t pBlockNum, int64_t pSeqLensSum, int64_t pBatchSize, int64_t pSeqLenType,
                                  std::string pLayout, std::vector<int64_t> pActSeqLens,
                                  ge::DataType pOptionalDataType, ge::DataType pActSeqLenOptionalDataType)
    : headNum(pHeadNum), headDim(pHeadDim), blockSize(pBlockSize), blockStride(pBlockStride), blockNum(pBlockNum),
      seqLensSum(pSeqLensSum), batchSize(pBatchSize), seqLenType(pSeqLenType), layout(pLayout), actSeqLens(std::move(pActSeqLens)),
      optionalDataType(pOptionalDataType), actSeqLenOptionalDataType(pActSeqLenOptionalDataType) 


{
}


bool NsaCompressGradCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    return true;
}