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
 * \file nsa_compress_case.cpp
 * \brief nsa_compress 测试用例.
 */
#include "nsa_compress_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling/nsa/tiling_data.h"
#include "tiling_base/tiling_templates_registry.h"
using namespace Ops::Transformer::OpTiling;
using NsaCompressKernelFunc = void(*) NSA_COMPRESS_KERNEL_PARAM;
using namespace ops::adv::tests::NsaCompress;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;
using NsaCompressCase = ops::adv::tests::NsaCompress::NsaCompressCase;

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */

namespace ops::adv::tests::NsaCompress {
extern "C" __global__ __aicore__ void nsa_compress_fp16 NSA_COMPRESS_KERNEL_PARAM;
} // namespace ops::adv::tests::NsaCompress

namespace optiling {
ASCENDC_EXTERN_C ge::graphStatus NsaCompress(gert::TilingContext *context);
} // namespace optiling

namespace {
const size_t NSA_COMPRESS_ACTUAL_SEQ_LENGTH_INPUT_INDEX = 2UL;

ASCENDC_EXTERN_C ge::graphStatus TilingForNsaCompressStub(gert::TilingContext *context)
{
    auto *nsaCompressCase = static_cast<NsaCompressCase *>(Case::GetCurrentCase());
    if (nsaCompressCase != nullptr) {
        NsaCompressCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        p.actSeqLenTensor =
            const_cast<gert::Tensor *>(context->GetOptionalInputTensor(NSA_COMPRESS_ACTUAL_SEQ_LENGTH_INPUT_INDEX));

        if (!nsaCompressCase->DoOpTiling(p)) {
            return p.ret;
        }
        return nsaCompressCase->mNsaCompressOriginTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool RunNsaCompress(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                    std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    (void)blockDim;
    // Kernel 运行
    auto kernelFunc = (NsaCompressCase::NsaCompressKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, 1,
                inputs[0]->GetDevData(),
                inputs[1]->GetDevData(),
                inputs[2]->GetDevData(),
                outputs[0]->GetDevData(),
                workspace, tilingData);
    return true;
}
} // namespace


NsaCompressCase::NsaCompressCase() : NsaCompressCase("Undefined", true, "", OpInfo(), NsaCompressParam(), 0)
{
}

NsaCompressCase::NsaCompressCase(const char *name, bool enable, const char *dbgInfo, OpInfo opinfo,
                                 NsaCompressParam param, int32_t tilingTemplatePriority)
    : Case(name, enable, dbgInfo, tilingTemplatePriority), mOpInfo(std::move(opinfo)), mParam(std::move(param))
{
    mOpInfo.mName = "NsaCompress";
    mNsaCompressOriginTilingFuncName = "TilingNsaCompress";

    mNsaCompressKernelFunc = (void *)nsa_compress_fp16;
}

bool NsaCompressCase::Run()
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


bool NsaCompressCase::InitParam()
{
    if (!mParam.Init()) {
        return false;
    }
    return true;
}


bool NsaCompressCase::InitOpInfo()
{
    bool rst = this->InitOpInfoCtx();
    if (!rst) {
        return rst;
    }

    if (!this->InitOriginTilingFunc()) {
        return false;
    }
    IMPL_OP(NsaCompress).Tiling(TilingForNsaCompressStub);

    return true;
}

bool NsaCompressCase::InitOpInfoCtx()
{
    bool rst = mCtx.SetOpName(mOpInfo.mName.c_str());
    rst = rst && mCtx.SetDeterministic(mOpInfo.mCtr.mDeterministic);
    rst = rst && mCtx.SetInputs({&mParam.input, &mParam.weight, &mParam.actSeqLenOptional});
    rst = rst && mCtx.SetOutputs({&mParam.output});
    rst = rst && mCtx.SetAttrs({{"layoutOptional", mParam.layoutOptional},
                                {"compressBlockSize", mParam.compressBlockSize},
                                {"compressStride", mParam.compressStride},
                                {"actSeqLenType", mParam.actSeqLenType}});
    rst = rst && mCtx.SetKernelRunCbf(RunNsaCompress);
    rst = rst && mCtx.SetKernelMainFunc((void *)mNsaCompressKernelFunc);
    rst = rst && mOpInfo.SetContext(&mCtx);
    return rst;
}

bool NsaCompressCase::InitOriginTilingFunc()
{
    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    mNsaCompressOriginTilingFunc =
        (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym(mNsaCompressOriginTilingFuncName.c_str());
    if (mNsaCompressOriginTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, nsaCompress(%p)", mNsaCompressOriginTilingFunc);
        return false;
    }
    return true;
}

bool NsaCompressCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}


bool NsaCompressCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    if (mPreTilingRunCbf != nullptr) {
        mPreTilingRunCbf(tilingParam);
    }

    if (tilingParam.actSeqLenTensor != nullptr) {
        tilingParam.actSeqLenTensor->SetData(gert::TensorData{mParam.actualSeqLenTensorData.data()});
    }
    tilingParam.ret = Ops::Transformer::OpTiling::TilingRegistryNew::GetInstance().DoTilingImpl(tilingParam.ctx);
    return true;
}