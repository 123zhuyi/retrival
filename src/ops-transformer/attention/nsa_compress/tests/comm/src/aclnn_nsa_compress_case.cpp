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
 * \file aclnn_nsa_compress_case.cpp
 * \brief NsaCompress Aclnn 测试用例.
 */

#include "aclnn_nsa_compress_case.h"
#include <utility>
#include "tests/utils/log.h"
#include "aclnn_nsa_compress.h"

using namespace ops::adv::tests::NsaCompress;

bool NsaCompressTilingRunCbf(void *curCase, uint64_t *workSpaceSize, aclOpExecutor **opExecutor)
{
    auto *cs = static_cast<AclnnNsaCompressCase *>(curCase);
    auto *mAclnnParam = &cs->mAclnnParam;
    auto *exp = &cs->mOpInfo.mExp;

    aclnnStatus ret;

    mAclnnParam->IsUnPaddingAttention();
    ret = aclnnNsaCompressGetWorkspaceSize(
        mAclnnParam->aclnnInput.GetAclTensor(), mAclnnParam->aclnnWeight.GetAclTensor(),
        mAclnnParam->aclnnActualSeqLenIntAry, const_cast<char *>(mAclnnParam->layoutOptional.c_str()),
        mAclnnParam->compressBlockSize, mAclnnParam->compressStride, mAclnnParam->actSeqLenType,
        mAclnnParam->aclnnOutput.GetAclTensor(), workSpaceSize, opExecutor);
    LOG_IF(ret != ACL_SUCCESS && exp->mSuccess, LOG_ERR("aclnnNsaCompressGetWorkspaceSize failed, ERROR: %d", ret));

    return ret == ACL_SUCCESS;
}

bool NsaCompressKernelRunCbf(void *curCase)
{
    auto *cs = static_cast<AclnnNsaCompressCase *>(curCase);
    auto *mAclnnParam = &cs->mAclnnParam;
    auto *ctx = &cs->mAclnnCtx;

    aclnnStatus ret;

    mAclnnParam->IsUnPaddingAttention();
    ret = aclnnNsaCompress(ctx->GetWorkspacePtr(), ctx->GetWorkspaceSize(), ctx->GetAclOpExecutor(),
                           ctx->GetAclRtStream());
    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnNsaCompress failed, ERROR: %d", ret));

    return ret == ACL_SUCCESS;
}


AclnnNsaCompressCase::AclnnNsaCompressCase()
    : NsaCompressCase(), mAclnnCtx(AclnnContext()), mAclnnParam(AclnnNsaCompressParam())
{
}

AclnnNsaCompressCase::AclnnNsaCompressCase(const char *name, bool enable, const char *dbgInfo, OpInfo opInfo,
                                           const AclnnNsaCompressParam &param, int32_t tilingTemplatePriority)
    : NsaCompressCase(name, enable, dbgInfo, std::move(opInfo), NsaCompressParam(), tilingTemplatePriority),
      mAclnnParam(param)
{
}

bool AclnnNsaCompressCase::InitParam()
{
    return mAclnnParam.Init();
}

bool AclnnNsaCompressCase::InitOpInfo()
{
    mParam.actualSeqLenTensorData = mAclnnParam.actualSeqLenTensorData;
    if (!NsaCompressCase::InitOpInfo()) {
        return false;
    }
    auto rst = mAclnnCtx.SetOpName(this->mOpInfo.mName.c_str());
    rst = rst && mAclnnCtx.SetTilingRunCbf(NsaCompressTilingRunCbf);
    rst = rst && mAclnnCtx.SetKernelRunCbf(NsaCompressKernelRunCbf);
    rst = rst && mAclnnCtx.SetOutputs({&mAclnnParam.aclnnOutput});
    rst = rst && mOpInfo.SetContext(&mAclnnCtx);
    return rst;
}

bool AclnnNsaCompressCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}


bool AclnnNsaCompressCase::Run()
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
