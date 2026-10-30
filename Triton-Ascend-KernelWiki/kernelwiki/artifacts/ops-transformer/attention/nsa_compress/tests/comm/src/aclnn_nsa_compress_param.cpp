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
 * \file aclnn_nsa_compress_param.cpp
 * \brief FlashAttentionScore / FlashAttentionScoreGrad Aclnn 参数信息.
 */

#include "aclnn_nsa_compress_param.h"
#include <utility>
#include "tests/utils/case.h"
#include "tests/utils/io.h"
#include "tests/utils/log.h"

namespace {
template <class T> bool InitAclIntArray(aclIntArray **intArray, std::vector<T> &hostData)
{
    if (intArray == nullptr) {
        return false;
    }
    if (*intArray != nullptr) {
        auto ret = aclDestroyIntArray(*intArray);
        LOG_IF_EXPR(ret != ACL_SUCCESS, LOG_ERR("aclDestroyIntArray failed, ERROR: %d", ret), *intArray = nullptr);
    }
    if (hostData.empty()) {
        return true;
    }
    *intArray = aclCreateIntArray(hostData.data(), hostData.size());
    if (*intArray == nullptr) {
        LOG_ERR("aclCreateIntArray failed.");
        return false;
    }
    return true;
}
} // namespace

using namespace ops::adv::tests::NsaCompress;

AclnnNsaCompressParam::AclnnNsaCompressParam(int64_t pT, int64_t pN, int64_t pD, ge::DataType pDtype,
                                             std::vector<int64_t> pActualSeqLenList, LayoutType pLayoutType,
                                             int64_t pCompressBlockSize, int64_t pCompressStride,
                                             int64_t pActSeqLenType)
    : NsaCompressParam(pT, pN, pD, pDtype, std::move(pActualSeqLenList), pLayoutType, pCompressBlockSize,
                       pCompressStride, pActSeqLenType),
      aclnnActualSeqLenIntAry(nullptr)
{
}


AclnnNsaCompressParam::~AclnnNsaCompressParam()
{
    if (aclnnActualSeqLenIntAry != nullptr) {
        auto ret = aclDestroyIntArray(aclnnActualSeqLenIntAry);
        LOG_IF_EXPR(ret != ACL_SUCCESS, LOG_ERR("aclDestroyIntArray failed, ERROR: %d", ret),
                    aclnnActualSeqLenIntAry = nullptr);
    }
}

bool AclnnNsaCompressParam::Init()
{
    if (!NsaCompressParam::Init()) {
        return false;
    }
    aclnnInput = ops::adv::tests::utils::AclnnTensor(input);
    aclnnWeight = ops::adv::tests::utils::AclnnTensor(weight);
    aclnnActSeqLenOptional = ops::adv::tests::utils::AclnnTensor(actSeqLenOptional);
    aclnnOutput = ops::adv::tests::utils::AclnnTensor(output);

    if (!InitAclIntArray(&aclnnActualSeqLenIntAry, actualSeqLenTensorData)) {
        return false;
    }

    auto *cs = static_cast<ops::adv::tests::utils::Case *>(ops::adv::tests::utils::Case::GetCurrentCase());
    LOG_IF_EXPR(cs == nullptr, LOG_ERR("Can't get current case"), return false);

    for (auto *it : {&aclnnInput, &aclnnWeight, &aclnnActSeqLenOptional, &aclnnOutput}) {
        it->FreeDevData();
        if (it->GetExpDataSize() <= 0) {
            continue;
        }
        auto *devData = it->AllocDevData(0, 0);
        if (devData == nullptr) {
            return false;
        }
        if (it->IsOutput()) {
            continue;
        }
        std::string filePath = std::string(cs->GetRootPath()) + it->Name() + ".bin";
        if (ops::adv::tests::utils::FileExist(filePath)) {
            if (!it->LoadFileToDevData(filePath)) {
                return false;
            }
        }
    }
    return true;
}

bool AclnnNsaCompressParam::IsUnPaddingAttention()
{
    if (!NsaCompressParam::IsUnPaddingAttention()) {
        return false;
    }
    aclnnStatus ret;
    uint64_t aclnnActualSeqLenIntArySize = 0;
    ret = aclGetIntArraySize(aclnnActualSeqLenIntAry, &aclnnActualSeqLenIntArySize);
    LOG_IF_EXPR(ret != ACL_SUCCESS, LOG_ERR("aclGetIntArraySize failed, ERROR: %d", ret), return false);
    return aclnnActualSeqLenIntArySize != 0;
}
