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
 * \file aclnn_nsa_compress_param.h
 * \brief NsaCompress Aclnn 参数信息.
 */

#ifndef UTEST_ACLNN_NSA_COMPRESS_PARAM_H
#define UTEST_ACLNN_NSA_COMPRESS_PARAM_H


#include "nsa_compress_case.h"
#include "tests/utils/aclnn_tensor.h"

namespace ops::adv::tests::NsaCompress {

class AclnnNsaCompressParam : public ops::adv::tests::NsaCompress::NsaCompressParam {
public:
    using AclnnTensor = ops::adv::tests::utils::AclnnTensor;

public:
    /* 输入输出 */
    AclnnTensor aclnnInput, aclnnWeight, aclnnActSeqLenOptional, aclnnOutput;
    aclIntArray *aclnnActualSeqLenIntAry = nullptr;

public:
    AclnnNsaCompressParam() = default;
    AclnnNsaCompressParam(int64_t pT, int64_t pN, int64_t pD, ge::DataType pDtype,
                          std::vector<int64_t> pActualSeqLenList, LayoutType pLayoutType, int64_t pCompressBlockSize,
                          int64_t pCompressStride, int64_t pActSeqLenType);
    ~AclnnNsaCompressParam();

    bool Init() override;
    bool IsUnPaddingAttention() override;
};

} // namespace ops::adv::tests::NsaCompress
#endif // UTEST_ACLNN_NSA_COMPRESS_PARAM_H