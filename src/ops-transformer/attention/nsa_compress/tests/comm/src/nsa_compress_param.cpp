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
 * \file nsa_compress_param.cpp
 * \brief NsaCompress  参数信息.
 */

#include "nsa_compress_param.h"
#include "tests/utils/log.h"

using Tensor = ops::adv::tests::utils::Tensor;

using namespace ops::adv::tests::NsaCompress;


NsaCompressParam::NsaCompressParam(int64_t pT, int64_t pN, int64_t pD, ge::DataType pDtype,
                                   std::vector<int64_t> pActualSeqLenList, LayoutType pLayoutType,
                                   int64_t pCompressBlockSize, int64_t pCompressStride, int64_t pActSeqLenType)

    : t(pT), n(pN), d(pD), dtype(pDtype), actualSeqLenList(std::move(pActualSeqLenList)), layoutType(pLayoutType),
      compressBlockSize(pCompressBlockSize), compressStride(pCompressStride), actSeqLenType(pActSeqLenType)
{
}


bool NsaCompressParam::Init()
{
    switch (layoutType) {
        case LayoutType::BSH:
            layoutOptional = "BSH";
            break;
        case LayoutType::SBH:
            layoutOptional = "SBH";
            break;
        case LayoutType::BNSD:
            layoutOptional = "BNSD";
            break;
        case LayoutType::BSND:
            layoutOptional = "BSND";
            break;
        case LayoutType::TND:
            layoutOptional = "TND";
            break;
        default:
            LOG_ERR("Unknown LayoutType=%d", static_cast<int32_t>(layoutType));
            return false;
    }

    input =
        Tensor("input", {t, n, d}, layoutOptional.c_str(), dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    weight = Tensor("weight", {compressBlockSize, n}, "2", dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);

    int64_t pre_seq_len = 0;
    for (int64_t &it : actualSeqLenList) {
        actualSeqLenTensorData.push_back(it);
        int64_t cur_seq_len = it - pre_seq_len;
        if (cur_seq_len >= compressBlockSize) {
            t1 += (cur_seq_len - compressBlockSize + compressStride) / compressStride;
        }
        pre_seq_len += cur_seq_len;
    }

    if (!actualSeqLenTensorData.empty()) {
        actSeqLenOptional = Tensor("actSeqLenOptional", {static_cast<int64_t>(actualSeqLenTensorData.size())}, "B",
                                   ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    } else {
        LOG_ERR("Currently, actSeqLenOptional must not be empty.");
    }

    output =
        Tensor("output", {t1, n, d}, layoutOptional.c_str(), dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);

    /**
     * TensorData 初始化
     * 出于性能角度考虑, 此处仅申请 Tiling 阶段必需的 TensorData
     */
    if (!ops::adv::tests::NsaCompress::NsaCompressParam::InitTensor(actSeqLenOptional, actualSeqLenTensorData)) {
        return false;
    }
    return true;
}

bool NsaCompressParam::IsUnPaddingAttention()
{
    return actSeqLenOptional.GetDimNum() != 0 && !actualSeqLenTensorData.empty();
}
