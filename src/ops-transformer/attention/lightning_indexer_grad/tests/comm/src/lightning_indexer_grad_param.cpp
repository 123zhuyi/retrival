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
 * \file lightning_indexer_grad_param.cpp
 * \brief LightningIndexerGrad  参数信息.
 */
#include "lightning_indexer_grad_param.h"
#include "tests/utils/log.h"
using Tensor = ops::adv::tests::utils::Tensor;
using namespace ops::adv::tests::LIG;

LightningIndexerGradParam::LightningIndexerGradParam(int64_t pBatch, int64_t pSeqlenQ, int64_t pSeqlenK, int64_t pTopK, int64_t pHeadNumQ,
    int64_t pHeadNumK, int64_t pGroupNum, int64_t pHeadDim, LayoutType pLayoutType, std::vector<int64_t> actualSeqLenQ,
    std::vector<int64_t> actualSeqLenKey, int64_t pSparseMode, int64_t pPreTokens, int64_t pNextTokens)
: batch_(pBatch), seqlenQ_(pSeqlenQ), seqlenK_(pSeqlenK), topK_(pTopK), headNumQ_(pHeadNumQ), headNumK_(pHeadNumK),
    groupNum_(pGroupNum), headDim_(pHeadDim), layoutType_(pLayoutType),actualSeqLenQuery_(std::move(actualSeqLenQ)),
    actualSeqLenKey_(std::move(actualSeqLenKey)), sparseMode_(pSparseMode), preTokens_(pPreTokens), nextTokens_(pNextTokens)
{
}

bool LightningIndexerGradParam::Init()
{
    std::vector<int64_t> layoutQ;
    std::vector<int64_t> layoutK;
    std::vector<int64_t> layoutIndex;
    std::vector<int64_t> layoutWeights;
    std::vector<int64_t> layoutdy;

    for (int i = 0; i < batch_ * seqlenQ_ * topK_; i++) {
        int32_t index = seqlenK_ / 2 ;
        sparseIndicesData_.push_back(index);
    }

    switch (layoutType_) {
        case LayoutType::BSND_SHAPE:
            layout_ = "BSND";
            layoutQ = {batch_, seqlenQ_, headNumQ_, headDim_};
            layoutK = {batch_, seqlenK_, headNumK_, headDim_};
            layoutIndex = {batch_, seqlenQ_, topK_};
            layoutWeights = {batch_, seqlenQ_, headNumQ_};
            layoutdy = {batch_, seqlenQ_, topK_};
            break;
        case LayoutType::TND:
            layout_ = "TND";
            /* 按 Q, KV 生成累加序列 */
            for (long &it : actualSeqLenQuery_) {
                auto pre = actualSeqLenQueryTensorData_.empty() ? 0 : actualSeqLenQueryTensorData_.back();
                actualSeqLenQueryTensorData_.push_back(it + pre);
                t1_ += it;
            }
            for (long &it : actualSeqLenKey_) {
                auto pre = actualSeqLenKeyTensorData_.empty() ? 0 : actualSeqLenKeyTensorData_.back();
                actualSeqLenKeyTensorData_.push_back(it + pre);
                t2_ += it;
            }
            break;
        default:
            LOG_ERR("Unknown LayoutType=%d", static_cast<int32_t>(layoutType_));
            return false;
    }

    query_ = Tensor("query", layoutQ, layout_.c_str(), dtype_, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    key_ = Tensor("key", layoutK, layout_.c_str(), dtype_, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    sparseIndices_ = Tensor("sparse_indices", layoutIndex, layout_.c_str(), ge::DataType::DT_INT32, ge::FORMAT_ND,Tensor::TensorType::REQUIRED_INPUT);
    dy_ = Tensor("dy", layoutdy, layout_.c_str(), dtype_, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    weights_ = Tensor("weights", layoutWeights, layout_.c_str(), dtype_, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    dQuery_ = Tensor("dq", layoutQ, layout_.c_str(), dtype_, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);
    dKey_ = Tensor("dk", layoutK, layout_.c_str(), dtype_, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);
    dWeights_ = Tensor("dweights", layoutWeights, layout_.c_str(), dtype_, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);

    if (!LightningIndexerGradParam::InitTensor(sparseIndices_, sparseIndicesData_)) {
        return false;
    }

    if (!actualSeqLenQuery_.empty()) {
        actualSeqLengthsQ_ = Tensor("actualSeqLengthsQ_", {static_cast<int64_t>(actualSeqLenQuery_.size())}, layout_.c_str(),
                                   ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    }
    if (!actualSeqLenKey_.empty()) {
        actualSeqLengthsK_ = Tensor("actualSeqLengthsK_", {static_cast<int64_t>(actualSeqLenKey_.size())}, layout_.c_str(),
                                   ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    }
    if (!actualSeqLenQuery_.empty() && !LightningIndexerGradParam::InitTensor(actualSeqLengthsQ_, actualSeqLenQuery_)) {
        return false;
    }
    if (!actualSeqLenKey_.empty() && !LightningIndexerGradParam::InitTensor(actualSeqLengthsK_, actualSeqLenKey_)) {
        return false;
    }
    return true;
}

bool LightningIndexerGradParam::IsUnPaddingAttention()
{
    return actualSeqLengthsQ_.GetDimNum() != 0 && !actualSeqLenQuery_.empty();
}
