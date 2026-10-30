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
 * \file lightning_indexer_grad_param.h
 * \brief LightningIndexerGrad  参数信息.
 */
#ifndef LIGHTNING_INDEXER_GRAD_PARAM_H
#define LIGHTNING_INDEXER_GRAD_PARAM_H

#include <cstdint>
#include <vector>
#include "graph/types.h"
#include "tests/utils/log.h"
#include "tests/utils/tensor.h"

namespace ops::adv::tests::LIG {
class LightningIndexerGradParam {
public:
    using Tensor = ops::adv::tests::utils::Tensor;
public:
    enum class LayoutType {
        BSND_SHAPE,
        TND,
    };
public:
    /* 设置参数 */
    int64_t batch_ = 0;
    uint32_t seqlenQ_ = 0;
    uint32_t seqlenK_ = 0;
    uint32_t topK_ = 0;
    uint32_t headNumQ_ = 0;
    uint32_t headNumK_ = 0;
    uint32_t groupNum_ = 0;
    uint32_t headDim_ = 0;

    ge::DataType dtype_ = ge::DataType::DT_BF16;
    std::vector<int64_t> actualSeqLenQuery_ = {};
    std::vector<int64_t> actualSeqLenKey_ = {};
    LayoutType layoutType_ = LayoutType::BSND_SHAPE;
    int64_t headNum_ = 0;
    std::string layout_;
    int64_t sparseMode_ = 0;
    int64_t preTokens_ = 0;
    int64_t nextTokens_ = 0;
    bool determinstic_ = false;
    /* 生成参数 */
    std::string layoutOptional_;
    std::vector<int64_t> actualSeqLenQueryTensorData_ = {};
    std::vector<int64_t> actualSeqLenKeyTensorData_ = {};
    std::vector<int64_t> sparseIndicesData_ = {};
    int64_t t1_ = 0;
    int64_t t2_ = 0;
    /* 输入输出 */
    Tensor query_, key_, dy_, sparseIndices_, weights_, actualSeqLengthsQ_, actualSeqLengthsK_, dQuery_,
        dKey_, dWeights_;
public:
    LightningIndexerGradParam() = default;
    LightningIndexerGradParam(int64_t pBatch, int64_t pSeqlenQ, int64_t pSeqlenK, int64_t pTopK, int64_t pHeadNumQ,
        int64_t pHeadNumK, int64_t pGroupNum, int64_t pHeadDim, LayoutType pLayoutType,
        std::vector<int64_t> actualSeqLenQ, std::vector<int64_t> actualSeqLenKey, int64_t pSparseMode,
        int64_t pPreTokens, int64_t pNextTokens);

    virtual ~LightningIndexerGradParam() = default;

    virtual bool Init();

    virtual bool IsUnPaddingAttention();

    template <class T> static bool InitTensor(Tensor &tensor, std::vector<T> &hostData)
    {
        if (hostData.empty()) {
            return true;
        }
        int64_t expMinSize = hostData.size() * sizeof(T);
        if (tensor.AllocDevData(0, expMinSize) == nullptr) {
            LOG_ERR("Tensor(%s, %ld) AllocDevData Failed.", tensor.Name().c_str(), expMinSize);
            return false;
        }
        return tensor.CopyHostToDevData(hostData);
    }
};
} // namespace ops::adv::tests::LIG
#endif // LIGHTNING_INDEXER_GRAD_PARAM_H
