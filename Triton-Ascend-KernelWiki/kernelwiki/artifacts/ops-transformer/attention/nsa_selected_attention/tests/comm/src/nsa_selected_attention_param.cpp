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
 * \file nsa_selected_attention_param.cpp
 * \brief NsaSelectedAttention 参数信息.
 */

#include "nsa_selected_attention_param.h"
#include "tests/utils/log.h"

using ops::adv::tests::utils::Tensor;

using namespace ops::adv::tests::nsa_selected_attention_ns;

Param::Param(int64_t pB, int64_t pN2, int64_t pG, int64_t pD, int64_t pD2, ge::DataType pDtype, std::string pLayout,
             float pScale, int64_t pSparseMode, int64_t pSelectedBlockSize, int64_t pSelectedBlockCount,
             std::vector<int64_t> pActualSeqQLenList, std::vector<int64_t> pActualSeqKVLenList)
    : b(pB), n2(pN2), g(pG), d(pD), d2(pD2), dtype(pDtype), layout(pLayout), scale(pScale), sparseMode(pSparseMode),
      selectedBlockSize(pSelectedBlockSize), selectedBlockCount(pSelectedBlockCount),
      actualSeqQLenList(std::move(pActualSeqQLenList)), actualSeqKVLenList(std::move(pActualSeqKVLenList))
{
}

Param::Param(int64_t pB, int64_t pN2, int64_t pG, int64_t pD, int64_t pD2, ge::DataType pDtype, std::string pLayout,
             float pScale, int64_t pSparseMode, int64_t pSelectedBlockSize, int64_t pSelectedBlockCount,
             std::vector<int64_t> pActualSeqQLenList, std::vector<int64_t> pActualSeqKVLenList, bool pNeedAttenMask)
    : b(pB), n2(pN2), g(pG), d(pD), d2(pD2), dtype(pDtype), layout(pLayout), scale(pScale), sparseMode(pSparseMode),
      selectedBlockSize(pSelectedBlockSize), selectedBlockCount(pSelectedBlockCount),
      actualSeqQLenList(std::move(pActualSeqQLenList)), actualSeqKVLenList(std::move(pActualSeqKVLenList)), needAttenMask(pNeedAttenMask)
{
}

bool Param::Init()
{
    n1 = g * n2;
    // 根据 seq len list 计算 t1、t2
    if (layout == "TND") {
        for (long &it : actualSeqQLenList) {
            t1 += it;
            auto pre = actualSeqQLenTensorData.empty() ? 0 : actualSeqQLenTensorData.back();
            actualSeqQLenTensorData.push_back(it + pre);
        }
        for (long &it : actualSeqKVLenList) {
            t2 += it;
            auto pre = actualSeqKVLenTensorData.empty() ? 0 : actualSeqKVLenTensorData.back();
            actualSeqKVLenTensorData.push_back(it + pre);
        }
    }
    std::vector<int64_t> qShape = {t1, n1, d};
    std::vector<int64_t> kShape = {t2, n2, d};
    std::vector<int64_t> vShape = {t2, n2, d2};
    std::vector<int64_t> topkIndicesShape = {t1, n2, selectedBlockCount};
    /*构造输入输出*/
    query = Tensor("query", qShape, layout.c_str(), dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    key = Tensor("key", kShape, layout.c_str(), dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    value = Tensor("value", vShape, layout.c_str(), dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    topkIndices = Tensor("topk_indices", topkIndicesShape, layout.c_str(), ge::DT_INT32, ge::FORMAT_ND,
                         Tensor::TensorType::REQUIRED_INPUT);
    softmaxMax = Tensor("softmax_max", {t1, n1, 8}, layout.c_str(), ge::DT_FLOAT, ge::FORMAT_ND,
                        Tensor::TensorType::REQUIRED_OUTPUT);
    softmaxSum = Tensor("softmax_sum", {t1, n1, 8}, layout.c_str(), ge::DT_FLOAT, ge::FORMAT_ND,
                        Tensor::TensorType::REQUIRED_OUTPUT);
    attentionOut = Tensor("attention_out", {t1, n1, d2}, layout.c_str(), ge::DT_FLOAT, ge::FORMAT_ND,
                          Tensor::TensorType::REQUIRED_OUTPUT);

    attenMask = !needAttenMask ? Tensor("atten_mask", {}, "S1_S2", ge::DT_BOOL, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT) :
        Tensor("atten_mask", {2048, 2048}, "S1_S2", ge::DT_BOOL, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);

    if (!actualSeqQLenTensorData.empty()) {
        actualSeqQLen = Tensor("actual_seq_qlen", {static_cast<int64_t>(actualSeqQLenTensorData.size())}, "B",
                               ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    } else {
        actualSeqQLen = Tensor("actual_seq_kvlen", {}, "None", ge::DataType::DT_INT64, ge::FORMAT_ND,
                               Tensor::TensorType::REQUIRED_INPUT);
    }
    if (!actualSeqKVLenTensorData.empty()) {
        actualSeqKvLen = Tensor("actualSeqKvLen", {static_cast<int64_t>(actualSeqKVLenTensorData.size())}, "B",
                                ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    } else {
        actualSeqKvLen = Tensor("actualSeqKvLen", {}, "None", ge::DataType::DT_INT64, ge::FORMAT_ND,
                                Tensor::TensorType::REQUIRED_INPUT);
    }
    // 初始化 tensor 数据
    if (!ops::adv::tests::nsa_selected_attention_ns::Param::InitTensor(actualSeqQLen, actualSeqQLenTensorData)) {
        return false;
    }
    if (!ops::adv::tests::nsa_selected_attention_ns::Param::InitTensor(actualSeqKvLen, actualSeqKVLenTensorData)) {
        return false;
    }

    return true;
}