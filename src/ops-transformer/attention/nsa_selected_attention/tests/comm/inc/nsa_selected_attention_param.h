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
 * \file nsa_selected_attention_param.h
 * \brief NsaSelectedAttention 参数信息.
 */

#ifndef UTEST_NSA_SELECTED_ATTENTION_PARAM_H
#define UTEST_NSA_SELECTED_ATTENTION_PARAM_H

#include <cstdint>
#include <vector>
#include <register/op_impl_registry.h>
#include "graph/types.h"
#include "tests/utils/log.h"
#include "tests/utils/tensor.h"

namespace ops::adv::tests::nsa_selected_attention_ns {
using ops::adv::tests::utils::Tensor;

class Param {
public:
    /* 参数设置 */
    int64_t b = 0;
    int64_t n2 = 0;
    int64_t g = 0;
    int64_t d = 0;
    int64_t d2 = 0;
    ge::DataType dtype = ge::DataType::DT_UNDEFINED;
    std::string layout = "TND";
    float scale = 1.0f;
    int64_t sparseMode = 0;
    int64_t selectedBlockSize = 64;
    int64_t selectedBlockCount = 16;
    std::vector<int64_t> actualSeqQLenList = {}; // 每个 batch Seq 实际长度, 内部累加计算 actualSeqQLenTensorData 值
    std::vector<int64_t> actualSeqKVLenList = {};
    bool needAttenMask = true;

    /* 内部计算 */
    int64_t n1 = 0; // g * n2
    int64_t t1 = 0;
    int64_t t2 = 0;
    std::vector<int64_t> actualSeqQLenTensorData = {};
    std::vector<int64_t> actualSeqKVLenTensorData = {};

    /* 输入输出 */
    Tensor query, key, value, topkIndices, attenMask, actualSeqQLen, actualSeqKvLen, softmaxMax, softmaxSum,
        attentionOut;

public:
    Param() = default;
    Param(int64_t pB, int64_t pN2, int64_t pG, int64_t pD, int64_t pD2, ge::DataType pDtype, std::string pLayout,
          float pScale, int64_t pSparseMode, int64_t pSelectedBlockSize, int64_t pSelectedBlockCount,
          std::vector<int64_t> pActualSeqQLenList, std::vector<int64_t> pActualSeqKVLenList);
    Param(int64_t pB, int64_t pN2, int64_t pG, int64_t pD, int64_t pD2, ge::DataType pDtype, std::string pLayout,
          float pScale, int64_t pSparseMode, int64_t pSelectedBlockSize, int64_t pSelectedBlockCount,
          std::vector<int64_t> pActualSeqQLenList, std::vector<int64_t> pActualSeqKVLenList, bool needAttenMask);

    virtual ~Param() = default;

    virtual bool Init();

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

} // namespace ops::adv::tests::nsa_selected_attention_ns
#endif // UTEST_NSA_SELECTED_ATTENTION_PARAM_H