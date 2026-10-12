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
 * \file nsa_compress_param.h
 * \brief NsaCompress  参数信息.
 */

#ifndef NSA_COMPRESS_PARAM_H
#define NSA_COMPRESS_PARAM_H

#include <cstdint>
#include <vector>
#include "graph/types.h"
#include "tests/utils/log.h"
#include "tests/utils/tensor.h"

namespace ops::adv::tests::NsaCompress {

class NsaCompressParam {
public:
    using Tensor = ops::adv::tests::utils::Tensor;

public:
    enum class LayoutType {
        BSH,
        SBH,
        BNSD,
        BSND,
        TND,
    };

public:
    /* 设置参数 */
    int64_t t = 0;
    int64_t n = 0;
    int64_t d = 0;
    int64_t t1 = 0;

    ge::DataType dtype = ge::DataType::DT_UNDEFINED;
    std::vector<int64_t> actualSeqLenList = {};
    LayoutType layoutType = LayoutType::TND;

    int64_t compressBlockSize = 0;
    int64_t compressStride = 0;
    int64_t actSeqLenType = 0;

    /* 生成参数 */
    std::string layoutOptional;
    std::vector<int64_t> actualSeqLenTensorData = {};

    /* 输入输出 */
    Tensor input, weight, actSeqLenOptional, output;

public:
    NsaCompressParam() = default;
    NsaCompressParam(int64_t pT, int64_t pN, int64_t pD, ge::DataType pDtype, std::vector<int64_t> pActualSeqLenList,
                     LayoutType pLayoutType, int64_t pCompressBlockSize, int64_t pCompressStride,
                     int64_t pActSeqLenType);

    virtual ~NsaCompressParam() = default;

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

} // namespace ops::adv::tests::NsaCompress

#endif // NSA_COMPRESS_PARAM_H
