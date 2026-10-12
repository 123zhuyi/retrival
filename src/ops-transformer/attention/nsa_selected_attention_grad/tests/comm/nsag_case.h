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
 * \file ifa_case.h
 * \brief IncreFlashAttention 测试用例.
 */

#pragma once
#include <vector>
#include <cstdint>
#include "graph/types.h"
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "tests/utils/tensor.h"
#include "tests/utils/tensor_list.h"
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>

namespace ops::adv::tests::nsaGrad {

/**
 * 算子 NativeSelectedAttentionGrad 参数
 */

class NsaGradCase : public ops::adv::tests::utils::Case {
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using Tensor = ops::adv::tests::utils::Tensor;
    using TensorList = ops::adv::tests::utils::TensorList;

public:
    class nsaGradParam {
    public:
        /* 设置参数 */
        int64_t B = 0;
        int64_t S1 = 0;
        int64_t S2 = 0;
        int64_t N1 = 0;
        int64_t D = 0;
        int64_t D2 = 0;
        int64_t N2 = 0;
        int64_t SelectedBlockCount = 1;
        int64_t SelectedBlockSize = 1;
        float scaleValue = 1.0;
        int64_t headNum = 1;
        std::string inputLayout = "TND";
        int64_t sparseMode = 0;
        ge::DataType qDataType = ge::DataType::DT_FLOAT16;
        ge::DataType kvDataType = ge::DataType::DT_FLOAT16;
        ge::DataType topkIndicesDataType = ge::DataType::DT_INT32;
        ge::DataType outDataType = ge::DataType::DT_FLOAT16;
        std::vector<int64_t> actualSeqQData = {};
        std::vector<int64_t> actualSeqKVData = {};
        nsaGradParam();
        nsaGradParam(int64_t pB, int64_t pS1, int64_t pS2, int64_t pN1, int64_t pD, int64_t pD2, int64_t pN2,
                     int64_t pSelectedBlockCount, int64_t pSelectedBlockSize, float pScaleValue,
                     std::string pInputLayout, int64_t pSparseMode);
    };
    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
        gert::Tensor *actualSeqQLengthsTensor = nullptr;
        gert::Tensor *actualSeqKVLengthsTensor = nullptr;
    };

    Tensor query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices,
        actual_seq_qlen, actual_seq_kvlen, atten_mask, dq, dk, dv;
    OpInfo mOpInfo;
    Context mCtx;
    nsaGradParam mParam;
    gert::OpImplRegisterV2::TilingKernelFunc NsagTilingFunc = nullptr;
    bool isDeterministic = false;
    NsaGradCase();
    NsaGradCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, nsaGradParam param);
    bool Run() override;
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
    bool DoOpTiling(DoTilingParam &tilingParam);
    template <class T> static bool InitTensor(Tensor &tensor, std::vector<T> &hostData)
    {
        if (hostData.empty()) {
            return true;
        }
        int64_t expMinSize = hostData.size() * sizeof(T);
        if (tensor.AllocDevData(0, expMinSize) == nullptr) {
            printf("Tensor(%s, %ld) AllocDevData Failed.", tensor.Name().c_str(), expMinSize);
            return false;
        }
        return tensor.CopyHostToDevData(hostData);
    }
};

} // namespace ops::adv::tests::nsaGrad