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
 * \file nsa_compress_attention_case.h
 * \brief NsaCompressAttention 测试用例.
 */

#pragma once
#include <vector>
#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>
#include "graph/types.h"
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "tests/utils/tensor.h"
#include "tests/utils/tensor_list.h"

namespace ops::adv::tests::NsaCompressAttention {

class NsaCompressAttentionCase : public ops::adv::tests::utils::Case {
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using Tensor = ops::adv::tests::utils::Tensor;
    using TensorList = ops::adv::tests::utils::TensorList;

public:
    enum class AttenMaskShapeType {
        NONE,
        S1_S2,
        _1_1_S1_S2,
        B_1_S1_S2,
        B_N1_S1_S2,
        SPARSE,
        PREFIXCOMPRESS,
    };

    enum class TopkMaskShapeType {
        NONE,
        S1_S2,
        _1_1_S1_S2,
        B_1_S1_S2,
        B_N1_S1_S2,
        SPARSE,
        PREFIXCOMPRESS,
    };

    enum class LayoutType {
        BSH,
        SBH,
        BNSD,
        BSND,
        TND,
    };

    class Param {
    public:
        /* 设置参数 */
        int64_t b = 0;
        int64_t n2 = 0;
        int64_t g = 0;
        int64_t s1 = 0;
        int64_t s2 = 0;
        int64_t d1 = 0;
        int64_t d2 = 0;
        ge::DataType dtype = ge::DataType::DT_UNDEFINED;
        LayoutType layoutType = LayoutType::TND;
        float scale = 1.0f;
        int64_t sparseMode = 0;
        int64_t cmpBlkSize = 0;
        int64_t cmpStride = 0;
        int64_t selBlkSize = 0;
        int64_t selBlkCnt = 0;
        AttenMaskShapeType attenMaskShapeType = AttenMaskShapeType::NONE;
        TopkMaskShapeType topkMaskShapeType = TopkMaskShapeType::NONE;
        ge::DataType attenMaskDtype = ge::DataType::DT_UNDEFINED;
        ge::DataType topkMaskDtype = ge::DataType::DT_UNDEFINED;
        std::vector<int64_t> actualSeqQLenList = {};
        std::vector<int64_t> actualCmpSeqKvLenList = {};
        std::vector<int64_t> actualSelSeqKvLenList = {};

        /* 生成参数 */
        int64_t t1 = 0;
        int64_t t2 = 0;
        std::string layout;
        std::vector<int64_t> actualSeqQLenTensorData = {};
        std::vector<int64_t> actualCmpSeqKVLenTensorData = {};
        std::vector<int64_t> actualSelSeqKVLenTensorData = {};

    public:
        /**
        * @param pActualSeqQLenList 传入实际 Seq 长度, 内部计算 T1 值
        * @param pActualSeqKvLenList 传入实际 Seq 长度, 内部计算 T2 值
        */
        Param() = default;
        Param(int64_t pB, int64_t pN2, int64_t pG, int64_t pS1, int64_t pS2, int64_t pD1, int64_t pD2, ge::DataType pDtype, 
              LayoutType pLayoutType, float pScale, int64_t pSparseMode, int64_t pCmpBlkSize, int64_t pCmpStride, 
              int64_t pSelBlkSize, int64_t pSelBlkCnt, AttenMaskShapeType pAttenMaskShapeType, 
              TopkMaskShapeType pTopkMaskShapeType, ge::DataType pAttenMaskDtype, ge::DataType pTopkMaskDtype,
              std::vector<int64_t> pActualSeqQLenList, std::vector<int64_t> pActualCmpSeqKvLenList, 
              std::vector<int64_t> pActualSelSeqKvLenList);
    };
    
    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
        gert::Tensor *actSeqQLenTensor = nullptr;
        gert::Tensor *actSeqCmpKVLenTensor = nullptr;
        gert::Tensor *actSeqSelKVLenTensor = nullptr;
    };
    Tensor query;
    Tensor key;
    Tensor value;
    Tensor attenMask;
    Tensor actualSeqQLen;
    Tensor actualCmpSeqKvLen;
    Tensor actualSelSeqKvLen;
    Tensor topkMask;
    Tensor softmaxMax;
    Tensor softmaxSum;
    Tensor attenOut;
    Tensor topkIndicesOut;

    OpInfo mOpInfo;
    Context mCtx;
    Param mParam;
    gert::OpImplRegisterV2::TilingKernelFunc nsaCompressAttentionTilingFunc = nullptr;

    NsaCompressAttentionCase();
    NsaCompressAttentionCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param);
    bool Run() override;
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
    bool DoOpTiling(DoTilingParam &tilingParam);
    bool InitQkvAndOut();
    bool InitOptInputs();
    static bool InitTensor(Tensor &tensor, std::vector<int64_t> &hostData)
    {
        if (hostData.empty()) {
            return true;
        }
        int64_t expMinSize = hostData.size() * sizeof(int64_t);
        if (tensor.AllocDevData(0, expMinSize) == nullptr) {
            return false;
        }
        return tensor.CopyHostToDevData(hostData);
    }
};

} // namespace ops::adv::tests::MoeInitRoutingV2
