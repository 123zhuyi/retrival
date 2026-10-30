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
 * \file nsa_compress_grad_case.h
 * \brief NsaCompressGrad 测试用例.
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
#include "tests/utils/log.h"

namespace ops::adv::tests::NsaCompressGrad {
class NsaCompressGradCase : public ops::adv::tests::utils::Case {
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using Tensor = ops::adv::tests::utils::Tensor;
    using TensorList = ops::adv::tests::utils::TensorList;

public:
    class Param {
    public:
        int64_t headNum = 0;
        int64_t headDim = 0;
        int64_t blockSize = 0;
        int64_t blockStride = 0;
        int64_t blockNum = 0;
        int64_t seqLensSum = 0;
        int64_t batchSize = 0;
        int64_t seqLenType = 0;
        std::string layout = "TND";
        std::vector<int64_t> actSeqLens = {};
        ge::DataType optionalDataType = ge::DataType::DT_FLOAT16;
        ge::DataType actSeqLenOptionalDataType = ge::DataType::DT_INT64;
        std::vector<int64_t> actSeqLensTensorData = {};
        
        Param();
        Param(int64_t pHeadNum, int64_t pHeadDim, int64_t pBlockSize, int64_t pBlockStride, int64_t pBlockNum,
              int64_t pSeqLensSum, int64_t pBatchSize, int64_t pSeqLenType, std::string pLayout, 
              std::vector<int64_t> pActSeqLens, ge::DataType optionalDataType, ge::DataType actSeqLenOptionalDataType);
    };
    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
    };
    OpInfo mOpInfo;
    Context mCtx;
    Param mParam;

    
    Tensor outputGrad, inputKV, weight, actSeqLenOptional, blockSize, blockStride, SeqLenType, layout,
           inputGradOut, weightGradOut;
    gert::OpImplRegisterV2::TilingKernelFunc nsaCompressGradTilingFunc = nullptr;

    NsaCompressGradCase();
    NsaCompressGradCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param);
    
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
            LOG_ERR("Tensor(%s, %ld) AllocDevData Failed.", tensor.Name().c_str(), expMinSize);
            return false;
        }
        return tensor.CopyHostToDevData(hostData);
    }
};

} // namespace ops::adv::tests::NsaCompressGrad
