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
 * \file ts_nsa_compress_grad_tc.cpp
 * \brief NsaCompressGrad用例.
 */

#include "ts_nsa_compress_grad.h"

NsaCompressGradCase InitNormalCase(int64_t headNum, int64_t headDim, int64_t blockSize, int64_t blockStride,
                                   int64_t blockNum, int64_t seqLensSum, int64_t batchSize, int64_t seqLenType,
                                   std::string layout, std::vector<int64_t> actSeqLens, ge::DataType optionalDataType, 
                                   ge::DataType actSeqLenOptionalDataType, ge::graphStatus result, int64_t tilingKey)
{
    NsaCompressGradCase cs;
    cs.mParam = {headNum, headDim, blockSize, blockStride, blockNum, seqLensSum, batchSize, seqLenType, layout,
                 actSeqLens, optionalDataType, actSeqLenOptionalDataType};
    if (result == ge::GRAPH_SUCCESS) {
        cs.mOpInfo.mExp.mTilingKey = tilingKey;
        cs.mOpInfo.mCtr.mRunTiling = true;
        cs.mOpInfo.mCtr.mRunKernel = true;
    } else {
        cs.mOpInfo.mExp.mSuccess = false;
        cs.mOpInfo.mCtr.mRunTiling = true;
        cs.mOpInfo.mCtr.mRunKernel = false;
    }
    cs.Init();
    return cs;
}

void InitAndRunNormalCase(int64_t headNum, int64_t headDim, int64_t blockSize, int64_t blockStride,
                          int64_t blockNum, int64_t seqLensSum, int64_t batchSize, int64_t seqLenType,
                          std::string layout, std::vector<int64_t> actSeqLens, ge::DataType optionalDataType,
                          ge::DataType actSeqLenOptionalDataType, ge::graphStatus result, int64_t tilingKey)
{
    NsaCompressGradCase cs = InitNormalCase(headNum, headDim, blockSize, blockStride, blockNum, seqLensSum, batchSize,
                                            seqLenType, layout, actSeqLens, optionalDataType, actSeqLenOptionalDataType,
                                            result, tilingKey);
    if (result == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

TEST_F(Ts_NsaCompressGrad, nsa_compress_grad_base)
{   
    InitAndRunNormalCase(128, 192, 32, 16, 13, 272, 4, 0, "TND", {0, 120, 140, 272}, ge::DT_FLOAT16, ge::DT_INT64,
                         ge::GRAPH_SUCCESS, 0);
}
