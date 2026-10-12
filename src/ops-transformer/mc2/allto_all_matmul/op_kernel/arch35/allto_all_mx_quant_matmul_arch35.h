/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_all_mx_quant_matmul_arch35.h
 * \brief
 */

#ifndef ALLTO_ALL_MX_QUANT_MATMUL_ARCH35_H
#define ALLTO_ALL_MX_QUANT_MATMUL_ARCH35_H

#include "allto_all_matmul_tiling_data.h"

namespace AlltoAllMatmulImpl {

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
class AlltoAllMxQuantMatmulArch35 {
public:
    __aicore__ inline AlltoAllMxQuantMatmulArch35(SchedulerType *pipeLine) : pipeLine_(pipeLine){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out,
                                GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR workspaceGM,
                                AlltoAllMatmulTilingDataType *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    SchedulerType *pipeLine_;
    SchedulerContextType pipeLineContext_;
    AlltoAllMatmulTilingDataType *tilingData_;
    TPipe *tPipe_;
    GM_ADDR x1_;
    GM_ADDR x2_;
    GM_ADDR y_;
    GM_ADDR bias_;
    GM_ADDR x1Scale_;
    GM_ADDR x2Scale_;
    GM_ADDR workspaceGM_;
    GM_ADDR commX1ScaleGM1_;
    GM_ADDR commOutGM_;
    GM_ADDR transX1ScaleGM1_;
    GM_ADDR transOutGM_;
private:
    static constexpr uint64_t MXFP_GROUP_SIZE = 64UL;
 	static constexpr uint64_t NUM_TWO = 2UL;
    static constexpr uint64_t ALIGN_NUM = 512UL;
    __aicore__ inline void ProcessTile(uint32_t taskCnt);
    __aicore__ inline void ProcessTail(uint32_t taskCnt);
    __aicore__ inline void ProcessScale();
};

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
__aicore__ inline void
AlltoAllMxQuantMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out, GM_ADDR x1_scale, GM_ADDR x2_scale,
    GM_ADDR workspaceGM, AlltoAllMatmulTilingDataType *tilingData, TPipe *tPipe)
{
    // 获取tilingdata数据
    tilingData_ = tilingData;
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;
    tPipe_ = tPipe;
    x1_ = x1;
    x2_ = x2;
    y_ = y;
    bias_ = bias;
    x1Scale_ = x1_scale;
    x2Scale_ = x2_scale;
    workspaceGM_ = workspaceGM;
    // MX场景下，通信数据的存储顺序是commX1Scale、commX1、transX1Scale_、transOut
    commX1ScaleGM1_ = workspaceGM;
    uint64_t x1ScaleLen = CeilDiv(mc2Tiling_.rankK, MXFP_GROUP_SIZE) * NUM_TWO *
                 (uint64_t)mc2Tiling_.rankM * sizeof(AscendC::fp8_e8m0_t);
    x1ScaleLen = CeilDiv(x1ScaleLen, ALIGN_NUM) * ALIGN_NUM; // 对齐
    commOutGM_ = workspaceGM + x1ScaleLen;
    transX1ScaleGM1_ = commOutGM_ + mc2Tiling_.commLen;
    transOutGM_ = all2all_out;
    if (all2all_out == nullptr) {
        transOutGM_ = transX1ScaleGM1_ + x1ScaleLen;
    }

    // 初始化流水线
    pipeLine_->Init();
    // 获取流水线上下文
    pipeLine_->GetContext(&pipeLineContext_);
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
__aicore__ inline void
AlltoAllMxQuantMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType>::Process()
{
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;

    // 启动scale通信和转置
    ProcessScale();

    // 启动主块流水
    if (mc2Tiling_.tileCnt > 0) {
        ProcessTile(mc2Tiling_.tileCnt);
    }

    // 启动尾块流水
    if (mc2Tiling_.tailCnt > 0) {
        ProcessTail(mc2Tiling_.tailCnt);
    }

    // 结束流水线
    pipeLine_->End();
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
__aicore__ inline void
AlltoAllMxQuantMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType>::ProcessScale()
{
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;

    uint64_t rankKdivByGroupSize = CeilDiv(mc2Tiling_.rankK, MXFP_GROUP_SIZE) * NUM_TWO;
    // x1Scale通信相关地址
    pipeLineContext_.communicationContext->taskCnt = 1;
    pipeLineContext_.communicationContext->sendBuffer = x1Scale_;
    pipeLineContext_.communicationContext->recvBuffer = commX1ScaleGM1_;
    pipeLineContext_.communicationContext->sendCount =
        (uint64_t)mc2Tiling_.rankM / mc2Tiling_.rankDim * rankKdivByGroupSize;
    pipeLineContext_.communicationContext->strideCount = pipeLineContext_.communicationContext->sendCount;
    pipeLineContext_.communicationContext->hcclDataType = mc2Tiling_.hcclDataType;

    // x1Scale转置相关地址和偏移
    pipeLineContext_.scaleTransposeContext->transposeSrcAddr = commX1ScaleGM1_;
    pipeLineContext_.scaleTransposeContext->transposeDstAddr = transX1ScaleGM1_;
    pipeLineContext_.scaleTransposeContext->nextSrcBlockOffset =
        (uint64_t)mc2Tiling_.rankM / mc2Tiling_.rankDim * rankKdivByGroupSize;
    pipeLineContext_.scaleTransposeContext->nextDstBlockOffset = rankKdivByGroupSize;
    pipeLineContext_.scaleTransposeContext->rankCnt = (uint64_t)mc2Tiling_.rankDim;
    pipeLineContext_.scaleTransposeContext->innerAxis = rankKdivByGroupSize;
    pipeLineContext_.scaleTransposeContext->transM = (uint64_t)mc2Tiling_.rankM / mc2Tiling_.rankDim;
    pipeLineContext_.scaleTransposeContext->innerOffsetIn = rankKdivByGroupSize;
    pipeLineContext_.scaleTransposeContext->innerOffsetOut = rankKdivByGroupSize * mc2Tiling_.rankDim;
    
    // 启动x1Scale的通信和转置
    pipeLine_->ProcessScale();
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
__aicore__ inline void
AlltoAllMxQuantMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType>::ProcessTile(uint32_t taskCnt)
{
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;
    // 复用的变量
    uint64_t tileMMultiRankK = (uint64_t)mc2Tiling_.tileM * (uint64_t)mc2Tiling_.rankK;

    // 通信相关地址和偏移
    pipeLineContext_.communicationContext->taskCnt = mc2Tiling_.tileCnt;
    pipeLineContext_.communicationContext->sendBuffer = x1_;
    pipeLineContext_.communicationContext->recvBuffer = commOutGM_;
    pipeLineContext_.communicationContext->sendOffset = tileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.communicationContext->recvOffset = pipeLineContext_.communicationContext->sendOffset;
    pipeLineContext_.communicationContext->sendCount = tileMMultiRankK;
    pipeLineContext_.communicationContext->strideCount =
        (uint64_t)mc2Tiling_.rankM * (uint64_t)mc2Tiling_.rankK / (uint64_t)mc2Tiling_.rankDim;
    pipeLineContext_.communicationContext->hcclDataType = mc2Tiling_.hcclDataType;

    // 转置相关地址和偏移
    pipeLineContext_.transposeContext->transposeSrcAddr = commOutGM_;
    pipeLineContext_.transposeContext->transposeDstAddr = transOutGM_;
    pipeLineContext_.transposeContext->transposeSrcOffset = tileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->transposeDstOffset =
        tileMMultiRankK * (uint64_t)mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->nextSrcBlockOffset =
        (uint64_t)mc2Tiling_.rankM * (uint64_t)mc2Tiling_.rankK / (uint64_t)mc2Tiling_.rankDim;
    pipeLineContext_.transposeContext->nextDstBlockOffset = (uint64_t)mc2Tiling_.rankK;
    pipeLineContext_.transposeContext->rankCnt = (uint64_t)mc2Tiling_.rankDim;
    pipeLineContext_.transposeContext->innerAxis = (uint64_t)mc2Tiling_.rankK;
    pipeLineContext_.transposeContext->transM = (uint64_t)mc2Tiling_.tileM;
    pipeLineContext_.transposeContext->innerOffsetIn = (uint64_t)mc2Tiling_.rankK;
    pipeLineContext_.transposeContext->innerOffsetOut = (uint64_t)mc2Tiling_.rankK * mc2Tiling_.rankDim;

    // quantMatmul相关的地址和偏移
    pipeLineContext_.computationContext->baseData.aGM = transOutGM_;
    pipeLineContext_.computationContext->baseData.bGM = x2_;
    pipeLineContext_.computationContext->baseData.cGM = y_;
    pipeLineContext_.computationContext->baseData.biasGM = bias_;
    pipeLineContext_.computationContext->baseData.aOffset =
        tileMMultiRankK * (uint64_t)mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.computationContext->baseData.bOffset = 0UL;
    pipeLineContext_.computationContext->baseData.cOffset =
        (uint64_t)mc2Tiling_.tileM * (uint64_t)mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.computationContext->additionalData.x1Scale = transX1ScaleGM1_;
    pipeLineContext_.computationContext->additionalData.x1ScaleOffset =
        (uint64_t)mc2Tiling_.tileM * CeilDiv(mc2Tiling_.rankK * (uint64_t)mc2Tiling_.rankDim, MXFP_GROUP_SIZE) *
        NUM_TWO * sizeof(AscendC::fp8_e8m0_t);
    pipeLineContext_.computationContext->additionalData.x2Scale = x2Scale_;
    pipeLineContext_.computationContext->tilingDataPtr = &(tilingData_->mc2QuantMmTileTilingData);

    pipeLine_->Process(taskCnt);
}

template <typename SchedulerType, typename SchedulerContextType, typename AlltoAllMatmulTilingDataType>
__aicore__ inline void
AlltoAllMxQuantMatmulArch35<SchedulerType, SchedulerContextType, AlltoAllMatmulTilingDataType>::ProcessTail(uint32_t taskCnt)
{
    auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;
    uint64_t tailMMultiRankK = (uint64_t)mc2Tiling_.tailM * (uint64_t)mc2Tiling_.rankK;
    uint64_t tileCntMultitileMMultiRankK =
        (uint64_t)mc2Tiling_.tileCnt * (uint64_t)mc2Tiling_.tileM * (uint64_t)mc2Tiling_.rankK;

    // 通信相关地址和偏移
    pipeLineContext_.communicationContext->taskCnt = mc2Tiling_.tailCnt;
    pipeLineContext_.communicationContext->sendBuffer = x1_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.communicationContext->recvBuffer = commOutGM_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.communicationContext->sendOffset = tailMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.communicationContext->recvOffset = pipeLineContext_.communicationContext->sendOffset;
    pipeLineContext_.communicationContext->sendCount = tailMMultiRankK;

    pipeLineContext_.transposeContext->transposeSrcAddr = commOutGM_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->transposeDstAddr =
        transOutGM_ + tileCntMultitileMMultiRankK * (uint64_t)mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->transposeSrcOffset = tailMMultiRankK * sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->transposeDstOffset = tailMMultiRankK * (uint64_t)mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.transposeContext->transM = (uint64_t)mc2Tiling_.tailM;

    pipeLineContext_.computationContext->baseData.aGM =
        transOutGM_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1) * mc2Tiling_.rankDim;
    pipeLineContext_.computationContext->baseData.bGM = x2_;
    pipeLineContext_.computationContext->baseData.cGM =
        y_ + (uint64_t)mc2Tiling_.tileCnt * mc2Tiling_.tileM * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.computationContext->baseData.aOffset =
        tailMMultiRankK * (uint64_t)mc2Tiling_.rankDim * sizeof(DTYPE_X1);
    pipeLineContext_.computationContext->baseData.bOffset = 0UL;
    pipeLineContext_.computationContext->baseData.cOffset =
        (uint64_t)mc2Tiling_.tailM * mc2Tiling_.rankN * (uint64_t)sizeof(DTYPE_Y);
    pipeLineContext_.computationContext->additionalData.x1Scale =
        transX1ScaleGM1_ + mc2Tiling_.tileCnt * mc2Tiling_.tileM * 
        CeilDiv(mc2Tiling_.rankK * (uint64_t)mc2Tiling_.rankDim, MXFP_GROUP_SIZE) * NUM_TWO * sizeof(AscendC::fp8_e8m0_t);
    pipeLineContext_.computationContext->additionalData.x1ScaleOffset =
        (uint64_t)mc2Tiling_.tailM * CeilDiv(mc2Tiling_.rankK * (uint64_t)mc2Tiling_.rankDim, MXFP_GROUP_SIZE) *
        NUM_TWO * sizeof(AscendC::fp8_e8m0_t);
    pipeLineContext_.computationContext->additionalData.x2Scale = x2Scale_;
    pipeLineContext_.computationContext->tilingDataPtr = &(tilingData_->mc2QuantMmTailTilingData);

    pipeLine_->Process(taskCnt);
}

} // namespace AlltoAllMatmulImpl

#endif