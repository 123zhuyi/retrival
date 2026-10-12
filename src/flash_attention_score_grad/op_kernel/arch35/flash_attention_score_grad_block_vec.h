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
 * \file flash_attention_score_grad_block_vec.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_BLOCK_VEC_H
#define FLASH_ATTENTION_SCORE_GRAD_BLOCK_VEC_H
 
#include "flash_attention_score_grad_common.h"
#include "vector_api/cast_softmax_grad.h"
#include "vector_api/dropout.h"
#include "vector_api/pse_atten_mask_muls_simple_softmax.h"
#include "vector_api/vf_broadcast_sub_mul.h"
#include "vector_api/vf_cast_transdata_deconflict.h"
namespace FagBaseApi {
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t SYNC_V0_V1_DS_A_MAX_DONE_FLAG = 10;
constexpr uint32_t BIT_MASK_NUM = 8;

TEMPLATES_DEF
class FAGBlockVec {
private:
    template <uint8_t MM_IDX>
    __aicore__ inline void DqkvMulsAndCastFromUB(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                                 LocalTensor<CALC_TYPE> &inputTensor,
                                                 TQue<QuePosition::VECOUT, 1> &outQue);
    template <uint8_t MM_IDX>
    __aicore__ inline void
    DqkvMulsAndCastFromGM(FagConstInfo &constInfo, FagRunInfo &runInfo, GlobalTensor<CALC_TYPE> &inputTensor,
                          TQue<QuePosition::VECIN, 1> &inQue, TQue<QuePosition::VECOUT, 1> &outQue);
 
public:
    __aicore__ inline FAGBlockVec(){};
    __aicore__ inline void SetVecBlockParams(TPipe *pipe, FagTilingType tilingData, uint32_t vBlockIdx,
                                             uint32_t cBlockIdx, uint32_t vSubBlockIdx, AttenMaskInfo &attenMaskInfo,
                                             PseInfo &pseInfo, DropMaskInfo &dropInfo);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR value, GM_ADDR dy, GM_ADDR y, GM_ADDR pseShift, GM_ADDR dropMask,
                                            GM_ADDR attenMask, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                            GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy,
                                            GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace);
    __aicore__ inline void InitUbBuffer();
    __aicore__ inline void InitCubeVecSharedParams(FagCVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, float qScaleDs);
    __aicore__ inline void ProcessVec1(FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void ProcessVec2(LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo);
    __aicore__ inline void ProcessVec3(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<CALC_TYPE> &mm1ResTensor,
                                       LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo);
    __aicore__ inline void ProcessVec3Quant(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                            Buffer<BufferType::L1, SyncType::NO_SYNC> &dstTransBuffer,
                                            LocalTensor<CALC_TYPE> &mm1ResTensor,
                                            LocalTensor<CALC_TYPE> &mm2ResTensor,
                                            FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void ProcessVec4(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                       FagConstInfo &constInfo, FagRunInfo &runInfo);
 
    template <typename T, bool IS_WRITE_UB, uint8_t MM_IDX>
    __aicore__ inline void ProcessMulsAndCast(typename DqkvResPos<T, IS_WRITE_UB>::PosType inputTensor,
                                              FagConstInfo &constInfo, FagRunInfo &runInfo);
 
    __aicore__ inline void CopyMaxSum(FagConstInfo &constInfo, FagRunInfo &runInfo, int64_t taskId);
    template <const bool IS_DQ = false>
    __aicore__ inline void CopyUB2L1(FagRunInfo &runInfo, LocalTensor<INPUT_TYPE> &dstTensor,
                                     LocalTensor<INPUT_TYPE> &srcTensor);
    template <const bool IS_MM1 = false>
    __aicore__ inline void CopyUB2L1Quant(FagRunInfo &runInfo, LocalTensor<OUTDTYPE> &dstTensor, LocalTensor<OUTDTYPE> &srcTensor,
                                          uint32_t loopNum = 0, uint32_t loopIdx = 0, uint32_t loopSize = 0, uint32_t curLoopSize = 0);
    __aicore__ inline void CopyUB2L1Deter(FagRunInfo &runInfo, LocalTensor<INPUT_TYPE> &dstTensor,
                                          LocalTensor<INPUT_TYPE> &srcTensor);
    template <const bool IS_DK>
    __aicore__ inline void ProcessPostDeter(FagConstInfo &constInfo, GlobalTensor<float> dkvWorkSpaceTensor, GlobalTensor<INPUT_TYPE> &dkvGmTensor,
        int64_t specialHalfS2RealSize, int64_t specialFirstHalfS2RealSize, uint64_t dAlign16, uint64_t dvAlign16, int64_t specialDkGmOffset, int64_t specialDvGmOffset);
    __aicore__ inline void DequantAndCopy2L1(Buffer<BufferType::L1, SyncType::NO_SYNC> &vL1Buffer, LocalTensor<CALC_TYPE> mm1ResTensor,
                                            FagConstInfo &constInfo, FagRunInfo &runInfo);

    constexpr static bool IS_FP8_INPUT =
        IsSameType<INPUT_TYPE, fp8_e5m2_t>::value || IsSameType<INPUT_TYPE, fp8_e4m3fn_t>::value || IsSameType<INPUT_TYPE, hifloat8_t>::value;
    constexpr static bool IS_FP32_INPUT = IsSameType<INPUT_TYPE, float>::value;
    constexpr static float FP8_MAX = IsSameType<INPUT_TYPE, fp8_e5m2_t>::value ? 57344 : (IsSameType<INPUT_TYPE, fp8_e4m3fn_t>::value ? 448 : 32768);
    constexpr static uint32_t DETER_OFFSET_UB_SIZE = 1024 * 3;
    constexpr static uint32_t CUBE_BASEM = (uint32_t)s1TemplateType;
    constexpr static uint32_t CUBE_BASEN = (uint32_t)s2TemplateType;
    constexpr static uint32_t HEAD_DIM_ALIGN = (uint32_t)dTemplateType;
    constexpr static uint32_t VECTOR_BASEM = CUBE_BASEM / CV_CORE_RATIO;
    constexpr static uint32_t VECTOR_BASEN = CUBE_BASEN;
    constexpr static uint32_t INPUT_BLOCK_NUM_FOR_INPUT_DTYPE = 32 / sizeof(INPUT_TYPE);
    constexpr static uint32_t FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE = 32 / sizeof(INPUT_TYPE);
    constexpr static uint32_t INPUT_BLOCK_NUM_FOR_OUT_DTYPE = 32 / sizeof(OUTDTYPE);
    constexpr static uint32_t FRACTAL_NZ_C0_SIZE_FOR_OUT_DTYPE = 32 / sizeof(OUTDTYPE);
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP16 = 32 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP32_D256 = 16 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP32_D512 = 64 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE =
        IS_FP32_INPUT ? (HEAD_DIM_ALIGN > 256 ? DETER_DQ_UB_SIZE_FP32_D512 : DETER_DQ_UB_SIZE_FP32_D256) :
                        DETER_DQ_UB_SIZE_FP16;
                        
    // vector gm addr
    GlobalTensor<INPUT_TYPE> valueGm;
    GlobalTensor<OUTDTYPE> yGm, pseGm, dyGm;
    GlobalTensor<uint8_t> dropMaskGm, attenMaskU8Gm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm, pseFloatGm;
    GlobalTensor<float> deqScaleQGm, deqScaleKGm, deqScaleVGm, deqScaleDyGm;
    GM_ADDR pseSlope;
    GlobalTensor<uint8_t> dropMaskWorkspaceGm;
    GlobalTensor<float> dsAmaxWorkSpaceGm;
    GlobalTensor<OUTDTYPE> dqGm, dkGm, dvGm;
 
    // ub buffer
    TQue<QuePosition::VECIN, 1> attenMaskOrYInQue;
    TQue<QuePosition::VECIN, 1> pseOrDyInQue;
    TQue<QuePosition::VECOUT, 1> dSOutQue;
    TQue<QuePosition::VECOUT, 1> pOutQue;
    TQue<QuePosition::VECIN, 1> maxSumQue[2];
    TBuf<> softmaxGradResBuf;
    TBuf<> dropMaskBuf;
    TBuf<> dropmaskIndexVecBuf;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> deterInOutQue;
    TBuf<> deterOffsetBuf;
    TBuf<> vselrIndexesBuf;
    TQue<QuePosition::VECOUT, 1> dsAmaxOutQue;
 
    TPipe *pipe;
    FagTilingType tilingData;
 
    uint32_t vBlockIdx;
    uint32_t cBlockIdx;
    uint32_t vSubBlockIdx;
 
    // optional info
    AttenMaskInfo *attenMaskInfoPtr;
    PseInfo *pseInfoPtr;
    DropMaskInfo *dropInfoPtr;
};
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::SetVecBlockParams(TPipe *pipe, FagTilingType tilingData,
                                                                     uint32_t vBlockIdx, uint32_t cBlockIdx,
                                                                     uint32_t vSubBlockIdx,
                                                                     AttenMaskInfo &attenMaskInfo, PseInfo &pseInfo,
                                                                     DropMaskInfo &dropInfo)
{
    this->pipe = pipe;
    this->tilingData = tilingData;
    this->vBlockIdx = vBlockIdx;
    this->cBlockIdx = cBlockIdx;
    this->vSubBlockIdx = vSubBlockIdx;
    attenMaskInfoPtr = &attenMaskInfo;
    pseInfoPtr = &pseInfo;
    dropInfoPtr = &dropInfo;
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(GM_ADDR value, GM_ADDR dy, GM_ADDR y, GM_ADDR pseShift,
                                                                    GM_ADDR dropMask, GM_ADDR attenMask,
                                                                    GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                                                    GM_ADDR deqScaleQ, GM_ADDR deqScaleK,
                                                                    GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR dq,
                                                                    GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace)
{
    valueGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)value);
    dyGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dy);
    yGm.SetGlobalBuffer((__gm__ OUTDTYPE *)y);
    pseGm.SetGlobalBuffer((__gm__ OUTDTYPE *)pseShift);
    pseFloatGm.SetGlobalBuffer((__gm__ float *)pseShift);
    dropMaskGm.SetGlobalBuffer((GM_ADDR)dropMask);
    attenMaskU8Gm.SetGlobalBuffer((GM_ADDR)attenMask);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    deqScaleQGm.SetGlobalBuffer((__gm__ float *)deqScaleQ);
    deqScaleKGm.SetGlobalBuffer((__gm__ float *)deqScaleK);
    deqScaleVGm.SetGlobalBuffer((__gm__ float *)deqScaleV);
    pseSlope = pseShift;
    dqGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
    dkGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dk);
    dvGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dv);
    if constexpr (IS_FP8_INPUT) {
        dsAmaxWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->postTilingData.vScaleDsWorkSpaceOffset / sizeof(float));
    }
    if constexpr (IS_DROP) {
        if (tilingData->preTilingData.dropoutIsDivisibleBy8 == 0) {
            dropMaskWorkspaceGm.SetGlobalBuffer((__gm__ uint8_t *)workspace + tilingData->postTilingData.dropMaskGmOffset);
        }
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::InitUbBuffer()
{
    /**
     * UB划分，buffer大小分配
     * attenMaskOrYInQue: for y and attenMask
     * pseOrDyInQue: for dy and pse
     * dSOutQue: for dq dk left ub matrix
     * pOutQue: for dv left ub matrix
     * softmaxGradResBuf: for softmax_grad result
     * dropMaskBuf: for dropMask
     * maxSumQue: for max sum double buffer
     **/
    pipe->InitBuffer(attenMaskOrYInQue, 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE));
    pipe->InitBuffer(pseOrDyInQue, 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(OUTDTYPE));
    pipe->InitBuffer(softmaxGradResBuf, VECTOR_BASEM * sizeof(CALC_TYPE));
    pipe->InitBuffer(maxSumQue[0], 1, VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * NUM_TWO);
    pipe->InitBuffer(maxSumQue[1], 1, VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * NUM_TWO);
    if constexpr (IS_DROP) {
        pipe->InitBuffer(dropMaskBuf, VECTOR_BASEM * VECTOR_BASEN * sizeof(uint8_t) / BIT_MASK_NUM);          // 1k
        pipe->InitBuffer(dropmaskIndexVecBuf, VECTOR_BASEM * VECTOR_BASEN / (NUM_TWO * BIT_MASK_NUM) * sizeof(int32_t)); // 2k
    }
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        pipe->InitBuffer(deterInOutQue, 1, DETER_DQ_UB_SIZE);
        pipe->InitBuffer(deterOffsetBuf, DETER_OFFSET_UB_SIZE); // 3k
    }
    if constexpr (IS_FP8_INPUT) {
        pipe->InitBuffer(vselrIndexesBuf, VECTOR_BASEN);
        LocalTensor<uint8_t> selrIndexesTensor = vselrIndexesBuf.Get<uint8_t>();
        for (int i = 0; i < VECTOR_BASEN; i++) {
            selrIndexesTensor.SetValue(i, i * NUM_TWO);
        }
        pipe->InitBuffer(dsAmaxOutQue, 1, VREG_SIZE / NUM_TWO);
    }
    if constexpr (!IS_FP32_INPUT) {
        pipe->InitBuffer(dSOutQue, 1, (VECTOR_BASEM + 1) * VECTOR_BASEN * sizeof(OUTDTYPE));
        pipe->InitBuffer(pOutQue, 1, (VECTOR_BASEM + 1) * VECTOR_BASEN * sizeof(OUTDTYPE));
    } else {
        // input type fp32, exceed ub size so need to reuse dSOutQue
        pipe->InitBuffer(dSOutQue, 1,
                         VECTOR_BASEM * VECTOR_BASEN * sizeof(INPUT_TYPE) + VECTOR_BASEN * sizeof(INPUT_TYPE));
        pOutQue = dSOutQue;
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec1(FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF1: Cast + SoftmaxGradFront
    ///////////////////////////////////////////////////////////////
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    LocalTensor<CALC_TYPE> softmaxGradResTensor = softmaxGradResBuf.Get<CALC_TYPE>();
    if constexpr (HEAD_DIM_ALIGN <= VECTOR_BASEN) {
        CopyInSoftmaxGrad<OUTDTYPE, CALC_TYPE, VECTOR_BASEM, HEAD_DIM_ALIGN, IS_D_NO_EQUAL>(
            constInfo, runInfo, 0, runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.halfS1RealSize,
            attenMaskOrYInQue, pseOrDyInQue, dyGm, yGm);
        CalculateCastSoftmaxGrad<OUTDTYPE, CALC_TYPE, VECTOR_BASEM, HEAD_DIM_ALIGN>(
            constInfo, runInfo.commonRunInfo.halfS1RealSize, attenMaskOrYInQue, pseOrDyInQue, softmaxGradResTensor);
    } else {
        uint32_t loopNum = Ceil<uint32_t>(runInfo.commonRunInfo.halfS1RealSize, constInfo.sfmgMaxLoopSize);
        uint32_t loopSize = Ceil<uint32_t>(runInfo.commonRunInfo.halfS1RealSize, loopNum);
        uint32_t tailLoopSize = runInfo.commonRunInfo.halfS1RealSize - (loopNum - 1) * loopSize;
        uint32_t curLoopSize = loopSize;
        for (int32_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
            if (loopIdx == loopNum - 1) {
                curLoopSize = tailLoopSize;
            }
            CopyInSoftmaxGrad<OUTDTYPE, CALC_TYPE, VECTOR_BASEM, HEAD_DIM_ALIGN, IS_D_NO_EQUAL>(
                constInfo, runInfo, loopIdx, curLoopSize, loopSize, attenMaskOrYInQue, pseOrDyInQue, dyGm, yGm);
            CalculateCastSoftmaxGrad<OUTDTYPE, CALC_TYPE, VECTOR_BASEM, HEAD_DIM_ALIGN>(
                constInfo, curLoopSize, attenMaskOrYInQue, pseOrDyInQue, softmaxGradResTensor[loopSize * loopIdx]);
        }
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopyMaxSum(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                                              int64_t taskId)
{
    CopyInMaxSum<float, VECTOR_BASEM>(constInfo, runInfo, maxSumQue[taskId & 1], softmaxMaxGm, softmaxSumGm);
}
 
TEMPLATES_DEF_NO_DEFAULT
template <const bool IS_DQ>
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopyUB2L1(FagRunInfo &runInfo, LocalTensor<INPUT_TYPE> &dstTensor,
                                                             LocalTensor<INPUT_TYPE> &srcTensor)
{
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    uint32_t scmOffset = vSubBlockIdx == 0 ? 0 : runInfo.commonRunInfo.firstHalfS1RealSize * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = VECTOR_BASEN / FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE;
    dataCopyParams.blockLen = (uint16_t)(runInfo.commonRunInfo.halfS1RealSize * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE / INPUT_BLOCK_NUM_FOR_INPUT_DTYPE);
    dataCopyParams.srcStride =
        (uint16_t)((VECTOR_BASEM + 1 - runInfo.commonRunInfo.halfS1RealSize) * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE / INPUT_BLOCK_NUM_FOR_INPUT_DTYPE);
    if constexpr (IS_FP8_INPUT) {
        uint32_t s1RealSizeAlign = 0;
        if (IS_DQ) {
            s1RealSizeAlign = AlignTo32(runInfo.commonRunInfo.s1RealSize);
        } else {
            s1RealSizeAlign = AlignTo64(runInfo.commonRunInfo.s1RealSize);            
        }
        dataCopyParams.dstStride =
            (s1RealSizeAlign - runInfo.commonRunInfo.halfS1RealSize) * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE / INPUT_BLOCK_NUM_FOR_INPUT_DTYPE;
    } else {
        uint32_t s1RealSizeAlignTo16 = AlignTo16(runInfo.commonRunInfo.s1RealSize);
        dataCopyParams.dstStride =
            (s1RealSizeAlignTo16 - runInfo.commonRunInfo.halfS1RealSize) * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE / INPUT_BLOCK_NUM_FOR_INPUT_DTYPE;
    }
    DataCopy(dstTensor[scmOffset], srcTensor, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <const bool IS_MM1>
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopyUB2L1Quant(FagRunInfo &runInfo, LocalTensor<OUTDTYPE> &dstTensor, LocalTensor<OUTDTYPE> &srcTensor,
                                                                  uint32_t loopNum, uint32_t loopIdx, uint32_t loopSize, uint32_t curLoopSize)
{
    if constexpr (IS_MM1) {
        // shape = S2 * D
        if (runInfo.halfS2RealSize == 0) {
            return;
        }
        uint32_t scmOffset = vSubBlockIdx == 0 ? 0 : runInfo.firstHalfS2RealSize * FRACTAL_NZ_C0_SIZE_FOR_OUT_DTYPE;
        scmOffset += loopIdx * loopSize * FRACTAL_NZ_C0_SIZE_FOR_OUT_DTYPE;
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = HEAD_DIM_ALIGN / FRACTAL_NZ_C0_SIZE_FOR_OUT_DTYPE;
        dataCopyParams.blockLen = (uint16_t)(curLoopSize * FRACTAL_NZ_C0_SIZE_FOR_OUT_DTYPE / INPUT_BLOCK_NUM_FOR_OUT_DTYPE);
        dataCopyParams.srcStride = 1;
        uint32_t s2RealSizeAlignTo16 = AlignTo16(runInfo.commonRunInfo.s2RealSize);
        dataCopyParams.dstStride =
                (s2RealSizeAlignTo16 - curLoopSize) * FRACTAL_NZ_C0_SIZE_FOR_OUT_DTYPE / INPUT_BLOCK_NUM_FOR_OUT_DTYPE; 
        DataCopy(dstTensor[scmOffset], srcTensor, dataCopyParams);
    } else {
        // shape = S1 * S2
        if (runInfo.commonRunInfo.halfS1RealSize == 0) {
            return;
        }
        uint32_t scmOffset = vSubBlockIdx == 0 ? 0 : runInfo.commonRunInfo.firstHalfS1RealSize * FRACTAL_NZ_C0_SIZE_FOR_OUT_DTYPE;
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = VECTOR_BASEN / FRACTAL_NZ_C0_SIZE_FOR_OUT_DTYPE;
        dataCopyParams.blockLen = (uint16_t)(runInfo.commonRunInfo.halfS1RealSize * FRACTAL_NZ_C0_SIZE_FOR_OUT_DTYPE / INPUT_BLOCK_NUM_FOR_OUT_DTYPE);
        dataCopyParams.srcStride =
            (uint16_t)((VECTOR_BASEM + 1 - runInfo.commonRunInfo.halfS1RealSize) * FRACTAL_NZ_C0_SIZE_FOR_OUT_DTYPE / INPUT_BLOCK_NUM_FOR_OUT_DTYPE);
        uint32_t s1RealSizeAlignTo16 = AlignTo16(runInfo.commonRunInfo.s1RealSize);
        dataCopyParams.dstStride =
            (s1RealSizeAlignTo16 - runInfo.commonRunInfo.halfS1RealSize) * FRACTAL_NZ_C0_SIZE_FOR_OUT_DTYPE / INPUT_BLOCK_NUM_FOR_OUT_DTYPE;
        DataCopy(dstTensor[scmOffset], srcTensor, dataCopyParams);
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopyUB2L1Deter(FagRunInfo &runInfo,
                                                                  LocalTensor<INPUT_TYPE> &dstTensor,
                                                                  LocalTensor<INPUT_TYPE> &srcTensor)
{
    uint32_t scmOffset = (vSubBlockIdx == 0 ? 0 : runInfo.commonRunInfo.firstHalfS1RealSize * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE);
    DataCopyParams dataCopyParams;
    if (runInfo.commonRunInfo.halfS1RealSize != 0) {
        dataCopyParams.blockCount = VECTOR_BASEN / FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE;
        dataCopyParams.blockLen =
            (uint16_t)(runInfo.commonRunInfo.halfS1RealSize * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE / INPUT_BLOCK_NUM_FOR_INPUT_DTYPE);
        dataCopyParams.srcStride = (uint16_t)((VECTOR_BASEM - runInfo.commonRunInfo.halfS1RealSize + 1) *
                                              FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE / INPUT_BLOCK_NUM_FOR_INPUT_DTYPE);
        dataCopyParams.dstStride =
            (uint16_t)(CUBE_BASEM - runInfo.commonRunInfo.halfS1RealSize) * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE / INPUT_BLOCK_NUM_FOR_INPUT_DTYPE;
        DataCopy(dstTensor[scmOffset], srcTensor, dataCopyParams);
    }
    if (runInfo.commonRunInfo.halfS1RealSize != VECTOR_BASEM) {
        // copy 补零的数据
        scmOffset =
            (vSubBlockIdx == 0 ? runInfo.commonRunInfo.s1RealSize * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE :
                                    (VECTOR_BASEM + runInfo.commonRunInfo.halfS1RealSize) * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE);
        dataCopyParams.blockCount = VECTOR_BASEN / FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE;
        dataCopyParams.blockLen = (uint16_t)((VECTOR_BASEM - runInfo.commonRunInfo.halfS1RealSize) *
                                                FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE / INPUT_BLOCK_NUM_FOR_INPUT_DTYPE);
        dataCopyParams.srcStride =
            (uint16_t)((runInfo.commonRunInfo.halfS1RealSize + 1) * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE / INPUT_BLOCK_NUM_FOR_INPUT_DTYPE);
        dataCopyParams.dstStride =
            (uint16_t)(VECTOR_BASEM + runInfo.commonRunInfo.halfS1RealSize) * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE / INPUT_BLOCK_NUM_FOR_INPUT_DTYPE;
        DataCopy(dstTensor[scmOffset], srcTensor[runInfo.commonRunInfo.halfS1RealSize * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE],
                    dataCopyParams);
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec2(LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                               FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF2: pse + attenMask + muls + simpleSoftmax copyIn+calculate
    ///////////////////////////////////////////////////////////////
    if constexpr (IS_FP8_INPUT) {
        Muls(mm2ResTensor, mm2ResTensor, runInfo.quantScaleInfo.deqScaleQValue * runInfo.quantScaleInfo.deqScaleKValue,
             VECTOR_BASEM * VECTOR_BASEN);
    }
    CopyInAttenMask<IS_ATTEN_MASK, VECTOR_BASEM, VECTOR_BASEN>(constInfo, runInfo, *attenMaskInfoPtr, attenMaskOrYInQue,
                                                               pseOrDyInQue, attenMaskU8Gm);
    CopyInPse<OUTDTYPE, CALC_TYPE, IS_PSE>(constInfo, runInfo, *pseInfoPtr, pseOrDyInQue, pseGm);
    CalculatePseMulsSelSimpleSoftMax<OUTDTYPE, CALC_TYPE, IS_FP8_INPUT, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD(DETER_SPARSE_TYPE),
                                     VECTOR_BASEM, VECTOR_BASEN>(
        constInfo, runInfo, *pseInfoPtr, *attenMaskInfoPtr, maxSumQue[runInfo.commonRunInfo.taskIdMod2],
        attenMaskOrYInQue, pseOrDyInQue, mm2ResTensor, mm2ResTensor, pseSlope);
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec3(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                                               LocalTensor<CALC_TYPE> &mm1ResTensor,
                                                               LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                               FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF3: sub + mul
    // VF4: dq dk cast + nd2nz
    ///////////////////////////////////////////////////////////////
    if (dropInfoPtr->dropMaskOuter) {
        if (dropInfoPtr->boolMode) {
            CopyInDropOuter<IS_DROP>(dropMaskBuf, attenMaskOrYInQue, dropMaskWorkspaceGm, runInfo.commonRunInfo,
                                     constInfo.commonConstInfo, *dropInfoPtr);
        } else {
            CopyInDropOuter<IS_DROP>(dropMaskBuf, attenMaskOrYInQue, dropMaskGm, runInfo.commonRunInfo,
                                     constInfo.commonConstInfo, *dropInfoPtr);
        }
    } else {
        GenDropMask<IS_DROP>(dropMaskBuf, dropmaskIndexVecBuf, runInfo.commonRunInfo, constInfo.commonConstInfo,
                             *dropInfoPtr);
    }
    CalculateDropout<CALC_TYPE, IS_DROP, VECTOR_BASEN>(constInfo, runInfo, *dropInfoPtr, mm1ResTensor, mm1ResTensor,
                                                       dropMaskBuf);
 
    LocalTensor<CALC_TYPE> softmaxGradResTensor = softmaxGradResBuf.Get<CALC_TYPE>();
    LocalTensor<INPUT_TYPE> vecOutBuffer = dSOutQue.AllocTensor<INPUT_TYPE>();
    if constexpr (IS_FP8_INPUT) {
        BroadcastSubMul<CALC_TYPE, static_cast<uint16_t>(VECTOR_BASEN), 0, IS_FP8_INPUT>(mm1ResTensor, mm1ResTensor,
            softmaxGradResTensor, mm2ResTensor, runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
    } else {
        if (runInfo.commonRunInfo.s2RealSize > static_cast<uint32_t>(S2TemplateType::Aligned64)) {
            BroadcastSubMul<CALC_TYPE, static_cast<uint16_t>(S2TemplateType::Aligned128), 0, IS_FP8_INPUT>(mm1ResTensor, mm1ResTensor,
                softmaxGradResTensor, mm2ResTensor, runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
        } else {
            if (constInfo.deterConstInfo.noNeedDeter) {
                BroadcastSubMul<CALC_TYPE, static_cast<uint16_t>(S2TemplateType::Aligned64), 0, IS_FP8_INPUT>(mm1ResTensor, mm1ResTensor,
                    softmaxGradResTensor, mm2ResTensor, runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
            } else { // 64~128的脏数据需要清零，避免后面的mm有脏数据参与计算
                BroadcastSubMul<CALC_TYPE, static_cast<uint16_t>(S2TemplateType::Aligned64), IS_DETER_OLD(DETER_SPARSE_TYPE), IS_FP8_INPUT>(
                    mm1ResTensor, mm1ResTensor, softmaxGradResTensor, mm2ResTensor, runInfo.commonRunInfo.halfS1RealSize,
                    runInfo.commonRunInfo.s2RealSize);
            }
        }
    }
 
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) { // 确定性计算尾块脏数据补零
        if (!constInfo.deterConstInfo.noNeedDeter) {
            if (runInfo.commonRunInfo.halfS1RealSize != VECTOR_BASEM) {
                Duplicate<CALC_TYPE>(mm1ResTensor[runInfo.commonRunInfo.halfS1RealSize * VECTOR_BASEN], 0,
                                     (VECTOR_BASEM - runInfo.commonRunInfo.halfS1RealSize) * VECTOR_BASEN);
            }
        }
    }
 
    // input type fp32, no post, mov muls here
    if constexpr (IS_FP32_INPUT) {
        Muls(mm1ResTensor, mm1ResTensor, constInfo.scaleValue, VECTOR_BASEM * VECTOR_BASEN);
    }
 
    LocalTensor<uint8_t> selrIndexesTensor;
    float qScaleDs = 1.0;
    constexpr uint32_t dsAmaxGmStride = 128;
    if constexpr (IS_FP8_INPUT) {
        LocalTensor<float> dsAmaxTensor = dsAmaxOutQue.AllocTensor<float>();
        DsAbsReduceMax<CALC_TYPE, static_cast<uint16_t>(VECTOR_BASEN)>(dsAmaxTensor, mm1ResTensor,
             runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
        event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDVToS);
        WaitFlag<HardEvent::V_S>(eventIDVToS);
        float curDsMax = dsAmaxTensor.GetValue(0);
 
        dsAmaxOutQue.EnQue(dsAmaxTensor);
        dsAmaxOutQue.DeQue<float>();
        DataCopyPad(dsAmaxWorkSpaceGm[vBlockIdx * dsAmaxGmStride], dsAmaxTensor, {1, 4, 0, 0});
 
        CrossCoreSetFlag<1, PIPE_MTE3>(SYNC_V0_V1_DS_A_MAX_DONE_FLAG);
        CrossCoreWaitFlag<1, PIPE_MTE3>(SYNC_V0_V1_DS_A_MAX_DONE_FLAG);
        event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
 
        int64_t anotherVBlockIdx = vSubBlockIdx == 0 ? (vBlockIdx + 1) : (vBlockIdx - 1);
        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            dsAmaxWorkSpaceGm[anotherVBlockIdx * dsAmaxGmStride]);
        float dsAmax = Max<float>(curDsMax, dsAmaxWorkSpaceGm.GetValue(anotherVBlockIdx * dsAmaxGmStride));
        if (dsAmax > 1e-6f) {
            qScaleDs = FP8_MAX / dsAmax;
            Muls(mm1ResTensor, mm1ResTensor, qScaleDs, VECTOR_BASEM * VECTOR_BASEN);
        }
        FagCVSharedParams sharedParams;
        InitCubeVecSharedParams(sharedParams, cBlockIdx, vSubBlockIdx, qScaleDs);
        dsAmaxOutQue.FreeTensor(dsAmaxTensor);
        selrIndexesTensor = vselrIndexesBuf.Get<uint8_t>();
    }
 
    CastTransdataDeconflict<INPUT_TYPE, CALC_TYPE, VECTOR_BASEN>(vecOutBuffer, mm1ResTensor, selrIndexesTensor,
                                                                 VECTOR_BASEM);
    dSOutQue.EnQue(vecOutBuffer);
    dSOutQue.DeQue<INPUT_TYPE>();
 
    // copy ds from ub to l1
    LocalTensor<INPUT_TYPE> dsL1Tensor = dstBuffer.GetTensor<INPUT_TYPE>();
    CopyUB2L1<true>(runInfo, dsL1Tensor, vecOutBuffer);
 
    dSOutQue.FreeTensor(vecOutBuffer);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec3Quant(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                                               Buffer<BufferType::L1, SyncType::NO_SYNC> &dstTransBuffer,
                                                               LocalTensor<CALC_TYPE> &mm1ResTensor,
                                                               LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                               FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF3: sub + mul
    // VF4: dq dk cast + nd2nz
    ///////////////////////////////////////////////////////////////
    if (dropInfoPtr->dropMaskOuter) {
        if (dropInfoPtr->boolMode) {
            CopyInDropOuter<IS_DROP>(dropMaskBuf, attenMaskOrYInQue, dropMaskWorkspaceGm, runInfo.commonRunInfo,
                                     constInfo.commonConstInfo, *dropInfoPtr);
        } else {
            CopyInDropOuter<IS_DROP>(dropMaskBuf, attenMaskOrYInQue, dropMaskGm, runInfo.commonRunInfo,
                                     constInfo.commonConstInfo, *dropInfoPtr);
        }
    } else {
        GenDropMask<IS_DROP>(dropMaskBuf, dropmaskIndexVecBuf, runInfo.commonRunInfo, constInfo.commonConstInfo,
                             *dropInfoPtr);
    }
    CalculateDropout<CALC_TYPE, IS_DROP, VECTOR_BASEN>(constInfo, runInfo, *dropInfoPtr, mm1ResTensor, mm1ResTensor,
                                                       dropMaskBuf);
 
    LocalTensor<CALC_TYPE> softmaxGradResTensor = softmaxGradResBuf.Get<CALC_TYPE>();
    LocalTensor<INPUT_TYPE> vecOutBuffer = dSOutQue.AllocTensor<INPUT_TYPE>();
    BroadcastSubMul<CALC_TYPE, static_cast<uint16_t>(VECTOR_BASEN), 0, IS_FP8_INPUT>(mm1ResTensor, mm1ResTensor,
        softmaxGradResTensor, mm2ResTensor, runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
 
    LocalTensor<uint8_t> selrIndexesTensor;
    float qScaleDs = 1.0;
    constexpr uint32_t dsAmaxGmStride = 128;

    LocalTensor<float> dsAmaxTensor = dsAmaxOutQue.AllocTensor<float>();
    DsAbsReduceMax<CALC_TYPE, static_cast<uint16_t>(VECTOR_BASEN)>(dsAmaxTensor, mm1ResTensor,
            runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
    float curDsMax = dsAmaxTensor.GetValue(0);

    dsAmaxOutQue.EnQue(dsAmaxTensor);
    dsAmaxOutQue.DeQue<float>();
    DataCopyPad(dsAmaxWorkSpaceGm[vBlockIdx * dsAmaxGmStride], dsAmaxTensor, {1, 4, 0, 0});

    CrossCoreSetFlag<1, PIPE_MTE3>(SYNC_V0_V1_DS_A_MAX_DONE_FLAG);
    CrossCoreWaitFlag<1, PIPE_MTE3>(SYNC_V0_V1_DS_A_MAX_DONE_FLAG);
    event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);

    int64_t anotherVBlockIdx = vSubBlockIdx == 0 ? (vBlockIdx + 1) : (vBlockIdx - 1);
    DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        dsAmaxWorkSpaceGm[anotherVBlockIdx * dsAmaxGmStride]);
    float dsAmax = Max<float>(curDsMax, dsAmaxWorkSpaceGm.GetValue(anotherVBlockIdx * dsAmaxGmStride));
    if (dsAmax > 1e-6f) {
        qScaleDs = FP8_MAX / dsAmax;
        Muls(mm1ResTensor, mm1ResTensor, qScaleDs, VECTOR_BASEM * VECTOR_BASEN);
    }
    FagCVSharedParams sharedParams;
    InitCubeVecSharedParams(sharedParams, cBlockIdx, vSubBlockIdx, qScaleDs);
    dsAmaxOutQue.FreeTensor(dsAmaxTensor);
    selrIndexesTensor = vselrIndexesBuf.Get<uint8_t>();
 
    CastTransdataDeconflict<INPUT_TYPE, CALC_TYPE, VECTOR_BASEN>(vecOutBuffer, mm1ResTensor, selrIndexesTensor,
                                                                 VECTOR_BASEM);
    dSOutQue.EnQue(vecOutBuffer);
    dSOutQue.DeQue<INPUT_TYPE>();
 
    // copy ds from ub to l1
    LocalTensor<INPUT_TYPE> dsL1Tensor = dstBuffer.GetTensor<INPUT_TYPE>();
    LocalTensor<INPUT_TYPE> dsTransL1Tensor = dstTransBuffer.GetTensor<INPUT_TYPE>();
    CopyUB2L1<true>(runInfo, dsL1Tensor, vecOutBuffer);
    CopyUB2L1<false>(runInfo, dsTransL1Tensor, vecOutBuffer); 
    dSOutQue.FreeTensor(vecOutBuffer);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec4(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                                               LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                               FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF5: cast + nd2nz
    ///////////////////////////////////////////////////////////////
    LocalTensor<uint8_t> selrIndexesTensor;
    LocalTensor<OUTDTYPE> vecOutBuffer1 = pOutQue.AllocTensor<OUTDTYPE>();
    CalculateDropout<CALC_TYPE, IS_DROP, VECTOR_BASEN>(constInfo, runInfo, *dropInfoPtr, mm2ResTensor, mm2ResTensor,
                                                       dropMaskBuf);
    CastTransdataDeconflict<OUTDTYPE, CALC_TYPE, VECTOR_BASEN>(vecOutBuffer1, mm2ResTensor, selrIndexesTensor,
                                                                 VECTOR_BASEM);
    pOutQue.EnQue(vecOutBuffer1);
    pOutQue.DeQue<OUTDTYPE>();
 
    // copy p from ub to l1
    LocalTensor<OUTDTYPE> pL1Tensor = dstBuffer.GetTensor<OUTDTYPE>();
    CopyUB2L1Quant<false>(runInfo, pL1Tensor, vecOutBuffer1);
 
    pOutQue.FreeTensor(vecOutBuffer1);
}
 
TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB, uint8_t MM_IDX>
__aicore__ inline void
FAGBlockVec<TEMPLATE_ARGS>::ProcessMulsAndCast(typename DqkvResPos<T, IS_WRITE_UB>::PosType inputTensor,
                                               FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if constexpr (IS_WRITE_UB) {
        DqkvMulsAndCastFromUB<MM_IDX>(constInfo, runInfo, inputTensor, dSOutQue);
    } else {
        DqkvMulsAndCastFromGM<MM_IDX>(constInfo, runInfo, inputTensor, attenMaskOrYInQue, dSOutQue);
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
template <uint8_t MM_IDX>
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::DqkvMulsAndCastFromUB(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                                                         LocalTensor<CALC_TYPE> &inputTensor,
                                                                         TQue<QuePosition::VECOUT, 1> &outQue)
{
    if ((MM_IDX == DQ_IDX && runInfo.commonRunInfo.halfS1RealSize == 0) ||
        (MM_IDX != DQ_IDX && runInfo.halfS2RealSize == 0)) {
        return;
    } 
    uint64_t dSize = (MM_IDX == DV_IDX) ? constInfo.commonConstInfo.dSizeV : constInfo.commonConstInfo.dSize;
    DataCopyExtParams intriParamsOut;
    intriParamsOut.blockCount = (MM_IDX == DQ_IDX) ? runInfo.commonRunInfo.halfS1RealSize : runInfo.halfS2RealSize;
    intriParamsOut.blockLen = dSize * sizeof(OUTDTYPE);
    intriParamsOut.srcStride = 0;
    GlobalTensor<OUTDTYPE> dqkvGmTensor = MM_IDX == DQ_IDX ? dqGm : (MM_IDX == DK_IDX ? dkGm : dvGm);
    uint32_t dataSize = intriParamsOut.blockCount * AlignTo16(dSize);
    uint64_t dqkvGmOffset = (MM_IDX == DQ_IDX) ?
                               runInfo.commonRunInfo.queryOffset :
                               (MM_IDX == DK_IDX ? runInfo.commonRunInfo.keyOffset : runInfo.commonRunInfo.valueOffset);
 
    uint64_t halfSRealSize =
        (MM_IDX == DQ_IDX) ? runInfo.commonRunInfo.firstHalfS1RealSize : runInfo.firstHalfS2RealSize;
    if constexpr (IS_TND) {
        intriParamsOut.dstStride =
            static_cast<uint32_t>((constInfo.commonConstInfo.n2G - 1) * dSize * sizeof(OUTDTYPE));
        dqkvGmOffset += vSubBlockIdx * dSize * constInfo.commonConstInfo.n2G * halfSRealSize;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            intriParamsOut.dstStride = 0;
            dqkvGmOffset += vSubBlockIdx * halfSRealSize * dSize;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            intriParamsOut.dstStride = static_cast<uint32_t>((constInfo.bSize * constInfo.commonConstInfo.n2G - 1) * dSize * sizeof(OUTDTYPE));
            dqkvGmOffset += vSubBlockIdx * halfSRealSize * constInfo.commonConstInfo.n2G * constInfo.bSize * dSize;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            intriParamsOut.dstStride = static_cast<uint32_t>((constInfo.commonConstInfo.n2G - 1) * dSize * sizeof(OUTDTYPE));
            dqkvGmOffset += vSubBlockIdx * halfSRealSize * constInfo.commonConstInfo.n2G * dSize;
        }
    }
 
    if constexpr (MM_IDX != DV_IDX) {
        Muls(inputTensor, inputTensor, constInfo.scaleValue, dataSize);
    }
 
    LocalTensor<OUTDTYPE> dqkvCastTensor = outQue.template AllocTensor<OUTDTYPE>();
    Cast(dqkvCastTensor, inputTensor, RoundMode::CAST_ROUND, dataSize);
    outQue.EnQue(dqkvCastTensor);
    outQue.template DeQue<OUTDTYPE>();
 
    DataCopyPad(dqkvGmTensor[dqkvGmOffset], dqkvCastTensor, intriParamsOut);
    outQue.FreeTensor(dqkvCastTensor);
}
 
TEMPLATES_DEF_NO_DEFAULT
template <uint8_t MM_IDX>
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::DqkvMulsAndCastFromGM(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                                                         GlobalTensor<CALC_TYPE> &inputTensor,
                                                                         TQue<QuePosition::VECIN, 1> &inQue,
                                                                         TQue<QuePosition::VECOUT, 1> &outQue)
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    uint32_t dSize;
    if constexpr (MM_IDX == DV_IDX && IS_D_NO_EQUAL) {
        dSize = constInfo.commonConstInfo.dSizeV;
    } else {
        dSize = constInfo.commonConstInfo.dSize;
    }
    uint32_t curDAlign = AlignTo16(dSize);
    uint64_t halfSRealSize = 0;
    uint64_t firsthalfSRealSize = 0;
    if constexpr (SPLIT_AXIS == BN2) {
        halfSRealSize = (MM_IDX == DQ_IDX) ? runInfo.commonRunInfo.halfS1RealSize : runInfo.halfS2RealSize;
        firsthalfSRealSize = (MM_IDX == DQ_IDX) ? runInfo.commonRunInfo.firstHalfS1RealSize : runInfo.firstHalfS2RealSize;
    } else {
        halfSRealSize = runInfo.halfS2RealSize;
        firsthalfSRealSize = runInfo.firstHalfS2RealSize;
    }
 
    uint32_t maxLoopSize = VECTOR_BASEM * VECTOR_BASEN / curDAlign; 
    uint32_t loopNum = Ceil<uint32_t>(halfSRealSize, maxLoopSize);
    if (loopNum == 0) {
        return;
    }
 
    uint32_t loopSize = Ceil<uint32_t>(halfSRealSize, loopNum);
    uint32_t tailLoopSize = halfSRealSize - (loopNum - 1) * loopSize;
    uint32_t curLoopSize = loopSize;
    DataCopyExtParams intriParamsOut;
    intriParamsOut.srcStride = 0;
    uint64_t dqkvGmOffset = (MM_IDX == DQ_IDX) ?    
                               runInfo.commonRunInfo.queryOffset :
                               (MM_IDX == DK_IDX ? runInfo.commonRunInfo.keyOffset : runInfo.commonRunInfo.valueOffset);
    if constexpr (IS_TND) {
        intriParamsOut.dstStride = static_cast<uint32_t>((constInfo.commonConstInfo.n2G - 1) * dSize * sizeof(OUTDTYPE));
        dqkvGmOffset += vSubBlockIdx * firsthalfSRealSize * dSize * constInfo.commonConstInfo.n2G;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            intriParamsOut.dstStride = 0;
            dqkvGmOffset += vSubBlockIdx * firsthalfSRealSize * dSize;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            intriParamsOut.dstStride = static_cast<uint32_t>((constInfo.bSize * constInfo.commonConstInfo.n2G - 1) * dSize * sizeof(OUTDTYPE));
            dqkvGmOffset += vSubBlockIdx * firsthalfSRealSize * constInfo.commonConstInfo.n2G * constInfo.bSize * dSize;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            intriParamsOut.dstStride = static_cast<uint32_t>((constInfo.commonConstInfo.n2G - 1) * dSize * sizeof(OUTDTYPE));
            dqkvGmOffset += vSubBlockIdx * firsthalfSRealSize * constInfo.commonConstInfo.n2G * dSize;
        }
    }
 
    GlobalTensor<OUTDTYPE> dqkvGmTensor = MM_IDX == DQ_IDX ? dqGm : (MM_IDX == DK_IDX ? dkGm : dvGm);
    int64_t dkvWorkSpaceOffet = this->cBlockIdx * this->CUBE_BASEN * this->HEAD_DIM_ALIGN + vSubBlockIdx * firsthalfSRealSize * curDAlign;
    if constexpr (IS_BN2_MULTIBLK && MM_IDX == DQ_IDX) {
        dkvWorkSpaceOffet = this->cBlockIdx * AlignTo128(constInfo.commonConstInfo.s1Size) * HEAD_DIM_ALIGN +
            runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * HEAD_DIM_ALIGN +
            vSubBlockIdx * firsthalfSRealSize * curDAlign;
    }
    uint32_t data_size = curLoopSize * curDAlign;
    for (uint32_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
        if (loopIdx == loopNum - 1) {
            curLoopSize = tailLoopSize;
            data_size = curLoopSize * curDAlign;
        }
 
        LocalTensor<CALC_TYPE> dqkvTensor = inQue.AllocTensor<CALC_TYPE>();
        DataCopy(dqkvTensor,
                 inputTensor[dkvWorkSpaceOffet + loopIdx * loopSize * curDAlign],
                 data_size);
        
        inQue.EnQue(dqkvTensor);
        inQue.DeQue();
        if constexpr (MM_IDX != DV_IDX) {
            Muls(dqkvTensor, dqkvTensor, constInfo.scaleValue, data_size);
        }
        LocalTensor<OUTDTYPE> dqkvCastTensor = outQue.template AllocTensor<OUTDTYPE>();
        Cast(dqkvCastTensor, dqkvTensor, RoundMode::CAST_ROUND, data_size);
        inQue.FreeTensor(dqkvTensor);
        outQue.EnQue(dqkvCastTensor);
        outQue.template DeQue<OUTDTYPE>();
 
        intriParamsOut.blockCount = curLoopSize;
        intriParamsOut.blockLen = dSize * sizeof(OUTDTYPE);
 
        DataCopyPad(dqkvGmTensor[dqkvGmOffset], dqkvCastTensor, intriParamsOut);
        outQue.FreeTensor(dqkvCastTensor);
 
        if constexpr (IS_TND) {
            dqkvGmOffset += loopSize * dSize * constInfo.commonConstInfo.n2G;
        } else {
            if (constInfo.commonConstInfo.layoutType == BNGSD) {
                dqkvGmOffset += loopSize * dSize;
            } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                dqkvGmOffset += loopSize * constInfo.commonConstInfo.n2G * constInfo.bSize * dSize;
            } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                dqkvGmOffset += loopSize * constInfo.commonConstInfo.n2G * dSize;
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <const bool IS_DK>
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessPostDeter(FagConstInfo &constInfo, GlobalTensor<float> dkvWorkSpaceTensor,
    GlobalTensor<INPUT_TYPE> &dkvGmTensor, int64_t specialHalfS2RealSize, int64_t specialFirstHalfS2RealSize, uint64_t dAlign16, uint64_t dvAlign16, int64_t specialDkGmOffset, int64_t specialDvGmOffset)
{
    TQue outQue = IS_DK ? dSOutQue : pOutQue;
    if (specialHalfS2RealSize == 0) {
        return;
    }
    uint32_t dSize = constInfo.commonConstInfo.dSize;
    uint32_t curDAlign = dAlign16;
    int64_t dkvGmOffset = IS_DK ? specialDkGmOffset : specialDvGmOffset;
    
    if constexpr (!IS_DK && IS_D_NO_EQUAL) {
        dSize = constInfo.commonConstInfo.dSizeV;
        curDAlign = dvAlign16;
    }
 
    uint32_t maxLoopSize = this->VECTOR_BASEM * this->VECTOR_BASEN / curDAlign; 
    uint32_t loopNum = Ceil<uint32_t>(specialHalfS2RealSize, maxLoopSize);
    if (loopNum == 0) {
        return;
    }
 
    uint32_t loopSize = Ceil<uint32_t>(specialHalfS2RealSize, loopNum);
    uint32_t tailLoopSize = specialHalfS2RealSize - (loopNum - 1) * loopSize;
    uint32_t curLoopSize = loopSize;
    DataCopyExtParams intriParamsOut;
    intriParamsOut.srcStride = 0;
    intriParamsOut.dstStride = static_cast<uint32_t>((constInfo.commonConstInfo.n2G - 1) * dSize * sizeof(INPUT_TYPE));
    dkvGmOffset += this->vSubBlockIdx * specialFirstHalfS2RealSize * dSize * constInfo.commonConstInfo.n2G;
 
    uint32_t data_size = curLoopSize * curDAlign;
    for (uint32_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
        if (loopIdx == loopNum - 1) {
            curLoopSize = tailLoopSize;
            data_size = curLoopSize * curDAlign;
        }
 
        LocalTensor<CALC_TYPE> dkvTensor = attenMaskOrYInQue.AllocTensor<CALC_TYPE>();
        DataCopy(dkvTensor, dkvWorkSpaceTensor[this->vSubBlockIdx * specialFirstHalfS2RealSize * curDAlign + loopIdx * loopSize * curDAlign],
                 data_size);
        
        attenMaskOrYInQue.EnQue(dkvTensor);
        attenMaskOrYInQue.DeQue();
        if constexpr (IS_DK) {
            Muls(dkvTensor, dkvTensor, constInfo.scaleValue, data_size);
        }
        LocalTensor<INPUT_TYPE> dkvCastTensor = outQue.template AllocTensor<INPUT_TYPE>();
        Cast(dkvCastTensor, dkvTensor, RoundMode::CAST_ROUND, data_size);
        attenMaskOrYInQue.FreeTensor(dkvTensor);
        outQue.EnQue(dkvCastTensor);
        outQue.template DeQue<INPUT_TYPE>();
 
        intriParamsOut.blockCount = curLoopSize;
        intriParamsOut.blockLen = dSize * sizeof(INPUT_TYPE);
 
        DataCopyPad(dkvGmTensor[dkvGmOffset], dkvCastTensor, intriParamsOut);
        outQue.FreeTensor(dkvCastTensor);
        dkvGmOffset += loopSize * dSize * constInfo.commonConstInfo.n2G;
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::InitCubeVecSharedParams(
    FagCVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, float qScaleDs)
{
    sharedParams.qScaleDs = qScaleDs;
    /* ssbuf send message */
    if ASCEND_IS_AIV {  
        if (subBlockIdx == 0) {
            auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
            auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);  // 
            #pragma unroll
            for (int i = 0; i < sizeof(FagCVSharedParams) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
                *tempTilingSSbuf = *tempTiling;
            }
            CrossCoreSetFlag<SYNC_MODE, PIPE_S>(15);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::DequantAndCopy2L1(Buffer<BufferType::L1, SyncType::NO_SYNC> &vL1Buffer, LocalTensor<CALC_TYPE> mm1ResTensor, FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if (runInfo.halfS2RealSize == 0) {
        return;
    }

    // 1 - valueGm to UB
    int64_t srcGmOffset = 0;
    int64_t transpose_stride = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    if (constInfo.commonConstInfo.layoutType == BNGSD) {
        bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2S2Dv;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.s2Dv;
        s2Offset = runInfo.s2oIdx * VECTOR_BASEN * constInfo.commonConstInfo.dSizeV +
                   runInfo.firstHalfS2RealSize * GetSubBlockIdx() * constInfo.commonConstInfo.dSizeV;
        transpose_stride = 0;
    } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
        s2Offset = runInfo.s2oIdx * VECTOR_BASEN * constInfo.commonConstInfo.bN2Dv +
                   runInfo.firstHalfS2RealSize * GetSubBlockIdx() * constInfo.commonConstInfo.bN2Dv;
        bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2Dv;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.dSizeV;
        transpose_stride = (constInfo.commonConstInfo.bN2Dv - constInfo.commonConstInfo.dSizeV) * sizeof(INPUT_TYPE);   
    } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
        bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2S2Dv;
        s2Offset = runInfo.s2oIdx * VECTOR_BASEN * constInfo.commonConstInfo.n2Dv +
                   runInfo.firstHalfS2RealSize * GetSubBlockIdx() * constInfo.commonConstInfo.n2Dv;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.dSizeV;
        transpose_stride = (constInfo.commonConstInfo.n2Dv - constInfo.commonConstInfo.dSizeV) * sizeof(INPUT_TYPE);
    }
    srcGmOffset = bOffset + n2Offset + s2Offset;
    uint32_t dstBlockStride = (HEAD_DIM_ALIGN - constInfo.dAlignToBlock) * sizeof(INPUT_TYPE) / 32;

    // attenMaskOrYInQue申请的UB size=VEC_M*VEC_N*4B，足够fp8拷入使用
    LocalTensor<INPUT_TYPE> valueB8Tensor = attenMaskOrYInQue.AllocTensor<INPUT_TYPE>();
    DataCopyPad(valueB8Tensor.template ReinterpretCast<uint8_t>(), valueGm[srcGmOffset].template ReinterpretCast<uint8_t>(),
        {static_cast<uint16_t>(runInfo.halfS2RealSize),
        static_cast<uint32_t>(constInfo.commonConstInfo.dSizeV * sizeof(INPUT_TYPE)),
        static_cast<uint32_t>(transpose_stride), dstBlockStride, 0},
        {true, 0, static_cast<uint8_t>((constInfo.dAlignToBlock - constInfo.commonConstInfo.dSizeV)), 0});

    attenMaskOrYInQue.EnQue(valueB8Tensor);
    attenMaskOrYInQue.DeQue();

    LocalTensor<uint8_t> selrIndexesTensor;
    LocalTensor<OUTDTYPE> vL1Tensor = vL1Buffer.template GetTensor<OUTDTYPE>();
    uint32_t loopNum = Ceil<uint32_t>(runInfo.halfS2RealSize, constInfo.sfmgMaxLoopSize);   // 2
    uint32_t loopSize = Ceil<uint32_t>(runInfo.halfS2RealSize, loopNum);    // 64
    uint32_t tailLoopSize = runInfo.halfS2RealSize - (loopNum - 1) * loopSize;
    uint32_t curLoopSize = loopSize;
    for (int32_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
        if (loopIdx == loopNum - 1) {
            curLoopSize = tailLoopSize;
        }
        Cast(mm1ResTensor, valueB8Tensor[loopIdx * loopSize * HEAD_DIM_ALIGN], RoundMode::CAST_NONE, curLoopSize * HEAD_DIM_ALIGN);
        Muls(mm1ResTensor, mm1ResTensor, runInfo.quantScaleInfo.deqScaleVValue, curLoopSize * HEAD_DIM_ALIGN);

        LocalTensor<OUTDTYPE> valueB16Tensor = dSOutQue.AllocTensor<OUTDTYPE>();
        CastTransdataDeconflict<OUTDTYPE, CALC_TYPE, HEAD_DIM_ALIGN>(valueB16Tensor, mm1ResTensor, selrIndexesTensor, curLoopSize);
        dSOutQue.EnQue(valueB16Tensor);
        dSOutQue.DeQue<OUTDTYPE>();

        CopyUB2L1Quant<true>(runInfo, vL1Tensor, valueB16Tensor, loopNum, loopIdx, loopSize, curLoopSize);
        dSOutQue.FreeTensor(valueB16Tensor);
    }

    attenMaskOrYInQue.FreeTensor(valueB8Tensor);
}
 
TEMPLATES_DEF
class FAGBlockVecDummy {
public:
    __aicore__ inline void InitUbBuffer(){};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR value, GM_ADDR dy, GM_ADDR y, GM_ADDR pseShift, GM_ADDR dropMask,
                                            GM_ADDR attenMask, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                            GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy,
                                            GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace){};
    __aicore__ inline void SetVecBlockParams(TPipe *pipe, FagTilingType tilingData, uint32_t vBlockIdx,
                                             uint32_t cBlockIdx, uint32_t vSubBlockIdx, AttenMaskInfo &attenMaskInfo,
                                             PseInfo &pseInfo, DropMaskInfo &dropInfo){};
    __aicore__ inline void ProcessVec1(FagConstInfo &constInfo, FagRunInfo &runInfo){};
    __aicore__ inline void ProcessVec2(LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo){};
    __aicore__ inline void ProcessVec3(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<CALC_TYPE> &mm1ResTensor,
                                       LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo){};
    __aicore__ inline void ProcessVec3Quant(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                            Buffer<BufferType::L1, SyncType::NO_SYNC> &dstTransBuffer,
                                            LocalTensor<CALC_TYPE> &mm1ResTensor,
                                            LocalTensor<CALC_TYPE> &mm2ResTensor,
                                            FagConstInfo &constInfo, FagRunInfo &runInfo){};
    __aicore__ inline void ProcessVec4(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                       FagConstInfo &constInfo, FagRunInfo &runInfo){};
    template <typename T, bool IS_WRITE_UB, uint8_t MM_IDX>
    __aicore__ inline void ProcessMulsAndCast(typename DqkvResPos<T, IS_WRITE_UB>::PosType inputTensor,
                                              FagConstInfo &constInfo, FagRunInfo &runInfo){};
    __aicore__ inline void CopyMaxSum(FagConstInfo &constInfo, FagRunInfo &runInfo, int64_t taskId){};
    __aicore__ inline void InitCubeVecSharedParams(FagCVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, float qScaleDs){};
    template <const bool IS_DK>
    __aicore__ inline void ProcessPostDeter(FagConstInfo &constInfo, GlobalTensor<float> dkvWorkSpaceTensor, GlobalTensor<INPUT_TYPE> &dkvGmTensor,
        int64_t specialHalfS2RealSize, int64_t specialFirstHalfS2RealSize, uint64_t dAlign16, uint64_t dvAlign16, int64_t specialDkGmOffset, int64_t specialDvGmOffset){};
    __aicore__ inline void DequantAndCopy2L1(Buffer<BufferType::L1, SyncType::NO_SYNC> &vL1Buffer, LocalTensor<CALC_TYPE> mm1ResTensor,
                                             FagConstInfo &constInfo, FagRunInfo &runInfo){};
};
 
} // namespace FagBaseApi
 
#endif