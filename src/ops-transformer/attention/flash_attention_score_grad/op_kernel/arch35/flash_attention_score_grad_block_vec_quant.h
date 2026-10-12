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
 * \file flash_attention_score_grad_block_vec_fp8.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_BLOCK_VEC_FP8_H
#define FLASH_ATTENTION_SCORE_GRAD_BLOCK_VEC_FP8_H
 
#include "flash_attention_score_grad_common.h"
#include "vector_api/vf_anti_quant_compute_p_ds.h"
namespace FagBaseApi {

TEMPLATES_DEF
class FAGBlockVecQuant {
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
    __aicore__ inline FAGBlockVecQuant(){};
    __aicore__ inline void SetVecBlockParams(TPipe *pipe, FagTilingType tilingData, uint32_t vBlockIdx,
                                             uint32_t cBlockIdx, uint32_t vSubBlockIdx, FagConstInfo &constInfo);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR value, GM_ADDR dy, GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                            GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy,
                                            GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, int64_t sfmgWorkSpaceOffset);
    __aicore__ inline void InitUbBuffer(const LocalTensor<CALC_TYPE> &commonTensors);
    __aicore__ inline void CopyMaxSumD(FagConstInfo &constInfo, FagRunInfo &runInfo, uint16_t firstHalfS1, uint16_t currentRealS1);
    __aicore__ inline void CopyUB2L1(FagRunInfo &runInfo, const LocalTensor<INPUT_TYPE> &dstTensor,
                                     const LocalTensor<CALC_TYPE> &srcTensor, uint16_t realS1, uint16_t realS2);
    __aicore__ inline void ProcessPDs(const LocalTensor<INPUT_TYPE> &pl1Tensor, const LocalTensor<INPUT_TYPE> &dsl1Tensor, const LocalTensor<CALC_TYPE> &spTensor, const LocalTensor<CALC_TYPE> &dpdsTensor, int8_t sdpId, FagConstInfo &constInfo, FagRunInfo &runInfo);
    template<uint64_t mode>
    __aicore__ inline void DequantDqkv(const LocalTensor<CALC_TYPE> &outTensor, const LocalTensor<CALC_TYPE> &outTensor2, FagConstInfo &constInfo, FagRunInfo &runInfo, int64_t i, int64_t id);
    __aicore__ inline void DequantOut(const LocalTensor<CALC_TYPE> &outTensor, uint16_t realLength, float scale);
    template<uint64_t mode>
    __aicore__ inline void CopyDqkv2GM(FagRunInfo &runInfo, FagConstInfo &constInfo, const GlobalTensor<CALC_TYPE> &outGm, const LocalTensor<CALC_TYPE> &outTensor, const GlobalTensor<CALC_TYPE> &outGm2, const LocalTensor<CALC_TYPE> &outTensor2);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    constexpr static bool IS_FP8_INPUT =
        IsSameType<INPUT_TYPE, fp8_e5m2_t>::value || IsSameType<INPUT_TYPE, fp8_e4m3fn_t>::value || IsSameType<INPUT_TYPE, hifloat8_t>::value;
    constexpr static bool IS_FP32_INPUT = IsSameType<INPUT_TYPE, float>::value;
    constexpr static float FP8_MAX = IsSameType<INPUT_TYPE, fp8_e5m2_t>::value ? 57344 : (IsSameType<INPUT_TYPE, fp8_e4m3fn_t>::value ? 448 : 32768);
    constexpr static uint32_t DETER_OFFSET_UB_SIZE = 1024 * 3;
    constexpr static uint32_t CUBE_BASEM = static_cast<uint32_t>(s1TemplateType);
    constexpr static uint32_t CUBE_BASEN = static_cast<uint32_t>(s2TemplateType);
    constexpr static uint32_t HEAD_DIM_ALIGN = static_cast<uint32_t>(dTemplateType);
    constexpr static uint32_t VECTOR_BASEM = CUBE_BASEM / CV_CORE_RATIO;
    constexpr static uint32_t VECTOR_BASEN = CUBE_BASEN;
    constexpr static uint32_t INNER_VECTOR_BASEM = 128;
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

    constexpr static uint8_t PDS_COPY_IN_MAXSUMD_INNER_CORE_SYNC_EVENTS[4] = {0, 1, 2, 3};
    constexpr static uint8_t PDS_PDS_COPY_TO_L1_INNER_CORE_SYNC_EVENTS[2] = {0, 1};
    constexpr static uint8_t DQKV_UB_TO_GM_INNER_CORE_SYNC_EVENTS[3] = {4, 5, 6};

    // vector gm addr
    GlobalTensor<INPUT_TYPE> valueGm, dyGm;
    GlobalTensor<OUTDTYPE> yGm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm;
    GlobalTensor<float> deqScaleQGm, deqScaleKGm, deqScaleVGm, deqScaleDyGm;
    GlobalTensor<float> sfmgWorkspaceGm;
    
    GlobalTensor<OUTDTYPE> dqGm, dkGm, dvGm;
 
    // ub buffer
    TBuf<> vecQue;
    LocalTensor<float> maxTensor[4]; // 64 * 4 * 4
    LocalTensor<float> sumTensor[4]; // 64 * 4 * 4
    LocalTensor<float> dTensor[4]; // 64 * 4 * 4
    LocalTensor<uint16_t> permTensor; // 128 * 2
 
    TPipe *pipe;
 
    uint32_t vBlockIdx;
    uint32_t cBlockIdx;
    uint32_t vSubBlockIdx;

    uint32_t maxsumIdx = 0;
};
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVecQuant<TEMPLATE_ARGS>::SetVecBlockParams(TPipe *pipe, FagTilingType tilingData,
                                                                     uint32_t vBlockIdx, uint32_t cBlockIdx,
                                                                     uint32_t vSubBlockIdx, FagConstInfo &constInfo)
{
    this->pipe = pipe;
    this->vBlockIdx = vBlockIdx;
    this->cBlockIdx = cBlockIdx;
    this->vSubBlockIdx = vSubBlockIdx;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVecQuant<TEMPLATE_ARGS>::InitGlobalBuffer(GM_ADDR value, GM_ADDR dy, GM_ADDR y,
                                                                    GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                                                    GM_ADDR deqScaleQ, GM_ADDR deqScaleK,
                                                                    GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR dq,
                                                                    GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, int64_t sfmgWorkSpaceOffset)
{
    valueGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)value);
    dyGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)dy);
    yGm.SetGlobalBuffer((__gm__ OUTDTYPE *)y);

    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    deqScaleQGm.SetGlobalBuffer((__gm__ float *)deqScaleQ);
    deqScaleKGm.SetGlobalBuffer((__gm__ float *)deqScaleK);
    deqScaleVGm.SetGlobalBuffer((__gm__ float *)deqScaleV);
    deqScaleDyGm.SetGlobalBuffer((__gm__ float *)deqScaleDy);
    
    dqGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
    dkGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dk);
    dvGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dv);

    sfmgWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + sfmgWorkSpaceOffset / sizeof(float));
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVecQuant<TEMPLATE_ARGS>::InitUbBuffer(const LocalTensor<CALC_TYPE> &commonTensors)
{
    uint32_t ubOffset = QUANT_S1_BASE_COUNT;
    for (int64_t i = 0; i < NUM_FOUR; i++) {
        maxTensor[i] = commonTensors[ubOffset];
        sumTensor[i] = commonTensors[ubOffset + QUANT_S1_BASE_COUNT_DB];
        dTensor[i] = commonTensors[ubOffset + QUANT_S1_BASE_COUNT_DB * NUM_TWO];
        ubOffset += QUANT_S1_BASE_COUNT_DB * NUM_THREE;
    }

    // 256B
    permTensor = commonTensors[ubOffset].template ReinterpretCast<uint16_t>();
    for (uint16_t i = 0; i < BLOCK_SIZE; i++) {
        permTensor.SetValue(i * NUM_FOUR + 0, 0 * BLOCK_SIZE + i);
        permTensor.SetValue(i * NUM_FOUR + 1, 1 * BLOCK_SIZE + i);
        permTensor.SetValue(i * NUM_FOUR + NUM_TWO, NUM_TWO * BLOCK_SIZE + i);
        permTensor.SetValue(i * NUM_FOUR + NUM_THREE, NUM_THREE * BLOCK_SIZE + i);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVecQuant<TEMPLATE_ARGS>::AllocEventID()
{
    for (int64_t i = 0; i < NUM_FOUR; i++) {
        SetFlag<AscendC::HardEvent::V_MTE2>(PDS_COPY_IN_MAXSUMD_INNER_CORE_SYNC_EVENTS[i]);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVecQuant<TEMPLATE_ARGS>::FreeEventID()
{
    for (int64_t i = 0; i < NUM_FOUR; i++) {
        WaitFlag<AscendC::HardEvent::V_MTE2>(PDS_COPY_IN_MAXSUMD_INNER_CORE_SYNC_EVENTS[i]);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVecQuant<TEMPLATE_ARGS>::ProcessPDs(const LocalTensor<INPUT_TYPE> &pl1Tensor,
                                                                   const LocalTensor<INPUT_TYPE> &dsl1Tensor,
                                                                   const LocalTensor<CALC_TYPE> &spTensor,
                                                                   const LocalTensor<CALC_TYPE> &dpdsTensor,
                                                                   int8_t sdpId, FagConstInfo &constInfo,
                                                                   FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF1: compute p & ds
    ///////////////////////////////////////////////////////////////
    uint64_t dl_buffid = runInfo.quantRunInfo.s1Idx;

    // 反向同步
    WaitFlag<AscendC::HardEvent::V_MTE2>(PDS_COPY_IN_MAXSUMD_INNER_CORE_SYNC_EVENTS[maxsumIdx]);

    uint16_t realS1 = static_cast<uint16_t>(runInfo.quantRunInfo.innerS1RealSize[runInfo.quantRunInfo.s1Idx]);
    uint16_t realS2 = static_cast<uint16_t>(runInfo.quantRunInfo.innerS2RealSize[runInfo.quantRunInfo.s2Idx]);

    uint16_t firstHalfS1 = Ceil<uint16_t>(realS1, QUANT_S1_BASE_COUNT) * QUANT_S1_BASE_COUNT / NUM_TWO;
    uint16_t currentRealS1 = vSubBlockIdx == 0 ? firstHalfS1 : realS1 - firstHalfS1;
    uint16_t currentRealS2 = Ceil<uint16_t>(realS2, QUANT_S2_BASE_COUNT) * QUANT_S2_BASE_COUNT;

    if (currentRealS1 <= 0) {
        // 反向同步
        SetFlag<AscendC::HardEvent::V_MTE2>(PDS_COPY_IN_MAXSUMD_INNER_CORE_SYNC_EVENTS[maxsumIdx]);
        maxsumIdx = (maxsumIdx + 1) & NUM_THREE;
        return;
    }

    uint16_t maxCopySize = realS1 < firstHalfS1 ? realS1 : firstHalfS1;
    CopyMaxSumD(constInfo, runInfo, firstHalfS1, maxCopySize);
    SetFlag<AscendC::HardEvent::MTE2_V>(PDS_COPY_IN_MAXSUMD_INNER_CORE_SYNC_EVENTS[maxsumIdx]);
    WaitFlag<AscendC::HardEvent::MTE2_V>(PDS_COPY_IN_MAXSUMD_INNER_CORE_SYNC_EVENTS[maxsumIdx]);

    ComputePDS(spTensor, dpdsTensor, permTensor, maxTensor[maxsumIdx], sumTensor[maxsumIdx], dTensor[maxsumIdx],
               currentRealS2, currentRealS1, runInfo.quantRunInfo.qkDScale * constInfo.scaleValue,
               runInfo.quantRunInfo.vdyDScale, constInfo.pScaleD, constInfo.pScaleD * constInfo.dsScale);

    // 反向同步
    SetFlag<AscendC::HardEvent::V_MTE2>(PDS_COPY_IN_MAXSUMD_INNER_CORE_SYNC_EVENTS[maxsumIdx]);

    SetFlag<AscendC::HardEvent::V_MTE3>(PDS_PDS_COPY_TO_L1_INNER_CORE_SYNC_EVENTS[sdpId]);
    WaitFlag<AscendC::HardEvent::V_MTE3>(PDS_PDS_COPY_TO_L1_INNER_CORE_SYNC_EVENTS[sdpId]);
    CopyUB2L1(runInfo, pl1Tensor, spTensor, currentRealS1, currentRealS2);
    CopyUB2L1(runInfo, dsl1Tensor, dpdsTensor, currentRealS1, currentRealS2);
    maxsumIdx = (maxsumIdx + 1) & NUM_THREE;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVecQuant<TEMPLATE_ARGS>::CopyMaxSumD(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                                                    uint16_t firstHalfS1, uint16_t currentRealS1)
{
    int64_t maxSumDGmOffset =
        (runInfo.maxsumOffset + runInfo.quantRunInfo.s1Idx * INNER_VECTOR_BASEM + firstHalfS1 * vSubBlockIdx);
    uint16_t maxSumDSize = currentRealS1 * sizeof(float);

    DataCopyPad(maxTensor[maxsumIdx], softmaxMaxGm[maxSumDGmOffset], {1, maxSumDSize, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(sumTensor[maxsumIdx], softmaxSumGm[maxSumDGmOffset], {1, maxSumDSize, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(dTensor[maxsumIdx], sfmgWorkspaceGm[maxSumDGmOffset], {1, maxSumDSize, 0, 0}, {false, 0, 0, 0});
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVecQuant<TEMPLATE_ARGS>::CopyUB2L1(FagRunInfo &runInfo,
                                                                  const LocalTensor<INPUT_TYPE> &dstTensor,
                                                                  const LocalTensor<CALC_TYPE> &srcTensor,
                                                                  uint16_t realS1, uint16_t realS2)
{
    LocalTensor<int8_t> l1Tensor = dstTensor.template ReinterpretCast<int8_t>();
    LocalTensor<int8_t> ubTensor = srcTensor.template ReinterpretCast<int8_t>();

    uint16_t blockCount = realS2 / QUANT_S2_BASE_COUNT;
    uint32_t scmOffsetSingle = BLOCK_SIZE * INNER_VECTOR_BASEM;
    uint32_t scmOffset =
        realS1 > BLOCK_SIZE ? vSubBlockIdx * scmOffsetSingle * NUM_TWO : vSubBlockIdx * scmOffsetSingle;

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = blockCount;
    dataCopyParams.blockLen = QUANT_S2_BASE_COUNT;
    dataCopyParams.srcStride = QUANT_UB2L1_SRC_STRIDE * QUANT_S2_BASE_COUNT;
    DataCopy(l1Tensor[scmOffset], ubTensor, dataCopyParams);
    if (realS1 > BLOCK_SIZE) {
        DataCopy(l1Tensor[scmOffset + scmOffsetSingle], ubTensor[QUANT_UB2L1_SRC_OFFSET], dataCopyParams);
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <uint64_t mode>
__aicore__ inline void FAGBlockVecQuant<TEMPLATE_ARGS>::DequantDqkv(const LocalTensor<CALC_TYPE> &outTensor,
                                                                    const LocalTensor<CALC_TYPE> &outTensor2,
                                                                    FagConstInfo &constInfo, FagRunInfo &runInfo,
                                                                    int64_t i, int64_t id)
{
    // mode:
    // 0 dq
    // 1 dv
    // 2 dk
    float scale;
    float scale2;
    uint16_t realLength;
    int64_t offset;
    if constexpr (mode == 0) {
        scale = runInfo.quantRunInfo.dsScaleDMulDeqScaleK;
        realLength = static_cast<uint16_t>(runInfo.quantRunInfo.innerS1RealSize[runInfo.quantRunInfo.s1Idx]);
    } else {
        scale = runInfo.quantRunInfo.pScaleDMulDeqScaleDy;
        realLength = static_cast<uint16_t>(runInfo.quantRunInfo.innerS2RealSize[runInfo.quantRunInfo.s2Idx]);
    }
    DequantOut(outTensor, Ceil<uint16_t>(realLength, NUM_TWO), scale);
    SetFlag<AscendC::HardEvent::V_MTE3>(DQKV_UB_TO_GM_INNER_CORE_SYNC_EVENTS[mode]);
    if constexpr (mode != 0) {
        scale = runInfo.quantRunInfo.dsScaleDMulDeqScaleQ;
        DequantOut(outTensor2, Ceil<uint16_t>(realLength, NUM_TWO), scale);
        SetFlag<AscendC::HardEvent::V_MTE3>(DQKV_UB_TO_GM_INNER_CORE_SYNC_EVENTS[2]);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVecQuant<TEMPLATE_ARGS>::DequantOut(const LocalTensor<CALC_TYPE> &outTensor,
                                                                   uint16_t realLength, float scale)
{
    uint64_t out = outTensor.GetPhyAddr();
    __VEC_SCOPE__
    {
        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> vreg_x1, vreg_x2;
        // 每个核处理(S1, 1/2 D)
        for (uint16_t i = 0; i < realLength; i++) {
            DataCopy(vreg_x1, (__ubuf__ float *)out + i * HEAD_DIM_ALIGN * NUM_TWO);
            DataCopy(vreg_x2, (__ubuf__ float *)out + i * HEAD_DIM_ALIGN * NUM_TWO + HEAD_DIM_ALIGN);

            Muls(vreg_x1, vreg_x1, scale, preg_all);
            Muls(vreg_x2, vreg_x2, scale, preg_all);

            DataCopy((__ubuf__ float *)out + i * HEAD_DIM_ALIGN * NUM_TWO, vreg_x1, preg_all);
            DataCopy((__ubuf__ float *)out + i * HEAD_DIM_ALIGN * NUM_TWO + HEAD_DIM_ALIGN, vreg_x2, preg_all);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <uint64_t mode>
__aicore__ inline void FAGBlockVecQuant<TEMPLATE_ARGS>::CopyDqkv2GM(FagRunInfo &runInfo, FagConstInfo &constInfo,
                                                                    const GlobalTensor<CALC_TYPE> &outGm,
                                                                    const LocalTensor<CALC_TYPE> &outTensor,
                                                                    const GlobalTensor<CALC_TYPE> &outGm2,
                                                                    const LocalTensor<CALC_TYPE> &outTensor2)
{
    int64_t offset;
    int64_t blockCount;
    if constexpr (mode == 0) {
        offset = runInfo.quantRunInfo.qInnerOffset[runInfo.quantRunInfo.s1Idx] + vSubBlockIdx * 64;
        blockCount = runInfo.quantRunInfo.innerS1RealSize[runInfo.quantRunInfo.s1Idx];
    } else {
        offset = runInfo.quantRunInfo.kvInnerOffset[runInfo.quantRunInfo.s2Idx] + vSubBlockIdx * 64;
        blockCount = runInfo.quantRunInfo.innerS2RealSize[runInfo.quantRunInfo.s2Idx];
    }
    WaitFlag<AscendC::HardEvent::V_MTE3>(DQKV_UB_TO_GM_INNER_CORE_SYNC_EVENTS[mode]);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = blockCount;
    dataCopyParams.blockLen = QUANT_S1_BASE_COUNT * sizeof(CALC_TYPE);
    dataCopyParams.srcStride = QUANT_S1_BASE_COUNT * sizeof(CALC_TYPE) / BLOCK_SIZE;
    dataCopyParams.dstStride = constInfo.copyOutDStride;

    bool needAtomic =
        ((mode == 0 && (constInfo.s2Outer != 1 || (constInfo.s2Outer == 1 && runInfo.quantRunInfo.s2Idx == 0))) ||
         ((constInfo.s2Outer != 1 && runInfo.isValueReuse) ||
          !(runInfo.isFirstProcessBlock && constInfo.s2Outer == 1 && runInfo.quantRunInfo.s1Idx == 0)));
    if (needAtomic) {
        SetAtomicAdd<CALC_TYPE>();
    }

    DataCopyPad(outGm[offset], outTensor, dataCopyParams);
    if constexpr (mode != 0) {
        WaitFlag<AscendC::HardEvent::V_MTE3>(DQKV_UB_TO_GM_INNER_CORE_SYNC_EVENTS[NUM_TWO]);
        DataCopyPad(outGm2[offset], outTensor2, dataCopyParams);
    }
    if (needAtomic) {
        SetAtomicNone();
    }
}

TEMPLATES_DEF
class FAGBlockVecQuantDummy {
public:
    __aicore__ inline void InitUbBuffer(const LocalTensor<CALC_TYPE> &commonTensors){};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR value, GM_ADDR dy, GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                            GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy,
                                            GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, int64_t sfmgWorkSpaceOffset){};
    __aicore__ inline void SetVecBlockParams(TPipe *pipe, FagTilingType tilingData, uint32_t vBlockIdx,
                                             uint32_t cBlockIdx, uint32_t vSubBlockIdx, FagConstInfo &constInfo){};
    __aicore__ inline void CopyUB2L1(FagRunInfo &runInfo, const LocalTensor<INPUT_TYPE> &dstTensor,
                                     const LocalTensor<CALC_TYPE> &srcTensor, uint16_t realS1, uint16_t realS2){};
    __aicore__ inline void CopyMaxSumD(FagConstInfo &constInfo, FagRunInfo &runInfo, uint16_t firstHalfS1, uint16_t currentRealS1){};
    __aicore__ inline void ProcessPDs(const LocalTensor<INPUT_TYPE> &pl1Tensor, const LocalTensor<INPUT_TYPE> &dsl1Tensor, const LocalTensor<CALC_TYPE> &spTensor, const LocalTensor<CALC_TYPE> &dpdsTensor, int8_t sdpId, FagConstInfo &constInfo, FagRunInfo &runInfo){};
    template<uint64_t mode>
    __aicore__ inline void DequantDqkv(const LocalTensor<CALC_TYPE> &outTensor, const LocalTensor<CALC_TYPE> &outTensor2, FagConstInfo &constInfo, FagRunInfo &runInfo, int64_t i, int64_t id){};
    __aicore__ inline void DequantOut(const LocalTensor<CALC_TYPE> &outTensor, uint16_t realLength, float scale){};
    template<uint64_t mode>
    __aicore__ inline void CopyDqkv2GM(FagRunInfo &runInfo, FagConstInfo &constInfo, const GlobalTensor<CALC_TYPE> &outGm, const LocalTensor<CALC_TYPE> &outTensor, const GlobalTensor<CALC_TYPE> &outGm2, const LocalTensor<CALC_TYPE> &outTensor2){};
    __aicore__ inline void AllocEventID(){};
    __aicore__ inline void FreeEventID(){};
};
 
} // namespace FagBaseApi
 
#endif
