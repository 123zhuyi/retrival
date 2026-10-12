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
 * \file flash_attention_score_grad_s1s2_bn2_regbase.h
 * \brief
 */
#ifndef FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2_REGBASE_H_
#define FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2_REGBASE_H_

#include "../../../common/op_kernel/arch35/dropmask.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "matmul_modules/fag_custom_matmul_policy.h"
#include "vector_api/cast_softmax_grad.h"
#include "vector_api/dropout.h"
#include "vector_api/pse_atten_mask_muls_simple_softmax.h"
#include "vector_api/vf_broadcast_sub_mul.h"
#include "vector_api/vf_cast_transdata_deconflict.h"
#include "matmul_modules/matmul_config.h"
#include "flash_attention_score_grad_tiling_data_regbase.h"

#define FAG_BN2_CLASS_TEMPLATE                                                                                             \
    template <typename T1, typename T2, const bool IS_ATTEN_MASK = 0, const bool IS_PSE = 0, const bool IS_DROP = 0,   \
              const bool IS_TND = 0, const bool HAS_TAIL = 0, const uint8_t DETER_SPARSE_TYPE = 0, const bool IS_D_NO_EQUAL = 0,   \
              const uint8_t SPLIT_AXIS = 0, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,                \
              S2TemplateType s2TemplateType = S2TemplateType::Aligned128,                                              \
              DTemplateType dTemplateType = DTemplateType::Aligned128>
#define FAG_BN2_FUNCTION_TEMPLATE                                                                                          \
    template <typename T1, typename T2, const bool IS_ATTEN_MASK, const bool IS_PSE, const bool IS_DROP,               \
              const bool IS_TND, const bool HAS_TAIL, const uint8_t DETER_SPARSE_TYPE, const bool IS_D_NO_EQUAL,                   \
              const uint8_t SPLIT_AXIS, S1TemplateType s1TemplateType, S2TemplateType s2TemplateType,                  \
              DTemplateType dTemplateType>
#define FAG_BN2_FUNCTION_PARAMS_TEMPLATE                                                                                   \
    T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, HAS_TAIL, DETER_SPARSE_TYPE, IS_D_NO_EQUAL, SPLIT_AXIS, s1TemplateType,     \
        s2TemplateType, dTemplateType

using namespace optiling::fag;

__aicore__ constexpr bool GetBN2PolicySwitch(const bool IS_TSCM_REUSE)
{
    return IS_TSCM_REUSE;
}

__aicore__ constexpr bool GetBN2TscmPreloadSwitch(const bool IS_TSCM_PRELOAD)
{
    return IS_TSCM_PRELOAD;
}

FAG_BN2_CLASS_TEMPLATE
class FlashAttentionScoreGradUs1s2Bbn2StaticRegbase {
public:
    __aicore__ inline FlashAttentionScoreGradUs1s2Bbn2StaticRegbase(){};

    __aicore__ inline void Init(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query,
                                __gm__ uint8_t *pseShift, __gm__ uint8_t *dropMask, __gm__ uint8_t *attenMask,
                                __gm__ uint8_t *y, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                __gm__ uint8_t *prefixN, __gm__ uint8_t *actualSeqQlen, __gm__ uint8_t *actualSeqKvlen,
                                __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dpse,
                                __gm__ uint8_t *workspace,
                                FagOldTilingType ordTilingData,
                                TPipe *pipeIn, TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> &dsScmIn,
                                TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> &pScmIn);
    __aicore__ inline void SetConstInfo();
    __aicore__ inline void SetRunInfo(FagRunInfo &runInfo, int64_t taskId, int64_t index);
    __aicore__ inline void SetOptionalInfo();
    __aicore__ inline void InitUbBuffer();
    __aicore__ inline void InitWorkSpace(__gm__ uint8_t *workspace, __gm__ uint8_t *dq, __gm__ uint8_t *dk,
                                         __gm__ uint8_t *dv);
    __aicore__ inline void Process();
    __aicore__ inline bool IsValid(FagRunInfo &runInfo, int64_t index);
    __aicore__ inline bool CheckIsValidBlock(FagRunInfo &runInfo, int64_t baseIdx, int64_t s1oDimIdx,
                                             int64_t s2oDimIdx);
    __aicore__ inline void IterateMm1Mm2(FagRunInfo &runInfo);      // qk dxv
    __aicore__ inline void ProcessSoftmaxGrad(FagRunInfo &runInfo); // softmaxGrad
    __aicore__ inline void WaitMm1Mm2Result();
    __aicore__ inline int64_t GetQueryOrDxOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetKeyOrValueOffset(FagRunInfo &runInfo);
    __aicore__ inline void ProcessReCompute(FagRunInfo &runInfo);
    __aicore__ inline void IterateMm3Mm4Mm5(FagRunInfo &runInfo); // dq dk dv
    __aicore__ inline void IterateMm3(FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer, int64_t dxOrQueryGmOffset,
                                      int64_t keyOrValueGmOffset);
    __aicore__ inline void IterateMm4(FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer, int64_t dxOrQueryGmOffset,
                                      int64_t keyOrValueGmOffset);
    __aicore__ inline void IterateMm5(FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer, int64_t dxOrQueryGmOffset,
                                      int64_t keyOrValueGmOffset);
    __aicore__ inline void CopyUB2L1(FagRunInfo &runInfo, LocalTensor<T1> &dstTensor, LocalTensor<T1> &srcTensor);
    __aicore__ inline void SyncALLCores();
    __aicore__ inline void GetSeqQlenKvlenByBidx(int64_t bIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvlen);
    __aicore__ inline void MulsCast(const LocalTensor<T1> &dstTensor, const LocalTensor<T2> &srcTensor,
                                    uint32_t srcM, float scaleValue, uint32_t realN);
    __aicore__ static constexpr CubeFormat GetC2CubeFormat(TPosition pos) {
        if (pos == TPosition::VECCALC) {
            return CubeFormat::ND_ALIGN;
        } else {
            return CubeFormat::ND;
        }
    };

    constexpr static uint32_t CUBE_BASEM = (uint32_t)s1TemplateType;
    constexpr static uint32_t CUBE_BASEN = (uint32_t)s2TemplateType;
    constexpr static uint32_t HEAD_DIM_ALIGN = (uint32_t)dTemplateType;
    constexpr static uint32_t VECTOR_BASEM = CUBE_BASEM / CV_CORE_RATIO;
    constexpr static uint32_t VECTOR_BASEN = CUBE_BASEN;
    constexpr static uint32_t INPUT_BLOCK_NUM = 32 / sizeof(T1);
    constexpr static uint32_t BASE_DQ_SIZE = CUBE_BASEM * HEAD_DIM_ALIGN;
    constexpr static uint32_t FRACTAL_NZ_C0_SIZE = 32 / sizeof(T1);
    constexpr static bool IS_MM3_L0_EXCEED = IS_L0_EXCEED(CUBE_BASEM, HEAD_DIM_ALIGN, CUBE_BASEN, T1);
    constexpr static uint32_t MM1_LEFT_BASE_RATIO = BN2CeilConst(CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(T1), L0_MAX_SIZE);
    constexpr static uint32_t MM1_RIGHT_BASE_RATIO =
        BN2CeilConst(CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(T1), L0_MAX_SIZE);
    constexpr static uint32_t MM1_MAX_BASE_RATIO =
        MM1_LEFT_BASE_RATIO > MM1_RIGHT_BASE_RATIO ? MM1_LEFT_BASE_RATIO : MM1_RIGHT_BASE_RATIO;

    constexpr static uint32_t MM2_LEFT_BASE_RATIO = BN2CeilConst(CUBE_BASEM * CUBE_BASEN * sizeof(T1), L0_MAX_SIZE);
    constexpr static uint32_t MM2_RIGHT_BASE_RATIO =
        BN2CeilConst(CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(T1), L0_MAX_SIZE);
    constexpr static uint32_t MM2_MAX_BASE_RATIO =
        MM2_LEFT_BASE_RATIO > MM2_RIGHT_BASE_RATIO ? MM2_LEFT_BASE_RATIO : MM2_RIGHT_BASE_RATIO;

    constexpr static uint32_t MM3_LEFT_BASE_RATIO = BN2CeilConst(CUBE_BASEM * CUBE_BASEN * sizeof(T1), L0_MAX_SIZE);
    constexpr static uint32_t MM3_RIGHT_BASE_RATIO =
        BN2CeilConst(CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(T1), L0_MAX_SIZE);
    constexpr static uint32_t MM3_MAX_BASE_RATIO =
        MM3_LEFT_BASE_RATIO > MM3_RIGHT_BASE_RATIO ? MM3_LEFT_BASE_RATIO : MM3_RIGHT_BASE_RATIO;

    // 开启c1Shared后，确保每一块L0C大小够用
    constexpr static uint32_t SHARED_C1_BUFFER_SZIE = GET_SHARED_C1_BUFFER_SZIE(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN);
    constexpr static bool IS_TSCM_REUSE = IS_TSCM_REUSE(HEAD_DIM_ALIGN, T1, IS_DETER_OLD(DETER_SPARSE_TYPE), false);
    constexpr static bool IS_L0DB =
        (HEAD_DIM_ALIGN <= (uint32_t)DTemplateType::Aligned128 && !IsSameType<T1, float>::value);
    constexpr static bool IS_TSCM_PRELOAD = IS_TSCM_PRELOAD(HEAD_DIM_ALIGN, T1, SPLIT_AXIS, IS_DETER_OLD(DETER_SPARSE_TYPE), IS_TND);
    constexpr static MatmulConfig MM1_CFG_SAMEAB =
        GetBN2Mm1Cfg<T1>(IS_ATTEN_MASK, IS_PSE, IS_DROP, HAS_TAIL, CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN,
                         IS_TSCM_REUSE, IS_L0DB, SHARED_C1_BUFFER_SZIE, MM1_MAX_BASE_RATIO);
    constexpr static MatmulConfig MM2_CFG_SAMEB =
        GetBN2Mm2Cfg<T1>(IS_ATTEN_MASK, IS_PSE, IS_DROP, HAS_TAIL, CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN,
                         IS_TSCM_REUSE, IS_L0DB, SHARED_C1_BUFFER_SZIE, MM2_MAX_BASE_RATIO);
    constexpr static MatmulConfig MM3_CFG_SAMEB =
        GetBN2Mm3Cfg<T1>(IS_ATTEN_MASK, IS_PSE, IS_DROP, HAS_TAIL, CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN,
                         IS_TSCM_REUSE, IS_L0DB, SHARED_C1_BUFFER_SZIE, MM3_MAX_BASE_RATIO);

    using aType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true, LayoutMode::NONE, true>;
    using bType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true, LayoutMode::NONE, true>;
    using cType1 = MatmulType<TPosition::VECCALC, CubeFormat::ND_ALIGN, T2>;
    using biasType1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using aType2 = MatmulType<TPosition::TSCM, CubeFormat::NZ, T1, true, LayoutMode::NONE, true, TPosition::VECOUT>;
    using bType2 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true, LayoutMode::NONE, true>;
    using cType2Dqk = MatmulType<GetC2Position(dTemplateType), GetC2CubeFormat(GetC2Position(dTemplateType)), T2>;
    using cType2Dv = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
    using biasType2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    constexpr static auto MM1_TILING_CFG_SAMEAB = GetMatmulApiTiling<aType1, bType1, cType1, biasType1>(MM1_CFG_SAMEAB);
    constexpr static auto MM2_TILING_CFG_SAMEB =
        GetMatmulApiTiling<aType2, bType2, cType2Dqk, biasType2>(MM2_CFG_SAMEB);
    constexpr static auto MM3_TILING_CFG_SAMEB = GetMatmulApiTiling<aType2, bType2, cType2Dv, biasType2>(MM3_CFG_SAMEB);
    Matmul<aType1, bType1, cType1, biasType1, MM1_TILING_CFG_SAMEAB,
           matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
           Mm1ConstPolicyBN2Selector<GetBN2PolicySwitch(IS_TSCM_REUSE),
                                     GetBN2TscmPreloadSwitch(IS_TSCM_PRELOAD)>::template Result>
        mm1;
    Matmul<aType2, bType2, cType2Dqk, biasType2, MM2_TILING_CFG_SAMEB,
           matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
           Mm3ConstPolicyBN2Selector<GetBN2PolicySwitch(IS_TSCM_REUSE),
                                     GetBN2TscmPreloadSwitch(IS_TSCM_PRELOAD)>::template Result>
        mm2;
    Matmul<aType2, bType2, cType2Dv, biasType2, MM3_TILING_CFG_SAMEB,
           matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
           Mm3ConstPolicyBN2Selector<GetBN2PolicySwitch(IS_TSCM_REUSE),
                                     GetBN2TscmPreloadSwitch(IS_TSCM_PRELOAD)>::template Result>
        mm3;

protected:
    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> attenMaskOrYInQue;
    TQue<QuePosition::VECIN, 1> pseOrDyInQue;
    TQue<QuePosition::VECOUT, 1> dSOutQue;
    TQue<QuePosition::VECOUT, 1> pOutQue;
    TQue<QuePosition::VECIN, 1> mm1ResInQue[2];
    TQue<QuePosition::VECIN, 1> mm2ResInQue[2];
    TQue<QuePosition::VECIN, 1> maxSumQue[2];
    TBuf<> softmaxGradResBuf;
    TBuf<> dropMaskBuf;
    TBuf<> dropmaskIndexVecBuf;
    TQue<QuePosition::VECIN, 1> dropMaskInQue;
    __gm__ uint8_t *prefixNAddr;
    __gm__ uint8_t *actualSeqQlenAddr;
    __gm__ uint8_t *actualSeqKvlenAddr;

    uint32_t vBlockIdx;
    uint32_t cBlockIdx;
    uint32_t vSubBlockIdx;
    int64_t lastS2oCvDimIdx = -1; // 上一次的s2方向基本块idx
    int64_t lastBdimIdx = -1;     // 上一次的b方向基本块idx
    int64_t lastN2dimIdx = -1;    // 上一次的n2方向基本块idx
    uint8_t kvPingPong = 1;
    bool isLastLoop = false;

    FagOldTilingType tilingData;
    // input
    GlobalTensor<T1> keyGm, valueGm, dxGm, queryGm, yGm, pseGm;
    GlobalTensor<uint8_t> dropMaskGm, attenMaskU8Gm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm, pseFloatGm;
    __gm__ uint8_t *pseSlope;
    // output
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
    GlobalTensor<T1> dqGm, dkGm, dvGm;
    TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> dsScm;
    TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> pScm;

    FagConstInfo constInfo;
    AttenMaskInfo attenMaskInfo;
    PseInfo pseInfo;
    DropMaskInfo dropInfo;
};

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::Init(
    __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *dropMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *y, __gm__ uint8_t *softmaxMax,
    __gm__ uint8_t *softmaxSum, __gm__ uint8_t *prefixN, __gm__ uint8_t *actualSeqQlen, __gm__ uint8_t *actualSeqKvlen,
    __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dpse, __gm__ uint8_t *workspace,
    FagOldTilingType ordTilingData, TPipe *pipeIn,
    TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> &dsScmIn, TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> &pScmIn)
{
    keyGm.SetGlobalBuffer((__gm__ T1 *)key);
    valueGm.SetGlobalBuffer((__gm__ T1 *)value);
    dxGm.SetGlobalBuffer((__gm__ T1 *)dx);
    queryGm.SetGlobalBuffer((__gm__ T1 *)query);
    yGm.SetGlobalBuffer((__gm__ T1 *)y);
    pseGm.SetGlobalBuffer((__gm__ T1 *)pseShift);
    pseFloatGm.SetGlobalBuffer((__gm__ float *)pseShift);
    dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)dropMask);
    attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    dqGm.SetGlobalBuffer((__gm__ T1 *)dq);
    dkGm.SetGlobalBuffer((__gm__ T1 *)dk);
    dvGm.SetGlobalBuffer((__gm__ T1 *)dv);
    pseSlope = pseShift;
    dsScm = dsScmIn;
    pScm = pScmIn;

    // init current core tilingInfo
    vBlockIdx = GetBlockIdx();
    cBlockIdx = vBlockIdx / CV_CORE_RATIO;
    vSubBlockIdx = GetSubBlockIdx();
    tilingData = ordTilingData;
    pipe = pipeIn;

    // 填充基础参数
    SetConstInfo();

    prefixNAddr = prefixN;
    actualSeqQlenAddr = actualSeqQlen;
    actualSeqKvlenAddr = actualSeqKvlen;
    constInfo.seqS1_addr = actualSeqQlen;
    constInfo.seqS2_addr = actualSeqKvlen;

    InitWorkSpace(workspace, dq, dk, dv);
    InitUbBuffer();
    SetOptionalInfo();
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::GetSeqQlenKvlenByBidx(
    int64_t bIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvlen)
{
    if (unlikely(bIdx == 0)) {
        actualSeqQlen = ((__gm__ int64_t *)actualSeqQlenAddr)[0];
        actualSeqKvlen = ((__gm__ int64_t *)actualSeqKvlenAddr)[0];
    } else {
        actualSeqQlen = ((__gm__ int64_t *)actualSeqQlenAddr)[bIdx] - ((__gm__ int64_t *)actualSeqQlenAddr)[bIdx - 1];
        actualSeqKvlen =
            ((__gm__ int64_t *)actualSeqKvlenAddr)[bIdx] - ((__gm__ int64_t *)actualSeqKvlenAddr)[bIdx - 1];
    }
    return;
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::InitWorkSpace(
    __gm__ uint8_t *workspace, __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv)
{
    // init workspace address
    if constexpr (!IsSameType<T1, float>::value) {
        uint64_t qPostBlockTotal = CUBE_BASEM * HEAD_DIM_ALIGN * MAX_CUBE_CORE_NUM;
        uint64_t kPostBlockTotal = CUBE_BASEN * HEAD_DIM_ALIGN * MAX_CUBE_CORE_NUM;
        uint64_t workspaceOffsets = RESERVED_WORKSPACE_SIZE;
        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + workspaceOffsets / sizeof(T2));
        workspaceOffsets = workspaceOffsets + qPostBlockTotal * sizeof(float);
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + workspaceOffsets / sizeof(T2));
        workspaceOffsets = workspaceOffsets + kPostBlockTotal * sizeof(float);
    } else {
        // input type fp32, dq dk dv write to output gm directly
        dqWorkSpaceGm.SetGlobalBuffer((__gm__ T1 *)dq);
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ T1 *)dk);
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ T1 *)dv);
    }
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::InitUbBuffer()
{
    /**
     * UB划分，buffer大小分配
     * attenMaskOrYInQue: for y and attenMask
     * pseOrDyInQue: for dx and pse
     * dSOutQue: for dq dk left ub matrix
     * pOutQue: for dv left ub matrix
     * mm1ResInQue: for mm1 ub double buffer
     * mm2ResInQue: for mm2 ub double buffer
     * softmaxGradResBuf: for softmax_grad result
     * dropMaskBuf: for dropMask
     * maxSumQue: for max sum double buffer
     **/
    pipe->InitBuffer(attenMaskOrYInQue, 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(T2));
    pipe->InitBuffer(pseOrDyInQue, 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(T1));
    pipe->InitBuffer(dSOutQue, 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(T1) + VECTOR_BASEN * sizeof(T1));
    pipe->InitBuffer(mm1ResInQue[0], 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(T2));
    pipe->InitBuffer(mm1ResInQue[1], 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(T2));
    pipe->InitBuffer(mm2ResInQue[0], 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(T2));
    pipe->InitBuffer(mm2ResInQue[1], 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(T2));
    pipe->InitBuffer(softmaxGradResBuf, VECTOR_BASEM * sizeof(T2));
    pipe->InitBuffer(maxSumQue[0], 1, VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * 2);
    pipe->InitBuffer(maxSumQue[1], 1, VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * 2);
    if constexpr (IS_DROP) {
        pipe->InitBuffer(dropMaskBuf, VECTOR_BASEM * VECTOR_BASEN * sizeof(uint8_t) / 8);          // 1k
        pipe->InitBuffer(dropmaskIndexVecBuf, VECTOR_BASEM * VECTOR_BASEN / 16 * sizeof(int32_t)); // 2k
        if (tilingData->s1s2BNGS1S2BaseParams.dropMaskOuter == 1) {
            pipe->InitBuffer(dropMaskInQue, 1, 8192);
        }
    }
    if constexpr (!IsSameType<T1, float>::value) {
        pipe->InitBuffer(pOutQue, 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(T1) + VECTOR_BASEN * sizeof(T1));
    } else {
        // input type fp32, exceed ub size so need to reuse dSOutQue
        pOutQue = dSOutQue;
    }
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::SetConstInfo()
{
    constInfo.s1Token = tilingData->s1s2BNGS1S2BaseParams.s1Token;
    constInfo.s2Token = tilingData->s1s2BNGS1S2BaseParams.s2Token;
    constInfo.sparseMode = tilingData->s1s2BNGS1S2BaseParams.sparseMode;
    // split info
    constInfo.s1Outer = tilingData->s1s2BNGS1S2SplitCoreParams.s1Outer;
    constInfo.s1CvTail = tilingData->s1s2BNGS1S2SplitCoreParams.s1CvTail;
    constInfo.s1Tail = tilingData->s1s2BNGS1S2SplitCoreParams.s1Tail;
    constInfo.s2Tail = tilingData->s1s2BNGS1S2SplitCoreParams.s2Tail;
    constInfo.s2Outer = tilingData->s1s2BNGS1S2SplitCoreParams.s2Outer;

    constInfo.commonConstInfo.s1BaseSize = CUBE_BASEM;
    constInfo.commonConstInfo.s2BaseSize = CUBE_BASEN;
    constInfo.bSize = tilingData->s1s2BNGS1S2BaseParams.b;
    constInfo.n2Size = tilingData->s1s2BNGS1S2BaseParams.n2;
    constInfo.commonConstInfo.gSize = tilingData->s1s2BNGS1S2BaseParams.g;
    constInfo.commonConstInfo.s1Size = tilingData->s1s2BNGS1S2BaseParams.s1;
    constInfo.commonConstInfo.s2Size = tilingData->s1s2BNGS1S2BaseParams.s2;
    constInfo.commonConstInfo.dSize = tilingData->s1s2BNGS1S2BaseParams.d;
    constInfo.commonConstInfo.dSizeV = tilingData->s1s2BNGS1S2BaseParams.d1;
    constInfo.commonConstInfo.layoutType = tilingData->s1s2BNGS1S2BaseParams.layout;

    constInfo.commonConstInfo.s1D = constInfo.commonConstInfo.s1Size * constInfo.commonConstInfo.dSize;
    constInfo.commonConstInfo.gS1D = constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.s1D;
    constInfo.commonConstInfo.n2GS1D = constInfo.n2Size * constInfo.commonConstInfo.gS1D;
    constInfo.commonConstInfo.s2D = constInfo.commonConstInfo.s2Size * constInfo.commonConstInfo.dSize;
    constInfo.commonConstInfo.n2S2D = constInfo.n2Size * constInfo.commonConstInfo.s2D;
    constInfo.commonConstInfo.s1S2 = constInfo.commonConstInfo.s1Size * constInfo.commonConstInfo.s2Size;
    constInfo.commonConstInfo.gS1 = constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.s1Size;
    constInfo.commonConstInfo.gD = constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.dSize;
    constInfo.commonConstInfo.n2D = constInfo.n2Size * constInfo.commonConstInfo.dSize;
    constInfo.commonConstInfo.bN2D = constInfo.bSize * constInfo.commonConstInfo.n2D;
    constInfo.commonConstInfo.n2G = constInfo.n2Size * constInfo.commonConstInfo.gSize;
    constInfo.commonConstInfo.n2GD = constInfo.commonConstInfo.n2G * constInfo.commonConstInfo.dSize;
    constInfo.commonConstInfo.bN2GD = constInfo.bSize * constInfo.commonConstInfo.n2GD;
    constInfo.commonConstInfo.gS2 = constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.s2Size;

    // for D_V
    if constexpr (IS_D_NO_EQUAL) {
        constInfo.commonConstInfo.s1Dv = constInfo.commonConstInfo.s1Size * constInfo.commonConstInfo.dSizeV;
        constInfo.commonConstInfo.gS1Dv = constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.s1Dv;
        constInfo.commonConstInfo.n2GS1Dv = constInfo.n2Size * constInfo.commonConstInfo.gS1Dv;
        constInfo.commonConstInfo.s2Dv = constInfo.commonConstInfo.s2Size * constInfo.commonConstInfo.dSizeV;
        constInfo.commonConstInfo.n2S2Dv = constInfo.n2Size * constInfo.commonConstInfo.s2Dv;
        constInfo.commonConstInfo.gDv = constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.dSizeV;
        constInfo.commonConstInfo.n2Dv = constInfo.n2Size * constInfo.commonConstInfo.dSizeV;
        constInfo.commonConstInfo.bN2Dv = constInfo.bSize * constInfo.commonConstInfo.n2Dv;
        constInfo.commonConstInfo.n2GDv = constInfo.commonConstInfo.n2G * constInfo.commonConstInfo.dSizeV;
        constInfo.commonConstInfo.bN2GDv = constInfo.bSize * constInfo.commonConstInfo.n2GDv;
    } else {
        constInfo.commonConstInfo.s1Dv = constInfo.commonConstInfo.s1D;
        constInfo.commonConstInfo.gS1Dv = constInfo.commonConstInfo.gS1D;
        constInfo.commonConstInfo.n2GS1Dv = constInfo.commonConstInfo.n2GS1D;
        constInfo.commonConstInfo.s2Dv = constInfo.commonConstInfo.s2D;
        constInfo.commonConstInfo.n2S2Dv = constInfo.commonConstInfo.n2S2D;
        constInfo.commonConstInfo.gDv = constInfo.commonConstInfo.gD;
        constInfo.commonConstInfo.n2Dv = constInfo.commonConstInfo.n2D;
        constInfo.commonConstInfo.bN2Dv = constInfo.commonConstInfo.bN2D;
        constInfo.commonConstInfo.n2GDv = constInfo.commonConstInfo.n2GD;
        constInfo.commonConstInfo.bN2GDv = constInfo.commonConstInfo.bN2GD;
    }

    constInfo.scaleValue = tilingData->s1s2BNGS1S2BaseParams.scaleValue;
    constInfo.n2GS1oS2o = constInfo.commonConstInfo.n2G * constInfo.s1Outer * constInfo.s2Outer;
    constInfo.gS1oS2o = constInfo.commonConstInfo.gSize * constInfo.s1Outer * constInfo.s2Outer;
    constInfo.s1oS2o = constInfo.s1Outer * constInfo.s2Outer;

    if constexpr (IS_TND) {
        constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.n2GD;
        constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.n2D;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.dSize;
            constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.dSize;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.bN2GD;
            constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.bN2D;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.n2GD;
            constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.n2D;
        }
    }
    constInfo.commonConstInfo.subBlockIdx = vSubBlockIdx;

    uint32_t tmp = 0xFF7FFFFF;
    constInfo.attenMaskMinValue = *((float *)&tmp);
    constInfo.commonConstInfo.keepProb = tilingData->s1s2BNGS1S2BaseParams.keepProb;

    constInfo.sfmgMaxLoopSize =
        Ceil<uint32_t>(VECTOR_BASEM * VECTOR_BASEN, HEAD_DIM_ALIGN); // softmaxGrad每次最大能处理的m轴大小
    constInfo.dAlignToBlock = AlignTo(constInfo.commonConstInfo.dSize, INPUT_BLOCK_NUM);
    constInfo.dAlignToBlockForFp8 = AlignTo(constInfo.commonConstInfo.dSize, INPUT_BLOCK_NUM);
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::SetOptionalInfo()
{
    if constexpr (IS_ATTEN_MASK) {
        attenMaskInfo.attenMaskShapeType = tilingData->s1s2BNGS1S2BaseParams.attenMaskShapeType;
        attenMaskInfo.compressMode = tilingData->s1s2BNGS1S2BaseParams.attenMaskCompressMode;
        attenMaskInfo.attenMaskS2Size = tilingData->s1s2BNGS1S2BaseParams.attenMaskS2Size;
        attenMaskInfo.preTokens = tilingData->s1s2BNGS1S2BaseParams.s1Token;
        attenMaskInfo.nextTokens = tilingData->s1s2BNGS1S2BaseParams.s2Token;
        attenMaskInfo.prefixNAddr = prefixNAddr;
        attenMaskInfo.bandIndex = tilingData->s1s2BNGS1S2SplitCoreParams.bandIdx;
    }

    if constexpr (IS_PSE) {
        uint32_t pseShapeType = tilingData->s1s2BNGS1S2BaseParams.pseShapeType;
        pseInfo.pseBSize =
            (pseShapeType == PSE_SHAPE_TYPE_1NSS || pseShapeType == PSE_SHAPE_TYPE_1NHS) ? 1 : constInfo.bSize;
        pseInfo.pseS1Size = PSE_COMPRESS_H; // 1024
        pseInfo.pseS2Size = constInfo.commonConstInfo.s2Size;
        pseInfo.pseLayoutType = tilingData->s1s2BNGS1S2BaseParams.pseLayoutType;
        pseInfo.pseEncodeType =
            (pseShapeType == PSE_SHAPE_TYPE_BNHS || pseShapeType == PSE_SHAPE_TYPE_1NHS) ? pseEncodeALibiS2Full : 0;
        pseInfo.pseType = tilingData->s1s2BNGS1S2BaseParams.pseType;
        pseInfo.qStartIdx = tilingData->s1s2BNGS1S2BaseParams.qStartIdx;
        pseInfo.kvStartIdx = tilingData->s1s2BNGS1S2BaseParams.kvStartIdx;
    }

    if constexpr (IS_DROP) {
        dropInfo.seed = tilingData->s1s2BNGS1S2BaseParams.seed;
        dropInfo.offset = tilingData->s1s2BNGS1S2BaseParams.offset;
        dropInfo.keepProbUint8 = static_cast<uint8_t>(tilingData->s1s2BNGS1S2BaseParams.keepProbUint8);
        dropInfo.dropMaskOuter = tilingData->s1s2BNGS1S2BaseParams.dropMaskOuter;
        dropInfo.boolMode = tilingData->preTilingData.dropoutIsDivisibleBy8 == 0 ? true : false;
    }
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::SetRunInfo(FagRunInfo &runInfo,
                                                                                      int64_t taskId, int64_t index)
{
    runInfo.qDxPingPongIdx = taskId & 1;
    if constexpr (IS_TND) {
        int64_t resbaseIdx = index - runInfo.lastBatchTotalBaseIdx;
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        for (int64_t bIdx = runInfo.lastBatchIdx; bIdx < constInfo.bSize; bIdx++) {
            GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
            int64_t s1OuterTmp = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
            int64_t s2OuterTmp = (actualS2Len + VECTOR_BASEN - 1) / VECTOR_BASEN;
            int64_t totalBaseIdx = constInfo.commonConstInfo.n2G * s1OuterTmp * s2OuterTmp;
            if (resbaseIdx < totalBaseIdx) {
                int64_t s1CvTailTmp = actualS1Len - (s1OuterTmp - 1) * CUBE_BASEM;
                runInfo.commonRunInfo.boIdx = bIdx;
                runInfo.lastBatchIdx = bIdx;
                int64_t bDimTail = resbaseIdx;
                runInfo.commonRunInfo.n2oIdx = bDimTail / (constInfo.commonConstInfo.gSize * s1OuterTmp * s2OuterTmp);
                int64_t n2DimTail = bDimTail % (constInfo.commonConstInfo.gSize * s1OuterTmp * s2OuterTmp);
                runInfo.commonRunInfo.goIdx = n2DimTail / (s1OuterTmp * s2OuterTmp);
                int64_t gDimTail = n2DimTail % (s1OuterTmp * s2OuterTmp);
                runInfo.s2oIdx = gDimTail / s1OuterTmp;
                runInfo.commonRunInfo.s1oIdx = gDimTail % s1OuterTmp;
                runInfo.commonRunInfo.s1RealSize =
                    (runInfo.commonRunInfo.s1oIdx == s1OuterTmp - 1) ? s1CvTailTmp : CUBE_BASEM;
                runInfo.commonRunInfo.taskId = taskId;
                runInfo.commonRunInfo.taskIdMod2 = taskId & 1;
                runInfo.commonRunInfo.s2RealSize = runInfo.s2CvEnd - runInfo.s2CvBegin; // 真实s2基本块大小
                runInfo.halfS2RealSize = (runInfo.commonRunInfo.s2RealSize + 1) >> 1;
                runInfo.firstHalfS2RealSize = runInfo.halfS2RealSize;
                runInfo.commonRunInfo.halfS1RealSize = (runInfo.commonRunInfo.s1RealSize + 1) >> 1;
                runInfo.commonRunInfo.firstHalfS1RealSize = runInfo.commonRunInfo.halfS1RealSize;
                if (vSubBlockIdx == 1) {
                    runInfo.commonRunInfo.halfS1RealSize =
                        runInfo.commonRunInfo.s1RealSize - runInfo.commonRunInfo.halfS1RealSize;
                    runInfo.halfS2RealSize = runInfo.commonRunInfo.s2RealSize - runInfo.halfS2RealSize;
                }
                runInfo.dAlign16 = AlignTo16(constInfo.commonConstInfo.dSize);
                runInfo.s1RealSizeAlign2 = ((runInfo.commonRunInfo.s1RealSize + 1) >> 1 << 1);
                break;
            } else {
                runInfo.lastBatchTotalBaseIdx += totalBaseIdx;
                resbaseIdx = index - runInfo.lastBatchTotalBaseIdx;
                runInfo.lastBatchTotalS1BOffset += actualS1Len * constInfo.commonConstInfo.n2GD;
                runInfo.lastBatchTotalS2BOffset += actualS2Len * constInfo.commonConstInfo.n2D;
                runInfo.lastBatchTotalS1S2SizeAlign += actualS1Len * AlignTo16(actualS2Len);
                runInfo.lastBatchTotalS1S2Size += actualS1Len * actualS2Len;
                runInfo.lastBatchTotalS2Size += actualS2Len;
            }
        }
        GetSeqQlenKvlenByBidx(runInfo.commonRunInfo.boIdx, actualS1Len, actualS2Len);
        runInfo.commonRunInfo.actualS1Size = actualS1Len;
        runInfo.commonRunInfo.actualS2Size = actualS2Len;
        runInfo.commonRunInfo.s2SizeAcc = runInfo.lastBatchTotalS2Size;
        runInfo.commonRunInfo.b1SSOffsetAlign = runInfo.lastBatchTotalS1S2SizeAlign;
        runInfo.commonRunInfo.b1SSOffset = runInfo.lastBatchTotalS1S2Size;
        runInfo.commonRunInfo.b1SSAttenMaskOffset = runInfo.commonRunInfo.b1SSOffset;
        runInfo.commonRunInfo.s2StartIdx = runInfo.s2CvBegin;
        runInfo.commonRunInfo.vecCoreOffset = vSubBlockIdx * runInfo.commonRunInfo.firstHalfS1RealSize;
        runInfo.commonRunInfo.s2AlignedSize = AlignTo16(runInfo.commonRunInfo.s2RealSize);
    } else {
        runInfo.commonRunInfo.boIdx = index / constInfo.n2GS1oS2o;
        int64_t bDimTail = index % constInfo.n2GS1oS2o;
        runInfo.commonRunInfo.n2oIdx = bDimTail / constInfo.gS1oS2o;
        int64_t n2DimTail = bDimTail % constInfo.gS1oS2o;
        runInfo.commonRunInfo.goIdx = n2DimTail / constInfo.s1oS2o;
        int64_t gDimTail = n2DimTail % constInfo.s1oS2o;
        runInfo.s2oIdx = gDimTail / constInfo.s1Outer;
        runInfo.commonRunInfo.s1oIdx = gDimTail % constInfo.s1Outer;
        runInfo.commonRunInfo.s1RealSize =
            (runInfo.commonRunInfo.s1oIdx == constInfo.s1Outer - 1) ? constInfo.s1CvTail : CUBE_BASEM;
        runInfo.commonRunInfo.taskId = taskId;
        runInfo.commonRunInfo.taskIdMod2 = taskId & 1;
        runInfo.commonRunInfo.s2RealSize = runInfo.s2CvEnd - runInfo.s2CvBegin; // 真实s2基本块大小
        runInfo.commonRunInfo.halfS1RealSize = (runInfo.commonRunInfo.s1RealSize + 1) >> 1;
        runInfo.commonRunInfo.firstHalfS1RealSize = runInfo.commonRunInfo.halfS1RealSize;
        runInfo.halfS2RealSize = (runInfo.commonRunInfo.s2RealSize + 1) >> 1;
        runInfo.firstHalfS2RealSize = runInfo.halfS2RealSize;
        if (vSubBlockIdx == 1) {
            runInfo.commonRunInfo.halfS1RealSize =
                runInfo.commonRunInfo.s1RealSize - runInfo.commonRunInfo.halfS1RealSize;
            runInfo.halfS2RealSize = runInfo.commonRunInfo.s2RealSize - runInfo.halfS2RealSize;
        }
        // pse
        runInfo.commonRunInfo.b1SSOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.s1S2;
        runInfo.commonRunInfo.b1SSAttenMaskOffset = runInfo.commonRunInfo.b1SSOffset;
        runInfo.commonRunInfo.s2SizeAcc = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.s2Size;
        runInfo.commonRunInfo.s2StartIdx = runInfo.s2CvBegin;
        runInfo.commonRunInfo.s2AlignedSize = AlignTo16(runInfo.commonRunInfo.s2RealSize);
        runInfo.commonRunInfo.actualS1Size = constInfo.commonConstInfo.s1Size;
        runInfo.commonRunInfo.actualS2Size = constInfo.commonConstInfo.s2Size;
        runInfo.dAlign16 = AlignTo16(constInfo.commonConstInfo.dSize);
        runInfo.commonRunInfo.vecCoreOffset = vSubBlockIdx * runInfo.commonRunInfo.firstHalfS1RealSize;
        runInfo.s1RealSizeAlign2 = ((runInfo.commonRunInfo.s1RealSize + 1) >> 1 << 1);
        runInfo.commonRunInfo.b1SSOffsetAlign = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.s1Size *
                                                AlignTo16(constInfo.commonConstInfo.s2Size);
        runInfo.commonRunInfo.preTokensPerBatch = attenMaskInfo.preTokens;
        runInfo.commonRunInfo.nextTokensPerBatch = attenMaskInfo.nextTokens;
    }
}

template <typename T1, typename T2>
__simd_vf__ inline void MulsCastVF(uint64_t dstLocalInt, uint64_t srcLocalInt, uint32_t srcM, float scaleValue, uint32_t realN, uint32_t realNAlign16)
{
    uint32_t OneN = realN / 2;
    uint32_t ZeroN = realN - OneN;
    static constexpr AscendC::MicroAPI::CastTrait castTraitFp322Fp16Zero = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_ROUND,
    };    
    static constexpr AscendC::MicroAPI::CastTrait castTraitFp322Fp16One = {
        AscendC::MicroAPI::RegLayout::ONE,
        AscendC::MicroAPI::SatMode::SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_ROUND,
    };
    RegTensor<T2> vregSrcZero;
    RegTensor<T2> vregSrcOne;
    RegTensor<T2> vregMulsZero;
    RegTensor<T2> vregMulsOne;
    RegTensor<T1> vregCastZero;
    RegTensor<T1> vregCastOne;
    RegTensor<T1> vregCast;

    MaskReg pregFullExeT2 = CreateMask<T2, MaskPattern::ALL>();
    MaskReg pregFullExeT1 = CreateMask<T1, MaskPattern::ALL>();
    MaskReg pregTailExeZero = UpdateMask<T2>(ZeroN);
    MaskReg pregTailExeOne = UpdateMask<T2>(OneN);
    MaskReg pregTailExeT1 = UpdateMask<T1>(realN);    

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        LoadAlign<T2, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            vregSrcZero, vregSrcOne, ((__ubuf__ T2 *&)srcLocalInt), realNAlign16);
        Muls(vregMulsZero, vregSrcZero, scaleValue, pregTailExeZero);
        Muls(vregMulsOne, vregSrcOne, scaleValue, pregTailExeOne);
        Cast<T1, T2, castTraitFp322Fp16Zero>(vregCastZero, vregMulsZero, pregTailExeZero);
        Cast<T1, T2, castTraitFp322Fp16One>(vregCastOne, vregMulsOne, pregTailExeOne);
        Or((RegTensor<uint16_t> &)vregCast, (RegTensor<uint16_t> &)vregCastZero,
            (RegTensor<uint16_t> &)vregCastOne, pregTailExeT1);
        StoreAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T1 *&)dstLocalInt), vregCast, realNAlign16, pregTailExeT1);
    }
}


FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::MulsCast(const LocalTensor<T1> &dstTensor, 
                                                                                    const LocalTensor<T2> &srcTensor,
                                                                                    uint32_t srcM, float scaleValue, uint32_t realN)
{
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint32_t realNAlign16 = AlignTo16(realN);

    if (realN <= (uint32_t)DTemplateType::Aligned128) {
        MulsCastVF<T1,T2>(dstLocalInt, srcLocalInt, srcM, scaleValue, realN, realNAlign16);
    }
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::Process()
{
    if (tilingData->s1s2BNGS1S2BlockNumList.blockEnds[cBlockIdx] == 0) {
        return;
    }
    int64_t taskId = 0;
    FagRunInfo runInfos[2]; // for ping pong
    for (int64_t blockInnerIdx = tilingData->s1s2BNGS1S2BlockNumList.blockStarts[cBlockIdx];
         blockInnerIdx < tilingData->s1s2BNGS1S2BlockNumList.blockEnds[cBlockIdx] + 1; blockInnerIdx++) {
        isLastLoop = (blockInnerIdx == tilingData->s1s2BNGS1S2BlockNumList.blockEnds[cBlockIdx]);
        // 无效块跳过
        if (!isLastLoop && !IsValid(runInfos[taskId & 1], blockInnerIdx)) {
            continue;
        }
        if (taskId > 0) {
            WaitMm1Mm2Result();
            ProcessSoftmaxGrad(runInfos[(taskId + 1) & 1]); // softmaxGrad
        }
        if (!isLastLoop) {
            SetRunInfo(runInfos[taskId & 1], taskId, blockInnerIdx);
            IterateMm1Mm2(runInfos[taskId & 1]);
            CopyInMaxSum<T2, VECTOR_BASEM>(constInfo, runInfos[taskId & 1], maxSumQue[taskId & 1], softmaxMaxGm,
                                           softmaxSumGm);
        }
        if (taskId > 0) {
            ProcessReCompute(runInfos[(taskId + 1) & 1]);
            IterateMm3Mm4Mm5(runInfos[(taskId + 1) & 1]);
        }
        taskId++;
    }
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::WaitMm1Mm2Result()
{
    mm1.WaitIterateAll();
    mm1.WaitIterateAll();
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::GetQueryOrDxOffset(FagRunInfo &runInfo)
{
    int64_t leftMatrixOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    if constexpr (IS_TND) {
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        bOffset = runInfo.lastBatchTotalS1BOffset;
        s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GD;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gD;
        gOffset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.dSize;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GS1D;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gS1D;
            gOffset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.s1D;
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.commonConstInfo.dSize;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.commonConstInfo.bN2GD;
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GD;
            n2Offset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.dSize;
            gOffset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gD;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GS1D;
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GD;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gD;
            gOffset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.dSize;
        }
    }
    leftMatrixOffset = bOffset + n2Offset + gOffset + s1Offset;
    return leftMatrixOffset;
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::GetKeyOrValueOffset(FagRunInfo &runInfo)
{
    int64_t rightMatrixOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    if constexpr (IS_TND) {
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        bOffset = runInfo.lastBatchTotalS2BOffset;
        s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.n2D;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.dSize;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2S2D;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.s2D;
            s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.dSize;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.bN2D;
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2D;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.dSize;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2S2D;
            s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.n2D;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.dSize;
        }
    }
    rightMatrixOffset = bOffset + n2Offset + s2Offset;
    return rightMatrixOffset;
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::IterateMm1Mm2(FagRunInfo &runInfo)
{
    // /////////////////////////////////////////////////////////////
    // MM1: dx@v
    // /////////////////////////////////////////////////////////////
    int64_t dxOrQueryGmOffset = GetQueryOrDxOffset(runInfo);
    int64_t keyOrValueGmOffset = GetKeyOrValueOffset(runInfo);
    if constexpr (IS_TND) {
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        GetSeqQlenKvlenByBidx(runInfo.commonRunInfo.boIdx, actualS1Len, actualS2Len);
        mm1.SetOrgShape(actualS1Len, actualS2Len, constInfo.commonConstInfo.mm1Ka, constInfo.commonConstInfo.mm1Kb,
                        CUBE_BASEN);
    } else {
        mm1.SetOrgShape(constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.s2Size,
                        constInfo.commonConstInfo.mm1Ka, constInfo.commonConstInfo.mm1Kb, CUBE_BASEN);
    }
    FagTscmFlagData flag{0};
    // loop reuse
    if (lastS2oCvDimIdx == runInfo.s2oIdx && lastBdimIdx == runInfo.commonRunInfo.boIdx &&
        lastN2dimIdx == runInfo.commonRunInfo.n2oIdx) {
        flag.kvNeedCopy = 0;
    } else {
        lastS2oCvDimIdx = runInfo.s2oIdx;
        lastBdimIdx = runInfo.commonRunInfo.boIdx;
        lastN2dimIdx = runInfo.commonRunInfo.n2oIdx;
        flag.kvNeedCopy = 1;
        kvPingPong = 1 - kvPingPong;
    }
    runInfo.kvPingPong = kvPingPong;
    flag.leftMatrixEncodingTableIdx = GET_Q_DX_ENCODING_TABLE_IDX(0, runInfo.qDxPingPongIdx);
    flag.rightMatrixEncodingTableIdx = GET_K_V_ENCODING_TABLE_IDX(0, 0);
    mm1.SetSelfDefineData(flag);
    // SingleM不支持设置奇数
    if constexpr (HAS_TAIL) {
        mm1.SetTail(runInfo.commonRunInfo.s1RealSize, runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.dSize);
    }
    mm1.SetTensorA(dxGm[dxOrQueryGmOffset]);
    mm1.SetTensorB(valueGm[keyOrValueGmOffset], true);
    LocalTensor<T2> mm1ResQueInTensor = mm1ResInQue[runInfo.commonRunInfo.taskIdMod2].AllocTensor<T2>();
    mm1.template IterateAll<false>(mm1ResQueInTensor, 0, false, true);
    mm1ResInQue[runInfo.commonRunInfo.taskIdMod2].FreeTensor(mm1ResQueInTensor);

    // /////////////////////////////////////////////////////////////
    // MM2: q@k
    // /////////////////////////////////////////////////////////////
    flag.leftMatrixEncodingTableIdx = GET_Q_DX_ENCODING_TABLE_IDX(1, runInfo.qDxPingPongIdx);
    flag.rightMatrixEncodingTableIdx = GET_K_V_ENCODING_TABLE_IDX(1, runInfo.kvPingPong);
    mm1.SetSelfDefineData(flag);
    mm1.SetTensorA(queryGm[dxOrQueryGmOffset]);
    mm1.SetTensorB(keyGm[keyOrValueGmOffset], true);
    LocalTensor<T2> mm2ResQueInTensor = mm2ResInQue[runInfo.commonRunInfo.taskIdMod2].AllocTensor<T2>();
    mm1.template IterateAll<false>(mm2ResQueInTensor, 0, false, true);
    mm2ResInQue[runInfo.commonRunInfo.taskIdMod2].FreeTensor(mm2ResQueInTensor);
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::ProcessSoftmaxGrad(FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF1: Cast + SoftmaxGradFront
    ///////////////////////////////////////////////////////////////
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    LocalTensor<T2> softmaxGradResTensor = softmaxGradResBuf.Get<T2>();
    if constexpr (HEAD_DIM_ALIGN <= VECTOR_BASEN) {
        CopyInSoftmaxGrad<T1, T2, T1, VECTOR_BASEM, HEAD_DIM_ALIGN>(
            constInfo, runInfo, 0, runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.halfS1RealSize,
            attenMaskOrYInQue, pseOrDyInQue, dxGm, yGm);
        CalculateCastSoftmaxGrad<T1, T2, T1, VECTOR_BASEM, HEAD_DIM_ALIGN>(
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
            CopyInSoftmaxGrad<T1, T2, T1, VECTOR_BASEM, HEAD_DIM_ALIGN>(constInfo, runInfo, loopIdx, curLoopSize, loopSize,
                                                                    attenMaskOrYInQue, pseOrDyInQue, dxGm, yGm);
            CalculateCastSoftmaxGrad<T1, T2, T1, VECTOR_BASEM, HEAD_DIM_ALIGN>(
                constInfo, curLoopSize, attenMaskOrYInQue, pseOrDyInQue, softmaxGradResTensor[loopSize * loopIdx]);
        }
    }
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::ProcessReCompute(FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF2: pse + attenMask + muls + simpleSoftmax copyIn+calculate
    ///////////////////////////////////////////////////////////////
    LocalTensor<T2> mm2ResQueInTensor = mm2ResInQue[runInfo.commonRunInfo.taskIdMod2].AllocTensor<T2>();
    CopyInAttenMask<IS_ATTEN_MASK, VECTOR_BASEM, VECTOR_BASEN>(constInfo, runInfo, attenMaskInfo, attenMaskOrYInQue,
                                                               pseOrDyInQue, attenMaskU8Gm);
    CopyInPse<T1, T2, IS_PSE>(constInfo, runInfo, pseInfo, pseOrDyInQue, pseGm);
    CalculatePseMulsSelSimpleSoftMax<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD(DETER_SPARSE_TYPE), VECTOR_BASEM, VECTOR_BASEN>(
        constInfo, runInfo, pseInfo, attenMaskInfo, maxSumQue[runInfo.commonRunInfo.taskIdMod2], attenMaskOrYInQue,
        pseOrDyInQue, mm2ResQueInTensor, mm2ResQueInTensor, pseSlope);
    LocalTensor<T2> mm1ResQueInTensor = mm1ResInQue[runInfo.commonRunInfo.taskIdMod2].AllocTensor<T2>();
    if (dropInfo.dropMaskOuter == 1) {
        CopyInDropOuter<IS_DROP>(dropMaskBuf, dropMaskInQue, dropMaskGm, runInfo.commonRunInfo, constInfo.commonConstInfo, dropInfo);
    } else {
        GenDropMask<IS_DROP>(dropMaskBuf, dropmaskIndexVecBuf, runInfo.commonRunInfo, constInfo.commonConstInfo, dropInfo);
    }
    CalculateDropout<T2, IS_DROP, VECTOR_BASEN>(constInfo, runInfo, dropInfo, mm1ResQueInTensor, mm1ResQueInTensor,
                                                dropMaskBuf);
    mm2ResInQue[runInfo.commonRunInfo.taskIdMod2].FreeTensor(mm2ResQueInTensor);
    mm1ResInQue[runInfo.commonRunInfo.taskIdMod2].FreeTensor(mm1ResQueInTensor);
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::IterateMm3Mm4Mm5(FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF3: sub + mul
    ///////////////////////////////////////////////////////////////
    LocalTensor<T2> mm1ResQueInTensor = mm1ResInQue[runInfo.commonRunInfo.taskIdMod2].AllocTensor<T2>();
    LocalTensor<T2> mm2ResQueInTensor = mm2ResInQue[runInfo.commonRunInfo.taskIdMod2].AllocTensor<T2>();
    LocalTensor<T2> softmaxGradResTensor = softmaxGradResBuf.Get<T2>();
    if (runInfo.commonRunInfo.s2RealSize > 64) {
        BroadcastSubMul<T2, 128, 0>(mm1ResQueInTensor, mm1ResQueInTensor, softmaxGradResTensor, mm2ResQueInTensor,
                                    runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
    } else {
        BroadcastSubMul<T2, 64, 0>(mm1ResQueInTensor, mm1ResQueInTensor, softmaxGradResTensor, mm2ResQueInTensor,
                                   runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
    }

    // input type fp32, no post, mov muls here
    if constexpr (IsSameType<T1, float>::value) {
        Muls(mm1ResQueInTensor, mm1ResQueInTensor, constInfo.scaleValue, VECTOR_BASEM * VECTOR_BASEN);
    }

    ///////////////////////////////////////////////////////////////
    // VF5: cast + nd2nz
    ///////////////////////////////////////////////////////////////
    LocalTensor<T1> vecOutBuffer1 = pOutQue.AllocTensor<T1>();
    CalculateDropout<T2, IS_DROP, VECTOR_BASEN>(constInfo, runInfo, dropInfo, mm2ResQueInTensor, mm2ResQueInTensor,
                                                dropMaskBuf);
    LocalTensor<uint8_t> selrIndexesTensor;
    CastTransdataDeconflict<T1, T2, VECTOR_BASEN>(vecOutBuffer1, mm2ResQueInTensor, selrIndexesTensor, VECTOR_BASEM);
    pOutQue.EnQue(vecOutBuffer1);
    pOutQue.DeQue<T1>();
    mm2ResInQue[runInfo.commonRunInfo.taskIdMod2].FreeTensor(mm2ResQueInTensor);

    int64_t dxOrQueryGmOffset = GetQueryOrDxOffset(runInfo);
    int64_t keyOrValueGmOffset = GetKeyOrValueOffset(runInfo);

    ///////////////////////////////////////////////////////////////
    // Matmal5 dv
    // left [B, N2, G, S1, S2] right [B, N2, G, S1, D] output [B, N2, 1, S2, D]
    ///////////////////////////////////////////////////////////////
    IterateMm5(runInfo, vecOutBuffer1, dxOrQueryGmOffset, keyOrValueGmOffset);
    pOutQue.FreeTensor(vecOutBuffer1);

    ///////////////////////////////////////////////////////////////
    // VF4: dq dk cast + nd2nz
    ///////////////////////////////////////////////////////////////
    LocalTensor<T1> vecOutBuffer = dSOutQue.AllocTensor<T1>();
    CastTransdataDeconflict<T1, T2, VECTOR_BASEN>(vecOutBuffer, mm1ResQueInTensor, selrIndexesTensor, VECTOR_BASEM);
    dSOutQue.EnQue(vecOutBuffer);
    dSOutQue.DeQue<T1>();
    mm1ResInQue[runInfo.commonRunInfo.taskIdMod2].FreeTensor(mm1ResQueInTensor);

    ///////////////////////////////////////////////////////////////
    // Matmal3 dq
    // left [B, N2, G, S1, s2] right [B, N2, 1, S2, D] output [B, N2, G, S1, D]
    ///////////////////////////////////////////////////////////////
    IterateMm3(runInfo, vecOutBuffer, dxOrQueryGmOffset, keyOrValueGmOffset);

    ///////////////////////////////////////////////////////////////
    // Matmal4 dk
    // left [B, N2, G, S1, S2] right [B, N2, 1, S1, D] output [B, N2, G, S2, D]
    ///////////////////////////////////////////////////////////////
    IterateMm4(runInfo, vecOutBuffer, dxOrQueryGmOffset, keyOrValueGmOffset);

    dSOutQue.FreeTensor(vecOutBuffer);
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::IterateMm3(
    FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer, int64_t dxOrQueryGmOffset, int64_t keyOrValueGmOffset)
{
    FagTscmFlagData flag{0};
    flag.rightMatrixEncodingTableIdx = GET_K_V_ENCODING_TABLE_IDX(2, runInfo.kvPingPong);
    mm2.SetSelfDefineData(flag);

    if constexpr (IS_TND) {
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        GetSeqQlenKvlenByBidx(runInfo.commonRunInfo.boIdx, actualS1Len, actualS2Len);
        mm2.SetOrgShape(actualS1Len, constInfo.commonConstInfo.n2D, actualS2Len, actualS2Len, runInfo.dAlign16);
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            mm2.SetOrgShape(constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.dSize,
                            constInfo.commonConstInfo.s2Size, constInfo.commonConstInfo.s2Size, runInfo.dAlign16);
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            mm2.SetOrgShape(runInfo.commonRunInfo.s1RealSize, constInfo.commonConstInfo.bN2D,
                            constInfo.commonConstInfo.s2Size, constInfo.commonConstInfo.s2Size, runInfo.dAlign16);
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            mm2.SetOrgShape(constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.n2D,
                            constInfo.commonConstInfo.s2Size, constInfo.commonConstInfo.s2Size, runInfo.dAlign16);
        }
    }
    LocalTensor<T1> dsScmTensor = dsScm.AllocTensor<T1>();
    CopyUB2L1(runInfo, dsScmTensor, vecOutBuffer);
    dsScm.EnQue(dsScmTensor);
    dsScm.DeQue<T1>();

    if constexpr (HAS_TAIL) {
        mm2.SetTail(runInfo.commonRunInfo.s1RealSize, constInfo.commonConstInfo.dSize, runInfo.commonRunInfo.s2RealSize);
    }
    mm2.SetTensorA(dsScmTensor);
    mm2.SetTensorB(keyGm[keyOrValueGmOffset]);

    DataCopyExtParams intriParams;
    intriParams.blockCount = runInfo.commonRunInfo.halfS1RealSize;
    intriParams.blockLen = constInfo.commonConstInfo.dSize * sizeof(T1);
    intriParams.srcStride = 0;
    if constexpr (IS_TND) {
        intriParams.dstStride = (constInfo.commonConstInfo.n2G - 1) * constInfo.commonConstInfo.dSize * sizeof(T1);
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            intriParams.dstStride = 0;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            intriParams.dstStride =
                (constInfo.bSize * constInfo.commonConstInfo.n2G - 1) * constInfo.commonConstInfo.dSize * sizeof(T1);
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            intriParams.dstStride = (constInfo.commonConstInfo.n2G - 1) * constInfo.commonConstInfo.dSize * sizeof(T1);
        }
    }

    if constexpr (dTemplateType <= DTemplateType::Aligned128) {
        LocalTensor<T2> mm1ResQueInTensor = mm1ResInQue[runInfo.commonRunInfo.taskIdMod2].AllocTensor<T2>();
        mm2.template IterateAll<false>(mm1ResQueInTensor, 0, false, true);

        mm2.WaitIterateAll();
        if (constInfo.commonConstInfo.s1Size == 1 && vSubBlockIdx == 1) {
            mm1ResInQue[runInfo.commonRunInfo.taskIdMod2].FreeTensor(mm1ResQueInTensor);
            mm2.End();
            dsScm.FreeTensor(dsScmTensor);
            return;
        }
        mm1ResInQue[runInfo.commonRunInfo.taskIdMod2].EnQue(mm1ResQueInTensor);
        mm1ResInQue[runInfo.commonRunInfo.taskIdMod2].DeQue<T2>();

        LocalTensor<T1> dqCastTensor = pOutQue.AllocTensor<T1>();
        MulsCast(dqCastTensor, mm1ResQueInTensor, runInfo.commonRunInfo.halfS1RealSize,
            (float)constInfo.scaleValue, constInfo.commonConstInfo.dSize);
        pOutQue.EnQue(dqCastTensor);
        pOutQue.DeQue<T1>();

        uint64_t dqGmOffset = dxOrQueryGmOffset;
        if constexpr (IS_TND) {
            dqGmOffset += vSubBlockIdx * constInfo.commonConstInfo.dSize * runInfo.commonRunInfo.firstHalfS1RealSize *
                          constInfo.commonConstInfo.n2G;
        } else {
            if (constInfo.commonConstInfo.layoutType == BNGSD) {
                dqGmOffset +=
                    vSubBlockIdx * constInfo.commonConstInfo.dSize * runInfo.commonRunInfo.firstHalfS1RealSize;
            } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                dqGmOffset += vSubBlockIdx * constInfo.commonConstInfo.dSize *
                              runInfo.commonRunInfo.firstHalfS1RealSize * constInfo.commonConstInfo.n2G;
            } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                dqGmOffset +=
                    vSubBlockIdx * constInfo.commonConstInfo.bN2GD * runInfo.commonRunInfo.firstHalfS1RealSize;
            }
        }
        if (runInfo.commonRunInfo.halfS1RealSize != 0) {
            AscendC::DataCopyPad(dqGm[dqGmOffset], dqCastTensor, intriParams);
        }

        pOutQue.FreeTensor(dqCastTensor);
        mm1ResInQue[runInfo.commonRunInfo.taskIdMod2].FreeTensor(mm1ResQueInTensor);
    } else {
        uint64_t dqWorkSpaceOffset = CUBE_BASEM * HEAD_DIM_ALIGN * cBlockIdx;
        mm2.template IterateAll<false>(dqWorkSpaceGm[dqWorkSpaceOffset], false, false, true);

        mm2.WaitIterateAll();
        if (constInfo.commonConstInfo.s1Size == 1 && vSubBlockIdx == 1) {
            mm2.End();
            dsScm.FreeTensor(dsScmTensor);
            return;
        }
        if (runInfo.commonRunInfo.halfS1RealSize == 0) {
            return;
        }

        uint64_t vSubBlockOffset =
            vSubBlockIdx * constInfo.commonConstInfo.dSize * runInfo.commonRunInfo.firstHalfS1RealSize;
        uint64_t vSubBlockOffsetDAlign = vSubBlockIdx * runInfo.dAlign16 * runInfo.commonRunInfo.firstHalfS1RealSize;
        uint32_t ubLoopProcessingCnt =
            BN2CeilConst(runInfo.commonRunInfo.halfS1RealSize * runInfo.dAlign16, VECTOR_BASEM * VECTOR_BASEN);
        for (uint32_t loop = 0; loop < ubLoopProcessingCnt; loop++) {
            intriParams.blockCount = BN2CeilConst(runInfo.commonRunInfo.halfS1RealSize, ubLoopProcessingCnt);
            uint64_t loopOffset = loop * constInfo.commonConstInfo.dSize * intriParams.blockCount;
            uint64_t loopOffsetDAlign = loop * runInfo.dAlign16 * intriParams.blockCount;
            if (loop == ubLoopProcessingCnt - 1) {
                intriParams.blockCount = runInfo.commonRunInfo.halfS1RealSize - intriParams.blockCount * loop;
            }
            uint32_t data_size = intriParams.blockCount * runInfo.dAlign16;
            LocalTensor<T2> mm1ResQueInTensor = mm1ResInQue[runInfo.commonRunInfo.taskIdMod2].AllocTensor<T2>();
            DataCopy(mm1ResQueInTensor, dqWorkSpaceGm[dqWorkSpaceOffset + vSubBlockOffsetDAlign + loopOffsetDAlign],
                     data_size);
            mm1ResInQue[runInfo.commonRunInfo.taskIdMod2].EnQue(mm1ResQueInTensor);
            mm1ResInQue[runInfo.commonRunInfo.taskIdMod2].DeQue<T2>();

            Muls(mm1ResQueInTensor, mm1ResQueInTensor, (float)constInfo.scaleValue, data_size);
            LocalTensor<T1> dqCastTensor = pOutQue.AllocTensor<T1>();
            Cast(dqCastTensor, mm1ResQueInTensor, RoundMode::CAST_ROUND, data_size);
            pOutQue.EnQue(dqCastTensor);
            pOutQue.DeQue<T1>();

            uint64_t dqGmOffset = dxOrQueryGmOffset;
            if (constInfo.commonConstInfo.layoutType == BNGSD) {
                dqGmOffset += (loopOffset + vSubBlockOffset);
            } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                dqGmOffset += (loopOffset + vSubBlockOffset) * constInfo.commonConstInfo.n2G;
            } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                dqGmOffset += (loopOffset + vSubBlockOffset) * constInfo.bSize * constInfo.commonConstInfo.n2G;
            }
            AscendC::DataCopyPad(dqGm[dqGmOffset], dqCastTensor, intriParams);

            pOutQue.FreeTensor(dqCastTensor);
            mm1ResInQue[runInfo.commonRunInfo.taskIdMod2].FreeTensor(mm1ResQueInTensor);
        }
    }

    mm2.End();
    dsScm.FreeTensor(dsScmTensor);
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::IterateMm4(
    FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer, int64_t dxOrQueryGmOffset, int64_t keyOrValueGmOffset)
{
    FagTscmFlagData flag{0};
    flag.rightMatrixEncodingTableIdx = GET_Q_DX_ENCODING_TABLE_IDX(3, runInfo.qDxPingPongIdx);
    mm2.SetSelfDefineData(flag);
    if constexpr (IS_TND) {
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        GetSeqQlenKvlenByBidx(runInfo.commonRunInfo.boIdx, actualS1Len, actualS2Len);
        mm2.SetOrgShape(actualS2Len, constInfo.commonConstInfo.n2GD, actualS1Len, actualS1Len, runInfo.dAlign16);
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            mm2.SetOrgShape(constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.dSize,
                            constInfo.commonConstInfo.s2Size, constInfo.commonConstInfo.s2Size, runInfo.dAlign16);
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            mm2.SetOrgShape(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.bN2GD,
                            constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.s1Size, runInfo.dAlign16);
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            mm2.SetOrgShape(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.n2GD,
                            constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.s1Size, runInfo.dAlign16);
        }
    }
    LocalTensor<T1> dsScmTensordq = dsScm.AllocTensor<T1>();
    dsScm.EnQue(dsScmTensordq);
    dsScm.DeQue<T1>();
    runInfo.s2RealSizeAlign2 = ((runInfo.commonRunInfo.s2RealSize + 1) >> 1 << 1);

    if constexpr (HAS_TAIL) {
        mm2.SetTail(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.dSize, runInfo.commonRunInfo.s1RealSize);
    }
    mm2.SetTensorA(dsScmTensordq, true);
    mm2.SetTensorB(queryGm[dxOrQueryGmOffset]); // sameB

    DataCopyExtParams intriParams;
    intriParams.blockCount = runInfo.halfS2RealSize;
    intriParams.blockLen = constInfo.commonConstInfo.dSize * sizeof(T1);
    intriParams.srcStride = 0;
    if constexpr (IS_TND) {
        intriParams.dstStride = (constInfo.commonConstInfo.n2G - 1) * constInfo.commonConstInfo.dSize * sizeof(T1);
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            intriParams.dstStride = 0;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            intriParams.dstStride =
                (constInfo.bSize * constInfo.commonConstInfo.n2G - 1) * constInfo.commonConstInfo.dSize * sizeof(T1);
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            intriParams.dstStride = (constInfo.commonConstInfo.n2G - 1) * constInfo.commonConstInfo.dSize * sizeof(T1);
        }
    }

    if constexpr (dTemplateType <= DTemplateType::Aligned128) {
        LocalTensor<T2> mm2ResQueInTensor = mm2ResInQue[runInfo.commonRunInfo.taskIdMod2].AllocTensor<T2>();
        mm2.template IterateAll<false>(mm2ResQueInTensor, 0, false, true);

        mm2.WaitIterateAll();
        mm2ResInQue[runInfo.commonRunInfo.taskIdMod2].EnQue(mm2ResQueInTensor);
        mm2ResInQue[runInfo.commonRunInfo.taskIdMod2].DeQue<T2>();
        if (constInfo.commonConstInfo.s2Size == 1 && vSubBlockIdx == 1) {
            mm2ResInQue[runInfo.commonRunInfo.taskIdMod2].FreeTensor(mm2ResQueInTensor);
            mm2.End();
            dsScm.FreeTensor(dsScmTensordq);
            return;
        }
        LocalTensor<T1> dkCastTensor = pOutQue.AllocTensor<T1>();
        MulsCast(dkCastTensor, mm2ResQueInTensor, runInfo.halfS2RealSize,
            (float)constInfo.scaleValue, constInfo.commonConstInfo.dSize);
        pOutQue.EnQue(dkCastTensor);
        pOutQue.DeQue<T1>();

        uint64_t dkGmOffset = keyOrValueGmOffset;
        if constexpr (IS_TND) {
            dkGmOffset += vSubBlockIdx * constInfo.commonConstInfo.dSize * runInfo.firstHalfS2RealSize *
                          constInfo.commonConstInfo.n2G;
        } else {
            if (constInfo.commonConstInfo.layoutType == BNGSD) {
                dkGmOffset += vSubBlockIdx * constInfo.commonConstInfo.dSize * runInfo.firstHalfS2RealSize;
            } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                dkGmOffset += vSubBlockIdx * constInfo.commonConstInfo.dSize * runInfo.firstHalfS2RealSize *
                              constInfo.commonConstInfo.n2G;
            } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                dkGmOffset += vSubBlockIdx * runInfo.firstHalfS2RealSize * constInfo.commonConstInfo.bN2GD;
            }
        }
        if (runInfo.halfS2RealSize != 0) {
            AscendC::DataCopyPad(dkGm[dkGmOffset], dkCastTensor, intriParams);
        }

        pOutQue.FreeTensor(dkCastTensor);
        mm2ResInQue[runInfo.commonRunInfo.taskIdMod2].FreeTensor(mm2ResQueInTensor);
    } else {
        uint64_t dkWorkSpaceOffset = CUBE_BASEN * HEAD_DIM_ALIGN * cBlockIdx;
        mm2.template IterateAll<false>(dkWorkSpaceGm[dkWorkSpaceOffset], false, false, true);

        mm2.WaitIterateAll();
        if (constInfo.commonConstInfo.s2Size == 1 && vSubBlockIdx == 1) {
            mm2.End();
            dsScm.FreeTensor(dsScmTensordq);
            return;
        }
        if (runInfo.halfS2RealSize == 0) {
            return;
        }

        uint64_t vSubBlockOffset = vSubBlockIdx * constInfo.commonConstInfo.dSize * runInfo.firstHalfS2RealSize;
        uint64_t vSubBlockOffsetDAlign = vSubBlockIdx * runInfo.dAlign16 * runInfo.firstHalfS2RealSize;
        uint32_t ubLoopProcessingCnt =
            BN2CeilConst(runInfo.halfS2RealSize * runInfo.dAlign16, VECTOR_BASEM * VECTOR_BASEN);
        for (uint32_t loop = 0; loop < ubLoopProcessingCnt; loop++) {
            intriParams.blockCount = BN2CeilConst(runInfo.halfS2RealSize, ubLoopProcessingCnt);
            uint64_t loopOffset = loop * constInfo.commonConstInfo.dSize * intriParams.blockCount;
            uint64_t loopOffsetDAlign = loop * runInfo.dAlign16 * intriParams.blockCount;
            if (loop == ubLoopProcessingCnt - 1) {
                intriParams.blockCount = runInfo.halfS2RealSize - intriParams.blockCount * loop;
            }
            uint32_t data_size = intriParams.blockCount * runInfo.dAlign16;
            LocalTensor<T2> mm2ResQueInTensor = mm2ResInQue[runInfo.commonRunInfo.taskIdMod2].AllocTensor<T2>();
            DataCopy(mm2ResQueInTensor, dkWorkSpaceGm[dkWorkSpaceOffset + vSubBlockOffsetDAlign + loopOffsetDAlign],
                     data_size);
            mm2ResInQue[runInfo.commonRunInfo.taskIdMod2].EnQue(mm2ResQueInTensor);
            mm2ResInQue[runInfo.commonRunInfo.taskIdMod2].DeQue<T2>();

            Muls(mm2ResQueInTensor, mm2ResQueInTensor, (float)constInfo.scaleValue, data_size);
            LocalTensor<T1> dkCastTensor = pOutQue.AllocTensor<T1>();
            Cast(dkCastTensor, mm2ResQueInTensor, RoundMode::CAST_ROUND, data_size);
            pOutQue.EnQue(dkCastTensor);
            pOutQue.DeQue<T1>();

            uint64_t dkGmOffset = keyOrValueGmOffset;
            if (constInfo.commonConstInfo.layoutType == BNGSD) {
                dkGmOffset += (loopOffset + vSubBlockOffset);
            } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                dkGmOffset += (loopOffset + vSubBlockOffset) * constInfo.commonConstInfo.n2G;
            } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                dkGmOffset += (loopOffset + vSubBlockOffset) * constInfo.bSize * constInfo.commonConstInfo.n2G;
            }
            AscendC::DataCopyPad(dkGm[dkGmOffset], dkCastTensor, intriParams);

            pOutQue.FreeTensor(dkCastTensor);
            mm2ResInQue[runInfo.commonRunInfo.taskIdMod2].FreeTensor(mm2ResQueInTensor);
        }
    }

    mm2.End();
    dsScm.FreeTensor(dsScmTensordq);
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::IterateMm5(
    FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer1, int64_t dxOrQueryGmOffset, int64_t keyOrValueGmOffset)
{
    FagTscmFlagData flag{0};
    flag.rightMatrixEncodingTableIdx = GET_Q_DX_ENCODING_TABLE_IDX(4, runInfo.qDxPingPongIdx);
    mm3.SetSelfDefineData(flag);
    if constexpr (IS_TND) {
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        GetSeqQlenKvlenByBidx(runInfo.commonRunInfo.boIdx, actualS1Len, actualS2Len);
        mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.n2GD, actualS1Len, actualS1Len,
                        constInfo.commonConstInfo.n2D);
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.dSize,
                            constInfo.commonConstInfo.s1Size);
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.bN2GD,
                            constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.s1Size,
                            constInfo.commonConstInfo.bN2D);
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.n2GD,
                            constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.s1Size,
                            constInfo.commonConstInfo.n2D);
        }
    }

    LocalTensor<T1> pScmTensor = pScm.AllocTensor<T1>();
    CopyUB2L1(runInfo, pScmTensor, vecOutBuffer1);
    pScm.EnQue(pScmTensor);
    pScm.DeQue<T1>();
    if constexpr (HAS_TAIL) {
        mm3.SetTail(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.dSize,
                    runInfo.commonRunInfo.s1RealSize);
    }
    mm3.SetTensorA(pScmTensor, true);
    mm3.SetTensorB(dxGm[dxOrQueryGmOffset]); // sameB
    mm3.template IterateAll<false>(dvGm[keyOrValueGmOffset], false, false, false);

    mm3.End();
    pScm.FreeTensor(pScmTensor);
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::CopyUB2L1(
    FagRunInfo &runInfo, LocalTensor<T1> &dstTensor, LocalTensor<T1> &srcTensor)
{
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    uint32_t scmOffset = (vSubBlockIdx == 0 ? 0 : runInfo.commonRunInfo.firstHalfS1RealSize * FRACTAL_NZ_C0_SIZE);
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = VECTOR_BASEN / FRACTAL_NZ_C0_SIZE;
    dataCopyParams.blockLen = (uint16_t)(runInfo.commonRunInfo.halfS1RealSize * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM);
    dataCopyParams.srcStride =
        (uint16_t)((VECTOR_BASEM + 1 - runInfo.commonRunInfo.halfS1RealSize) * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM);
    uint32_t s1RealSizeAlignTo16 = AlignTo16(runInfo.commonRunInfo.s1RealSize);
    dataCopyParams.dstStride =
        (s1RealSizeAlignTo16 - runInfo.commonRunInfo.halfS1RealSize) * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM;
    DataCopy(dstTensor[scmOffset], srcTensor, dataCopyParams);
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline bool FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::CheckIsValidBlock(
    FagRunInfo &runInfo, int64_t baseIdx, int64_t s1oDimIdx, int64_t s2oDimIdx)
{
    int64_t s2IdxLeft = s2oDimIdx * CUBE_BASEN;
    int64_t s2IdxRight = (s2oDimIdx + 1) * CUBE_BASEN;
    int64_t s2IgnoredEndLen =
        static_cast<int64_t>(constInfo.commonConstInfo.s1Size) - static_cast<int64_t>(CUBE_BASEM * (s1oDimIdx + 1));
    int64_t s2EndLen = 0;
    if (static_cast<int64_t>(constInfo.commonConstInfo.s2Size) > s2IgnoredEndLen) {
        s2EndLen = static_cast<int64_t>(constInfo.commonConstInfo.s2Size) - s2IgnoredEndLen;
    } else {
        s2EndLen = 0;
    }

    if (constInfo.sparseMode == PREFIX || constInfo.sparseMode == PREFIX_COMPRESS) {
        int64_t curBIdx = baseIdx / constInfo.n2GS1oS2o;
        s2EndLen = Min(Max(s2EndLen, ((__gm__ int64_t *)prefixNAddr)[curBIdx]),
                       static_cast<int64_t>(constInfo.commonConstInfo.s2Size));
    }
    bool isValid = s2IdxLeft < s2EndLen;
    if (isValid) {
        runInfo.s2CvBegin = s2IdxLeft;
        runInfo.s2CvEnd = runInfo.s2CvBegin + CUBE_BASEN; // 非尾块s2按照+CUBE_BASEN处理
        if (s2oDimIdx == constInfo.s2Outer - 1) {         // 默认s2 cv tail相等
            runInfo.s2CvEnd = runInfo.s2CvBegin + constInfo.s2Tail;
        }
    }
    return isValid;
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline bool
FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::IsValid(FagRunInfo &runInfo, int64_t index)
{
    if constexpr (IS_TND) {
        int64_t resbaseIdx = index - runInfo.lastBatchTotalBaseIdx;
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        for (int64_t bIdx = runInfo.lastBatchIdx; bIdx < constInfo.bSize; bIdx++) {
            GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
            int64_t s1OuterTmp = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
            int64_t s2OuterTmp = (actualS2Len + VECTOR_BASEN - 1) / VECTOR_BASEN;
            int64_t totalBaseIdx = constInfo.n2Size * constInfo.commonConstInfo.gSize * s1OuterTmp * s2OuterTmp;
            if (resbaseIdx < totalBaseIdx) {
                int64_t gDimTail = resbaseIdx % (s1OuterTmp * s2OuterTmp);
                int64_t s2oDimIdx = gDimTail / s1OuterTmp;
                int64_t s1oDimIdx = gDimTail % s1OuterTmp;
                int64_t s2IdxLeft = s2oDimIdx * VECTOR_BASEN;
                int64_t s2IdxRight = Min((s2oDimIdx + 1) * VECTOR_BASEN, actualS2Len);
                int64_t s2SparseLeft = Max(CUBE_BASEM * s1oDimIdx - constInfo.s1Token, 0);
                s2SparseLeft = s2SparseLeft / 64 * 64;
                int64_t s2SparseRight = AlignTo64(CUBE_BASEM * (s1oDimIdx + 1) + constInfo.s2Token);
                s2SparseRight = Min(s2SparseRight, actualS2Len);
                bool isValid = s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft;
                runInfo.s2CvBegin = s2IdxLeft;
                runInfo.s2CvEnd = runInfo.s2CvBegin + CUBE_BASEN; // 非尾块s2按照+CUBE_BASEN处理
                if (s2oDimIdx == s2OuterTmp - 1) {                // 默认s2 cv tail相等
                    runInfo.s2CvEnd = runInfo.s2CvBegin + actualS2Len - s2oDimIdx * CUBE_BASEN;
                }
                return isValid;
            } else {
                resbaseIdx -= totalBaseIdx;
            }
        }
        return false;
    } else {
        int64_t gDimTail = index % constInfo.s1oS2o;
        int64_t s2oDimIdx = gDimTail / constInfo.s1Outer;
        int64_t s1oDimIdx = gDimTail % constInfo.s1Outer;
        int64_t s2IdxLeft = s2oDimIdx * VECTOR_BASEN;
        int64_t s2IdxRight = Min((s2oDimIdx + 1) * VECTOR_BASEN, constInfo.commonConstInfo.s2Size);
        if constexpr (IS_ATTEN_MASK) {
            if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL || constInfo.sparseMode == PREFIX ||
                constInfo.sparseMode == PREFIX_COMPRESS) {
                return CheckIsValidBlock(runInfo, index, s1oDimIdx, s2oDimIdx);
            } else {
                int64_t s2SparseLeft = Max(CUBE_BASEM * s1oDimIdx - constInfo.s1Token, 0);
                s2SparseLeft = s2SparseLeft / 64 * 64;
                int64_t s2SparseRight = AlignTo64(CUBE_BASEM * (s1oDimIdx + 1) + constInfo.s2Token);
                s2SparseRight = Min(s2SparseRight, constInfo.commonConstInfo.s2Size);
                bool isValid = s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft;
                runInfo.s2CvBegin = s2IdxLeft;
                runInfo.s2CvEnd = runInfo.s2CvBegin + CUBE_BASEN; // 非尾块s2按照+CUBE_BASEN处理
                if (s2oDimIdx == constInfo.s2Outer - 1) {         // 默认s2 cv tail相等
                    runInfo.s2CvEnd = runInfo.s2CvBegin + constInfo.s2Tail;
                }
                return isValid;
            }
        } else {
            runInfo.s2CvBegin = s2IdxLeft;
            runInfo.s2CvEnd = s2IdxRight;
            return true;
        }
    }
}

FAG_BN2_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2StaticRegbase<FAG_BN2_FUNCTION_PARAMS_TEMPLATE>::SyncALLCores()
{
    SyncAll();
}

#endif // _FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2_REGBASE_H_