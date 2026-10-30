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
 * \file flash_attention_score_grad_s1s2_bn2gs1s2_regbase.h
 * \brief
 */
#ifndef FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2GS1S2_REGBASE_H_
#define FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2GS1S2_REGBASE_H_

#include <algorithm>
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
#include "vector_api/vf_ds_abs_reduce_max.h"
#include "deter.h"
#include "../../../common/op_kernel/arch35/dropmask.h"
#include "flash_attention_score_grad_tiling_data_regbase.h"

#define FAG_CLASS_TEMPLATE                                                                                             \
    template <typename T1, typename T2, const bool IS_ATTEN_MASK = 0, const bool IS_PSE = 0, const bool IS_DROP = 0,   \
              const bool IS_TND = 0, const bool IS_BN2_MULTIBLK = 0, const uint8_t DETER_SPARSE_TYPE = 0, bool IS_N_EQUAL = 0, const bool IS_D_NO_EQUAL = 0,   \
              const bool IS_ROPE = 0, const bool FP8_OPEN_TSCM = 0, const uint8_t SPLIT_AXIS = 0, S1TemplateType s1TemplateType = S1TemplateType::Aligned128, \
              S2TemplateType s2TemplateType = S2TemplateType::Aligned128,                                              \
              DTemplateType dTemplateType = DTemplateType::Aligned128, typename OUTDTYPE = T1>
#define FAG_FUNCTION_TEMPLATE                                                                                          \
    template <typename T1, typename T2, const bool IS_ATTEN_MASK, const bool IS_PSE, const bool IS_DROP,               \
              const bool IS_TND, const bool IS_BN2_MULTIBLK, const uint8_t DETER_SPARSE_TYPE, const bool IS_N_EQUAL, const bool IS_D_NO_EQUAL,                   \
              const bool IS_ROPE, const bool FP8_OPEN_TSCM, const uint8_t SPLIT_AXIS, S1TemplateType s1TemplateType, S2TemplateType s2TemplateType,                  \
              DTemplateType dTemplateType, typename OUTDTYPE>
#define FAG_FUNCTION_PARAMS_TEMPLATE                                                                                   \
    T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, IS_ROPE, FP8_OPEN_TSCM, SPLIT_AXIS, s1TemplateType,     \
        s2TemplateType, dTemplateType, OUTDTYPE

using namespace matmul;
using namespace optiling::fag;

__aicore__ constexpr uint32_t CeilConst(uint32_t a, uint32_t b) 
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

FAG_CLASS_TEMPLATE
class FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase
{
public:
    __aicore__ inline FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase(){};

    __aicore__ inline void Init(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query,
                                __gm__ uint8_t *pseShift, __gm__ uint8_t *dropMask, __gm__ uint8_t *attenMask,
                                __gm__ uint8_t *y, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                __gm__ uint8_t *prefixN, __gm__ uint8_t *actualSeqQlen, __gm__ uint8_t *actualSeqKvlen,
                                __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *deqScaleDy, 
                                __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dpse,
                                __gm__ uint8_t *dpRope, __gm__ uint8_t *dkRope, __gm__ uint8_t *workspace,
                                FagOldTilingType ordTilingData,
                                TPipe *pipeIn, TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> &dsScmIn,
                                TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> &pScmIn);
    __aicore__ inline void SetConstInfo();
    __aicore__ inline void SetRunInfo(FagRunInfo &runInfo, int64_t taskId, int64_t index);
    __aicore__ inline void GetIsNeedDeter(int64_t computeLoopIdx);
    __aicore__ inline void SetOptionalInfo();
    __aicore__ inline void InitUbBuffer();
    __aicore__ inline void InitWorkSpace(__gm__ uint8_t *workspace, __gm__ uint8_t *dq, __gm__ uint8_t *dk,
                                         __gm__ uint8_t *dv);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessNormal();
    __aicore__ inline void ProcessDeter();
    __aicore__ inline bool IsValid(FagRunInfo &runInfo, int64_t index);
    __aicore__ inline void UpdateToken(FagRunInfo &runInfo, int64_t bIdx);
    __aicore__ inline bool CheckIsValidBlock(FagRunInfo &runInfo, int64_t baseIdx, int64_t s1oDimIdx,
                                             int64_t s2oDimIdx);
    __aicore__ inline int64_t GetNextValidIdx(FagRunInfo &runInfo, int64_t startIndex, int64_t loopIdx=0);
    __aicore__ inline int64_t GetNextValidIdxFromFormula(int64_t loopIdx);
    __aicore__ inline void SetTscmPreloadFlag(FagRunInfo &runInfo, FagTscmFlagData &flag, int64_t nextIndex, 
                                             uint64_t& nextDxOffset, uint64_t& nextQueryOffset);
    __aicore__ inline void IterateMm1Mm2(FagRunInfo &runInfo, int64_t nextIndex, int64_t computeBlockIdx = 0,
                                         int64_t remainLoopNum = 0); // qk dxv
    __aicore__ inline void WriteOffsetToGM(int64_t queryGmOffset, int64_t keyGmOffset, int64_t valueGmOffset,
                                           int64_t computeBlockIdx, int64_t remainLoopNum);
    __aicore__ inline void ProcessSoftmaxGrad(FagRunInfo &runInfo); // softmaxGrad
    __aicore__ inline void WaitMm1Mm2Result();
    __aicore__ inline void WaitMm3Result(int64_t computeBlockIdx);
    __aicore__ inline int64_t GetQueryOffset(FagRunInfo &runInfo, bool isMm1Mm2 = true);
    __aicore__ inline int64_t GetQueryRopeOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetDxOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetKeyOffset(FagRunInfo &runInfo, bool isMm1Mm2 = true);
    __aicore__ inline int64_t GetKeyRopeOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetValueOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetDeqScaleQOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetDeqScaleKOffset(FagRunInfo &runInfo);
    __aicore__ inline void ProcessReCompute(FagRunInfo &runInfo);
    __aicore__ inline void IterateMm3Mm4Mm5(FagRunInfo &runInfo, int64_t nextIndex = -1); // dq dk dv
    __aicore__ inline void IterateMm3(FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer, int64_t dxOrQueryGmOffset,
                                      int64_t keyOrValueGmOffset, float qScaleDs = 1.0);
    __aicore__ inline void IterateMm3Deter(FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer,
                                           int64_t dxOrQueryGmOffset, int64_t keyOrValueGmOffset);
    __aicore__ inline void IterateMm4(FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer, int64_t dxOrQueryGmOffset,
                                      int64_t keyOrValueGmOffset, int64_t nextIndex, bool isNextS2IdxNoChange, float qScaleDs = 1.0);
    __aicore__ inline void IterateMm4Deter(FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer,
                                           int64_t dxOrQueryGmOffset, int64_t keyOrValueGmOffset);
    __aicore__ inline void IterateMm5(FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer, int64_t dxOrQueryGmOffset,
                                      int64_t keyOrValueGmOffset, int64_t nextIndex, bool isNextS2IdxNoChange);
    __aicore__ inline void IterateMm5Deter(FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer,
                                           int64_t dxOrQueryGmOffset, int64_t keyOrValueGmOffset);
    template <const bool IS_DQ = false>
    __aicore__ inline void CopyUB2L1(FagRunInfo &runInfo, LocalTensor<T1> &dstTensor, LocalTensor<T1> &srcTensor);
    __aicore__ inline void CopyUB2L1Deter(FagRunInfo &runInfo, LocalTensor<T1> &dstTensor, LocalTensor<T1> &srcTensor);
    __aicore__ inline void DeterCompute(bool isLastSort, int64_t computeBlockIdx, int64_t remainLoopNum);
    __aicore__ inline void DeterComputeDq(bool isLastSort, int64_t computeBlockIdx, int64_t remainLoopNum);
    __aicore__ inline void DeterComputeDkv(bool isLastSort, int64_t computeBlockIdx, int64_t remainLoopNum);
    __aicore__ inline void DeterComputeDqkv(bool isLastSort, int64_t computeBlockIdx, int64_t remainLoopNum);
    __aicore__ inline void SyncALLCores();
    __aicore__ inline void GetSeqQlenKvlenByBidx(int64_t bIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvlen);

    constexpr static bool IS_FP8_INPUT = IsSameType<T1, fp8_e5m2_t>::value || IsSameType<T1, fp8_e4m3fn_t>::value;
    constexpr static bool IS_FP32_INPUT = IsSameType<T1, float>::value;
    constexpr static float FP8_MAX = IsSameType<T1, fp8_e5m2_t>::value ? 57344 : 448;
    constexpr static uint32_t BITS_EACH_UINT64 = 64;
    constexpr static uint32_t MAX_BITS_IN_TILING = 32 * BITS_EACH_UINT64;
    constexpr static uint32_t INT64_BLOCK_NUM = 32 / sizeof(int64_t);
    constexpr static uint32_t DETER_OFFSET_UB_SIZE = 1024 * 3;
    constexpr static uint32_t CUBE_BASEM = (uint32_t)s1TemplateType;
    constexpr static uint32_t CUBE_BASEN = (uint32_t)s2TemplateType;
    constexpr static uint32_t HEAD_DIM_ALIGN = (uint32_t)dTemplateType;
    constexpr static uint32_t VECTOR_BASEM = CUBE_BASEM / CV_CORE_RATIO;
    constexpr static uint32_t VECTOR_BASEN = CUBE_BASEN;
    constexpr static uint32_t INPUT_BLOCK_NUM = 32 / sizeof(T1);
    constexpr static uint32_t INPUT_BLOCK_NUM_FOR_FP8 = 32 / sizeof(OUTDTYPE);
    constexpr static uint32_t BASE_DQ_SIZE = CUBE_BASEM * HEAD_DIM_ALIGN;
    constexpr static uint32_t BASE_DKV_SIZE = CUBE_BASEN * HEAD_DIM_ALIGN;
    constexpr static int64_t OUTINDEX = -1;
    constexpr static uint32_t FRACTAL_NZ_C0_SIZE = 32 / sizeof(T1);
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP16 = 32 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP32_D256 = 16 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP32_D512 = 64 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE = 
        IS_FP32_INPUT ? (HEAD_DIM_ALIGN > 256 ? DETER_DQ_UB_SIZE_FP32_D512 : DETER_DQ_UB_SIZE_FP32_D256) : DETER_DQ_UB_SIZE_FP16;
    constexpr static uint32_t DETER_DKV_UB_SIZE = VECTOR_BASEM * VECTOR_BASEN * sizeof(T2);

    constexpr static bool IS_MM3_L0_EXCEED = IS_L0_EXCEED(CUBE_BASEM, HEAD_DIM_ALIGN, CUBE_BASEN, T1);
    constexpr static uint32_t MM1_LEFT_BASE_RATIO = CeilConst(CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(T1), L0_MAX_SIZE);
    constexpr static uint32_t MM1_RIGHT_BASE_RATIO = CeilConst(CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(T1), L0_MAX_SIZE);
    constexpr static uint32_t MM1_MAX_BASE_RATIO =
        MM1_LEFT_BASE_RATIO > MM1_RIGHT_BASE_RATIO ? MM1_LEFT_BASE_RATIO : MM1_RIGHT_BASE_RATIO;

    constexpr static uint32_t MM2_LEFT_BASE_RATIO = CeilConst(CUBE_BASEM * CUBE_BASEN * sizeof(T1), L0_MAX_SIZE);
    constexpr static uint32_t MM2_RIGHT_BASE_RATIO = CeilConst(CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(T1), L0_MAX_SIZE);
    constexpr static uint32_t MM2_MAX_BASE_RATIO =
        MM2_LEFT_BASE_RATIO > MM2_RIGHT_BASE_RATIO ? MM2_LEFT_BASE_RATIO : MM2_RIGHT_BASE_RATIO;

    constexpr static uint32_t MM3_LEFT_BASE_RATIO = CeilConst(CUBE_BASEM * CUBE_BASEN * sizeof(T1), L0_MAX_SIZE);
    constexpr static uint32_t MM3_RIGHT_BASE_RATIO = CeilConst(CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(T1), L0_MAX_SIZE);
    constexpr static uint32_t MM3_MAX_BASE_RATIO =
        MM3_LEFT_BASE_RATIO > MM3_RIGHT_BASE_RATIO ? MM3_LEFT_BASE_RATIO : MM3_RIGHT_BASE_RATIO;

    // 开启c1Shared后，确保每一块L0C大小够用
    constexpr static uint32_t SHARED_C1_BUFFER_SZIE = GET_SHARED_C1_BUFFER_SZIE(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN);
    constexpr static bool IS_TSCM_REUSE = IS_TSCM_REUSE(HEAD_DIM_ALIGN, T1, IS_DETER_OLD(DETER_SPARSE_TYPE), FP8_OPEN_TSCM);
    constexpr static bool IS_L0DB =
        (HEAD_DIM_ALIGN <= (uint32_t)DTemplateType::Aligned128 && !IS_FP32_INPUT);
    constexpr static bool IS_L0C_REUSE = IS_L0C_REUSE(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN, IS_DETER_OLD(DETER_SPARSE_TYPE), T1, IS_TND);
    constexpr static bool IS_TSCM_PRELOAD = IS_TSCM_PRELOAD_ROPE(HEAD_DIM_ALIGN, T1, SPLIT_AXIS, IS_DETER_OLD(DETER_SPARSE_TYPE), IS_TND, FP8_OPEN_TSCM, IS_ROPE);
    constexpr static MatmulConfig MM1_CFG_SAMEAB =
        GetMm1Cfg<T1>(IS_ATTEN_MASK, IS_PSE, IS_DROP, CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN, IS_TSCM_REUSE,
                      IS_L0DB, IS_L0C_REUSE, SHARED_C1_BUFFER_SZIE, MM1_MAX_BASE_RATIO);
    constexpr static MatmulConfig MM2_CFG_SAMEB =
        GetMm2Cfg<T1>(IS_ATTEN_MASK, IS_PSE, IS_DROP, CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN, IS_TSCM_REUSE,
                      IS_L0DB, IS_L0C_REUSE, SHARED_C1_BUFFER_SZIE, MM2_MAX_BASE_RATIO);
    constexpr static MatmulConfig MM3_CFG_SAMEB =
        GetMm3Cfg<T1>(IS_ATTEN_MASK, IS_PSE, IS_DROP, CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN, IS_TSCM_REUSE,
                      IS_L0DB, IS_L0C_REUSE, SHARED_C1_BUFFER_SZIE, MM3_MAX_BASE_RATIO);
    constexpr static uint32_t L0C_BUF_NUM = GET_L0C_BUF_NUM(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN);
    constexpr static uint8_t ALIGN_NUM_32 = 32;
    constexpr static uint8_t VEC_CORE_NUM_64 = 64;

    using aType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true, LayoutMode::NONE, true>;
    using bType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true, LayoutMode::NONE, true>;
    using cType1 = MatmulType<TPosition::VECCALC, CubeFormat::ND_ALIGN, T2>;
    using biasType1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using aType2 = MatmulType<TPosition::TSCM, CubeFormat::NZ, T1, true, LayoutMode::NONE, true, TPosition::VECOUT>;
    using bType2 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true, LayoutMode::NONE, true>;
    using bType3 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true, LayoutMode::NONE, true>;
    using cType2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using biasType2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    constexpr static auto MM1_TILING_CFG_SAMEAB = GetMatmulApiTiling<aType1, bType1, cType1, biasType1>(MM1_CFG_SAMEAB);
    constexpr static auto MM2_TILING_CFG_SAMEB = GetMatmulApiTiling<aType2, bType2, cType2, biasType2>(MM2_CFG_SAMEB);
    constexpr static auto MM3_TILING_CFG_SAMEB = GetMatmulApiTiling<aType2, bType2, cType2, biasType2>(MM3_CFG_SAMEB);
    Matmul<aType1, bType1, cType1, biasType1, MM1_TILING_CFG_SAMEAB,
           matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
           Mm1ConstPolicySelector<IS_TSCM_REUSE, IS_TSCM_PRELOAD, IS_L0C_REUSE, IS_ROPE>::template Result>
        mm1;
    Matmul<aType2, bType2, cType2, biasType2, MM2_TILING_CFG_SAMEB,
           matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
           Mm2ConstPolicySelector<IS_TSCM_REUSE, IS_TSCM_PRELOAD, IS_L0C_REUSE>::template Result>
        mm2;
    Matmul<aType2, bType3, cType2, biasType2, MM3_TILING_CFG_SAMEB,
           matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
           Mm3ConstPolicySelector<IS_TSCM_REUSE, IS_TSCM_PRELOAD, IS_L0C_REUSE>::template Result>
        mm3;

protected:
    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> attenMaskOrYInQue;
    TQue<QuePosition::VECIN, 1> pseOrDyInQue;
    TQue<QuePosition::VECOUT, 1> dSOutQue;
    TQue<QuePosition::VECOUT, 1> pOutQue;
    TBuf<> mm1ResBuf[2];
    TBuf<> mm2ResBuf[2];
    TQue<QuePosition::VECIN, 1> maxSumQue[2];
    TBuf<> softmaxGradResBuf;
    TBuf<> dropMaskBuf;
    TBuf<> dropmaskIndexVecBuf;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> deterInOutQue;
    TBuf<> deterOffsetBuf;
    TBuf<> vselrIndexesBuf;
    TQue<QuePosition::VECOUT, 1> dsAmaxOutQue;
    __gm__ uint8_t *prefixNAddr;
    __gm__ uint8_t *actualSeqQlenAddr;
    __gm__ uint8_t *actualSeqKvlenAddr;
    __gm__ uint8_t *queryRopeAddr;
    __gm__ uint8_t *keyRopeAddr;

    uint32_t vBlockIdx;
    uint32_t cBlockIdx;
    uint32_t vSubBlockIdx;
    int64_t lastS2oCvDimIdx = -1; // 上一次的s2方向基本块idx
    int64_t lastBdimIdx = -1;     // 上一次的b方向基本块idx
    int64_t lastN2dimIdx = -1;    // 上一次的n2方向基本块idx
    uint8_t kvPingPong = 1;
    bool isLastLoop = false;
    int64_t s2CvBegin;
    int64_t s2CvEnd;
    int64_t actualCalcS1Token;  // 转换后实际计算使用的S1Token
    int64_t actualCalcS2Token;

    // BN2S2模板判断是否有无效S2列
    int64_t curS2oIdx = -1;
    int64_t curS2InvalidTotalNum = 0;

    FagOldTilingType tilingData;
    // input
    GlobalTensor<T1> keyGm, valueGm, dxGm, queryGm, queryRopeGm, keyRopeGm;
    GlobalTensor<OUTDTYPE> pseGm, yGm;
    GlobalTensor<uint8_t> dropMaskGm, attenMaskU8Gm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm, pseFloatGm;
    GlobalTensor<float> deqScaleQGm, deqScaleKGm, deqScaleVGm, deqScaleDyGm;
    __gm__ uint8_t *pseSlope;
    // output
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, dqRopeGm, dkRopeGm;
    GlobalTensor<uint8_t> dropMaskWorkspaceGm;
	GlobalTensor<float> dsAmaxWorkSpaceGm;
    GlobalTensor<T1> dkGm, dvGm;
    GlobalTensor<float> deterGm;
    GlobalTensor<int64_t> deterOffsetGm;
    TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> dsScm;
    TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> pScm;

    FagConstInfo constInfo;
    AttenMaskInfo attenMaskInfo;
    PseInfo pseInfo;
    DropMaskInfo dropInfo;

    // deter ping pong flag
    int16_t deterPpFlag = 1;
    bool isFirstDeter = true;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), int64_t[36], std::nullptr_t>::type dqOffset;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), int64_t[36], std::nullptr_t>::type dkOffset;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), int64_t[36], std::nullptr_t>::type dvOffset;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), bool[2], std::nullptr_t>::type dqIsNeedDeter{};
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), bool[2], std::nullptr_t>::type dkDvIsNeedDeter{};

    typename std::conditional<IS_DETER_NEW(DETER_SPARSE_TYPE), CoordinateInfo[2], std::nullptr_t>::type coordinateInfos{};
    typename std::conditional<DETER_SPARSE_TYPE == DETER_BAND, BandInfo, std::nullptr_t>::type bandInfo;
    bool isMm3NeedWait = false;
};

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::Init(
    __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *dropMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *y, __gm__ uint8_t *softmaxMax,
    __gm__ uint8_t *softmaxSum, __gm__ uint8_t *prefixN, __gm__ uint8_t *actualSeqQlen, __gm__ uint8_t *actualSeqKvlen,
    __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *deqScaleDy,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dpse,
    __gm__ uint8_t *dqRope, __gm__ uint8_t *dkRope, __gm__ uint8_t *workspace,
    FagOldTilingType ordTilingData, TPipe *pipeIn,
    TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> &dsScmIn, TSCM<QuePosition::VECIN, 1, GROUP_TSCM_MASK> &pScmIn)
{
    keyGm.SetGlobalBuffer((__gm__ T1 *)key);
    valueGm.SetGlobalBuffer((__gm__ T1 *)value);
    dxGm.SetGlobalBuffer((__gm__ T1 *)dx);
    queryGm.SetGlobalBuffer((__gm__ T1 *)query);
    yGm.SetGlobalBuffer((__gm__ OUTDTYPE *)y);
    pseGm.SetGlobalBuffer((__gm__ OUTDTYPE *)pseShift);
    pseFloatGm.SetGlobalBuffer((__gm__ float *)pseShift);
    dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)dropMask);
    attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    deqScaleQGm.SetGlobalBuffer((__gm__ float *)deqScaleQ);
    deqScaleKGm.SetGlobalBuffer((__gm__ float *)deqScaleK);
    deqScaleVGm.SetGlobalBuffer((__gm__ float *)deqScaleV);
    deqScaleDyGm.SetGlobalBuffer((__gm__ float *)deqScaleDy);
    dkGm.SetGlobalBuffer((__gm__ T1 *)dk);
    dvGm.SetGlobalBuffer((__gm__ T1 *)dv);
    queryRopeGm.SetGlobalBuffer((__gm__ T1 *)queryRope);
    keyRopeGm.SetGlobalBuffer((__gm__ T1 *)keyRope);
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
    actualCalcS1Token = constInfo.s1Token;
    actualCalcS2Token = constInfo.s2Token;

    prefixNAddr = prefixN;
    actualSeqQlenAddr = actualSeqQlen;
    actualSeqKvlenAddr = actualSeqKvlen;
    constInfo.seqS1_addr = actualSeqQlen;
    constInfo.seqS2_addr = actualSeqKvlen;
    queryRopeAddr = queryRope;
    keyRopeAddr = keyRope;

    InitWorkSpace(workspace, dq, dk, dv);
    InitUbBuffer();
    SetOptionalInfo();
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::GetSeqQlenKvlenByBidx(
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

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::InitWorkSpace(
    __gm__ uint8_t *workspace, __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv)
{
    // init workspace address
    if constexpr (!IS_FP32_INPUT) {
        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->postTilingData.dqWorkSpaceOffset / sizeof(T2));
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->postTilingData.dkWorkSpaceOffset / sizeof(T2));
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->postTilingData.dvWorkSpaceOffset / sizeof(T2));
        if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
            deterGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->postTilingData.deterGmOffset / sizeof(T2));
            deterOffsetGm.SetGlobalBuffer((__gm__ int64_t *)workspace + tilingData->postTilingData.deterWorkSpaceOffset / sizeof(int64_t));
        }
		if constexpr (IS_FP8_INPUT) {
			dsAmaxWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->postTilingData.vScaleDsWorkSpaceOffset / sizeof(T2));
		}
    } else {
        // input type fp32, dq dk dv write to output gm directly
        dqWorkSpaceGm.SetGlobalBuffer((__gm__ T1 *)dq);
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ T1 *)dk);
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ T1 *)dv);
        if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
            deterGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->postTilingData.deterGmOffset / sizeof(T2));
            deterOffsetGm.SetGlobalBuffer((__gm__ int64_t *)workspace + tilingData->postTilingData.deterWorkSpaceOffset / sizeof(int64_t));
        }  
    }
    if constexpr (IS_DROP) {
        if (tilingData->preTilingData.dropoutIsDivisibleBy8 == 0) {
            dropMaskWorkspaceGm.SetGlobalBuffer((__gm__ uint8_t *)workspace + tilingData->postTilingData.dropMaskGmOffset);
        }
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::InitUbBuffer()
{
    /**
     * UB划分，buffer大小分配
     * attenMaskOrYInQue: for y and attenMask
     * pseOrDyInQue: for dx and pse
     * dSOutQue: for dq dk left ub matrix
     * pOutQue: for dv left ub matrix
     * mm1ResBuf: for mm1 ub double buffer
     * mm2ResBuf: for mm2 ub double buffer
     * softmaxGradResBuf: for softmax_grad result
     * dropMaskBuf: for dropMask
     * maxSumQue: for max sum double buffer
     **/
    pipe->InitBuffer(attenMaskOrYInQue, 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(T2));
    pipe->InitBuffer(pseOrDyInQue, 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(OUTDTYPE));
    pipe->InitBuffer(mm1ResBuf[0], VECTOR_BASEM * VECTOR_BASEN * sizeof(T2));
    pipe->InitBuffer(mm1ResBuf[1], VECTOR_BASEM * VECTOR_BASEN * sizeof(T2));
    pipe->InitBuffer(mm2ResBuf[0], VECTOR_BASEM * VECTOR_BASEN * sizeof(T2));
    pipe->InitBuffer(mm2ResBuf[1], VECTOR_BASEM * VECTOR_BASEN * sizeof(T2));
    pipe->InitBuffer(softmaxGradResBuf, VECTOR_BASEM * sizeof(T2));
    pipe->InitBuffer(maxSumQue[0], 1, VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * 2);
    pipe->InitBuffer(maxSumQue[1], 1, VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * 2);
    if constexpr (IS_DROP) {
        pipe->InitBuffer(dropMaskBuf, VECTOR_BASEM * VECTOR_BASEN * sizeof(uint8_t) / 8);          // 1k
        pipe->InitBuffer(dropmaskIndexVecBuf, VECTOR_BASEM * VECTOR_BASEN / 16 * sizeof(int32_t)); // 2k
    }
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        pipe->InitBuffer(deterInOutQue, 1, DETER_DQ_UB_SIZE);
        pipe->InitBuffer(deterOffsetBuf, DETER_OFFSET_UB_SIZE); // 3k
    }
    if constexpr (IS_FP8_INPUT) {
        pipe->InitBuffer(vselrIndexesBuf, VECTOR_BASEN);
        LocalTensor<uint8_t> selrIndexesTensor = vselrIndexesBuf.Get<uint8_t>();
        for (int i = 0; i < VECTOR_BASEN; i++) {
            selrIndexesTensor.SetValue(i, i * 2);
        }
        pipe->InitBuffer(dsAmaxOutQue, 1, VREG_SIZE / 2);
    }
    if constexpr (!IS_FP32_INPUT) {
        pipe->InitBuffer(dSOutQue, 1, VECTOR_BASEM * VREG_SIZE + VREG_SIZE);
        pipe->InitBuffer(pOutQue, 1, VECTOR_BASEM * VREG_SIZE + VREG_SIZE);
    } else {
        // input type fp32, exceed ub size so need to reuse dSOutQue
        pipe->InitBuffer(dSOutQue, 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(T1) + VECTOR_BASEN * sizeof(T1));
        pOutQue = dSOutQue;
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::SetConstInfo()
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
        if constexpr (IS_ROPE) {
            constInfo.commonConstInfo.s1Dr = constInfo.commonConstInfo.s1Size * constInfo.dRopeSize;
            constInfo.commonConstInfo.gS1Dr = constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.s1Dr;
            constInfo.commonConstInfo.n2GS1Dr = constInfo.n2Size * constInfo.commonConstInfo.gS1Dr;
            constInfo.commonConstInfo.s2Dr = constInfo.commonConstInfo.s2Size * constInfo.dRopeSize;
            constInfo.commonConstInfo.n2S2Dr = constInfo.n2Size * constInfo.commonConstInfo.s2Dr;
            constInfo.commonConstInfo.gDr = constInfo.commonConstInfo.gSize * constInfo.dRopeSize;
            constInfo.commonConstInfo.n2Dr = constInfo.n2Size * constInfo.dRopeSize;
            constInfo.commonConstInfo.bN2Dr = constInfo.bSize * constInfo.commonConstInfo.n2Dr;
            constInfo.commonConstInfo.n2GDr = constInfo.commonConstInfo.n2G * constInfo.dRopeSize;
            constInfo.commonConstInfo.bN2GDr = constInfo.bSize * constInfo.commonConstInfo.n2GDr;
        }
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

    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        constInfo.deterConstInfo.noNeedDeter =
            static_cast<bool>(tilingData->s1s2BNGS1S2SplitCoreParams.noNeedDeter);
        constInfo.deterConstInfo.usedCubeCoreNum =
            static_cast<uint8_t>(tilingData->s1s2BNGS1S2SplitCoreParams.blockOuter);
        if (static_cast<uint8_t>(tilingData->s1s2BNGS1S2BaseParams.coreNum) == VEC_CORE_NUM_64) {
            // 64V核场景确定性计算中会用满V核
            constInfo.deterConstInfo.usedVectorCoreNum = static_cast<uint8_t>(tilingData->s1s2BNGS1S2BaseParams.coreNum);
        } else {
            // 非64V核场景确定性计算使用V核数量为往下32取整
            constInfo.deterConstInfo.usedVectorCoreNum = static_cast<uint8_t>(tilingData->s1s2BNGS1S2BaseParams.coreNum) / ALIGN_NUM_32 * ALIGN_NUM_32;
        }
        // 确定性计算中每个v核处理两行s1
        constInfo.deterConstInfo.eachVecCoreS1Offset =
            static_cast<uint8_t>(CUBE_BASEM / constInfo.deterConstInfo.usedVectorCoreNum);
        constInfo.deterConstInfo.eachVecCoreS2Offset =
            static_cast<uint8_t>(CUBE_BASEN / constInfo.deterConstInfo.usedVectorCoreNum);
        // 确定性计算中，V核开满64核，按照CUBE_BASEM=128的基本块处理是能够整除的
        constInfo.deterConstInfo.dqEachVectorSize =
            static_cast<uint32_t>(BASE_DQ_SIZE / constInfo.deterConstInfo.usedVectorCoreNum);
        constInfo.deterConstInfo.dkvEachVectorSize =
            static_cast<uint32_t>(BASE_DKV_SIZE / constInfo.deterConstInfo.usedVectorCoreNum);
        if constexpr (IS_TND) {
            constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.n2GDv;
            constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.n2Dv;
            constInfo.mm2Ka = constInfo.commonConstInfo.n2GD;
            constInfo.mm2Kb = constInfo.commonConstInfo.n2D;
            constInfo.deterConstInfo.deterN2Stride = constInfo.commonConstInfo.gD;
            constInfo.deterConstInfo.deterGStride = constInfo.commonConstInfo.dSize;
            constInfo.deterConstInfo.deterS1oStride = constInfo.commonConstInfo.n2GD;
        } else {
            if (constInfo.commonConstInfo.layoutType == BNGSD) {
                constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.dSizeV;
                constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.dSizeV;
                constInfo.mm2Ka = constInfo.commonConstInfo.dSize;
                constInfo.mm2Kb = constInfo.commonConstInfo.dSize;
                constInfo.deterConstInfo.deterBStride = constInfo.commonConstInfo.n2GS1D;
                constInfo.deterConstInfo.deterN2Stride = constInfo.commonConstInfo.gS1D;
                constInfo.deterConstInfo.deterGStride = constInfo.commonConstInfo.s1D;
                constInfo.deterConstInfo.deterS1oStride = constInfo.commonConstInfo.dSize;
            } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.bN2GDv;
                constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.bN2Dv;
                constInfo.mm2Ka = constInfo.commonConstInfo.bN2GD;
                constInfo.mm2Kb = constInfo.commonConstInfo.bN2D;
                constInfo.deterConstInfo.deterBStride = constInfo.commonConstInfo.n2GD;
                constInfo.deterConstInfo.deterN2Stride = constInfo.commonConstInfo.gD;
                constInfo.deterConstInfo.deterGStride = constInfo.commonConstInfo.dSize;
                constInfo.deterConstInfo.deterS1oStride = constInfo.commonConstInfo.bN2GD;
            } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.n2GDv;
                constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.n2Dv;
                constInfo.mm2Ka = constInfo.commonConstInfo.n2GD;
                constInfo.mm2Kb = constInfo.commonConstInfo.n2D;
                constInfo.deterConstInfo.deterBStride = constInfo.commonConstInfo.n2GS1D;
                constInfo.deterConstInfo.deterN2Stride = constInfo.commonConstInfo.gD;
                constInfo.deterConstInfo.deterGStride = constInfo.commonConstInfo.dSize;
                constInfo.deterConstInfo.deterS1oStride = constInfo.commonConstInfo.n2GD;
            }
        }
        constInfo.deterConstInfo.deterDqkSrcStride =
            static_cast<uint32_t>((HEAD_DIM_ALIGN - constInfo.commonConstInfo.dSize) / FLOAT_BLOCK_SIZE);
        constInfo.deterConstInfo.deterDvSrcStride =
            static_cast<uint32_t>((HEAD_DIM_ALIGN - constInfo.commonConstInfo.dSizeV) / FLOAT_BLOCK_SIZE);
        constInfo.deterConstInfo.deterDqDstStride =
            static_cast<uint32_t>((constInfo.mm2Ka - constInfo.commonConstInfo.dSize) * sizeof(T2));
        constInfo.deterConstInfo.deterDkDstStride =
            static_cast<uint32_t>((constInfo.mm2Kb - constInfo.commonConstInfo.dSize) * sizeof(T2));
        constInfo.deterConstInfo.deterDvDstStride =
            static_cast<uint32_t>((constInfo.commonConstInfo.mm1Kb - constInfo.commonConstInfo.dSizeV) * sizeof(T2));
        constInfo.deterConstInfo.deterVecCoreS1Offset =
            vBlockIdx * constInfo.deterConstInfo.eachVecCoreS1Offset * constInfo.mm2Ka;
        constInfo.deterConstInfo.deterDkVecCoreS2Offset =
            vBlockIdx * constInfo.deterConstInfo.eachVecCoreS2Offset * constInfo.mm2Kb;
        constInfo.deterConstInfo.deterDvVecCoreS2Offset =
            vBlockIdx * constInfo.deterConstInfo.eachVecCoreS2Offset * constInfo.commonConstInfo.mm1Kb;
        constInfo.deterConstInfo.eventIDScalarToMte2 =
            static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        constInfo.deterConstInfo.eventIDMte2ToScalar =
            static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        constInfo.deterConstInfo.eventIDScalarToMte3 =
            static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        constInfo.deterConstInfo.eventIDMte3ToScalar =
            static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        constInfo.deterConstInfo.eventIDMte3ToMte2 =
            static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        constInfo.deterConstInfo.eventIDMte2ToMte3 =
            static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        if constexpr (IS_ROPE) {
            constInfo.mm2Ka = constInfo.mm2Ka / 3 << 1;
            constInfo.mm2Kb = constInfo.mm2Kb / 3 << 1;
        }
    } else {
        if constexpr (IS_TND) {
            constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.n2GDv;
            constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.n2Dv;
            constInfo.mm2Ka = constInfo.commonConstInfo.n2GD;
            constInfo.mm2Kb = constInfo.commonConstInfo.n2D;
        } else {
            if (constInfo.commonConstInfo.layoutType == BNGSD) {
                constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.dSizeV;
                constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.dSizeV;
                constInfo.mm2Ka = constInfo.commonConstInfo.dSize;
                constInfo.mm2Kb = constInfo.commonConstInfo.dSize;
            } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.bN2GDv;
                constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.bN2Dv;
                constInfo.mm2Ka = constInfo.commonConstInfo.bN2GD;
                constInfo.mm2Kb = constInfo.commonConstInfo.bN2D;
            } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.n2GDv;
                constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.n2Dv;
                constInfo.mm2Ka = constInfo.commonConstInfo.n2GD;
                constInfo.mm2Kb = constInfo.commonConstInfo.n2D;
            }
        }
        if constexpr (IS_ROPE) {
            constInfo.mm2Ka = constInfo.mm2Ka / 3 << 1;
            constInfo.mm2Kb = constInfo.mm2Kb / 3 << 1;
        }
    }
    constInfo.commonConstInfo.subBlockIdx = vSubBlockIdx;

    uint32_t tmp = 0xFF7FFFFF;
    constInfo.attenMaskMinValue = *((float *)&tmp);
    constInfo.commonConstInfo.keepProb = tilingData->s1s2BNGS1S2BaseParams.keepProb;

    constInfo.sfmgMaxLoopSize = VECTOR_BASEM * VECTOR_BASEN / HEAD_DIM_ALIGN; // softmaxGrad每次最大能处理的m轴大小
    constInfo.dAlignToBlock = AlignTo(constInfo.commonConstInfo.dSizeV, INPUT_BLOCK_NUM);
    constInfo.dAlignToBlockForFp8 = AlignTo(constInfo.commonConstInfo.dSizeV, INPUT_BLOCK_NUM_FOR_FP8);
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::SetOptionalInfo()
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

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::SetRunInfo(
    FagRunInfo &runInfo, int64_t taskId, int64_t index)
{
    runInfo.s2CvBegin = s2CvBegin;
    runInfo.s2CvEnd = s2CvEnd;
    if constexpr (IS_TSCM_PRELOAD) {
        runInfo.qDxPingPongIdx = taskId % 3;
    } else {
        runInfo.qDxPingPongIdx = taskId & 1;
    }
    if constexpr (IS_TND) {
        int64_t resbaseIdx = index;
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        runInfo.lastBatchTotalBaseIdx = 0;
        runInfo.lastBatchTotalS1BOffset = 0;
        runInfo.lastBatchTotalS2BOffset = 0;
        runInfo.lastBatchTotalS1BOffsetForDv = 0;
        runInfo.lastBatchTotalS2BOffsetForDv = 0;
        runInfo.lastBatchTotalS1S2SizeAlign = 0;
        runInfo.lastBatchTotalS1S2Size = 0;
        runInfo.lastBatchTotalS2Size = 0;
        for (int64_t bIdx = 0; bIdx < constInfo.bSize; bIdx++) {
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
                break;
            } else {
                runInfo.lastBatchTotalBaseIdx += totalBaseIdx;
                resbaseIdx = index - runInfo.lastBatchTotalBaseIdx;
                runInfo.lastBatchTotalS1BOffset += actualS1Len * constInfo.commonConstInfo.n2GD;
                runInfo.lastBatchTotalS2BOffset += actualS2Len * constInfo.commonConstInfo.n2D;
                runInfo.lastBatchTotalS1BOffsetForDv += actualS1Len * constInfo.commonConstInfo.n2GDv;
                runInfo.lastBatchTotalS2BOffsetForDv += actualS2Len * constInfo.commonConstInfo.n2Dv;
                runInfo.lastBatchTotalS1S2SizeAlign += actualS1Len * AlignTo16(actualS2Len);
                runInfo.lastBatchTotalS1S2Size += actualS1Len * actualS2Len;
                runInfo.lastBatchTotalS2Size += actualS2Len;
                if constexpr (IS_ROPE) {
                    runInfo.lastBatchTotalS1BRopeOffset += actualS1Len * constInfo.commonConstInfo.n2GDr;
                    runInfo.lastBatchTotalS2BRopeOffset += actualS2Len * constInfo.commonConstInfo.n2Dr;
                }
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
        runInfo.halfS2RealSize = (runInfo.commonRunInfo.s2RealSize + 1) >> 1;
        runInfo.firstHalfS2RealSize = runInfo.halfS2RealSize;
        runInfo.commonRunInfo.halfS1RealSize = (runInfo.commonRunInfo.s1RealSize + 1) >> 1;
        runInfo.commonRunInfo.firstHalfS1RealSize = runInfo.commonRunInfo.halfS1RealSize;
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
        runInfo.commonRunInfo.vecCoreOffset = vSubBlockIdx * runInfo.commonRunInfo.firstHalfS1RealSize;
        runInfo.commonRunInfo.b1SSOffsetAlign = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.s1Size *
                                                AlignTo16(constInfo.commonConstInfo.s2Size);
        runInfo.commonRunInfo.preTokensPerBatch = attenMaskInfo.preTokens;
        runInfo.commonRunInfo.nextTokensPerBatch = attenMaskInfo.nextTokens;
    }

    runInfo.isS2IdxNoChange = (lastS2oCvDimIdx == runInfo.s2oIdx && lastBdimIdx == runInfo.commonRunInfo.boIdx &&
                               lastN2dimIdx == runInfo.commonRunInfo.n2oIdx);
    if (!runInfo.isS2IdxNoChange) {
        lastS2oCvDimIdx = runInfo.s2oIdx;
        lastBdimIdx = runInfo.commonRunInfo.boIdx;
        lastN2dimIdx = runInfo.commonRunInfo.n2oIdx;
    }
	if constexpr (IS_FP8_INPUT) {
		int64_t deqScaleQGmOffset = GetDeqScaleQOffset(runInfo);
		int64_t deqScaleKGmOffset = GetDeqScaleKOffset(runInfo);
		runInfo.quantScaleInfo.deqScaleQValue = deqScaleQGm.GetValue(deqScaleQGmOffset);
		runInfo.quantScaleInfo.deqScaleKValue = deqScaleKGm.GetValue(deqScaleKGmOffset);
		runInfo.quantScaleInfo.deqScaleVValue = deqScaleVGm.GetValue(deqScaleKGmOffset);
		runInfo.quantScaleInfo.deqScaleDyValue = deqScaleDyGm.GetValue(deqScaleQGmOffset);
	}
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::GetIsNeedDeter(int64_t computeLoopIdx)
{
    if (constInfo.deterConstInfo.noNeedDeter) {
        return;
    }
    if (unlikely(computeLoopIdx >= MAX_BITS_IN_TILING)) {
        dqIsNeedDeter[computeLoopIdx & 1] = true;
        dkDvIsNeedDeter[computeLoopIdx & 1] = true;
    } else {
        // caculate index and bit position
        int64_t arrayIndex = computeLoopIdx / BITS_EACH_UINT64;
        int64_t bitShift = computeLoopIdx % BITS_EACH_UINT64;
        uint64_t mask = 1ULL << bitShift;
        dqIsNeedDeter[computeLoopIdx & 1] = (tilingData->s1s2BNGS1S2SplitCoreParams.dqIsNeedDeter[arrayIndex] & mask) != 0;
        dkDvIsNeedDeter[computeLoopIdx & 1] = (tilingData->s1s2BNGS1S2SplitCoreParams.dkDvIsNeedDeter[arrayIndex] & mask) != 0;
    }
}


FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::ProcessDeter()
{
    // maxValidBBLen表示所有核中处理有效块最多的数量，需要做这么多次确定性计算
    int64_t maxValidBBLen = tilingData->s1s2BNGS1S2SplitCoreParams.maxValidBBLen;
    int64_t remainLoopNum = maxValidBBLen;
    int64_t blockInnerIdx = 0;
    int64_t nextValidBlockInnerIdx = 0;
    int64_t taskId = 0;
    int64_t loopIdx = 0;
    FagRunInfo runInfos[2]; // for ping pong

    while (remainLoopNum > 0) { // remainLoop每次做完确定性计算减1
        blockInnerIdx = tilingData->s1s2BNGS1S2BlockNumList.blockStarts[cBlockIdx] + loopIdx;
        if (tilingData->s1s2BNGS1S2BlockNumList.blockEnds[cBlockIdx] != 0) { // 有主流程需要处理的核
            if (blockInnerIdx < tilingData->s1s2BNGS1S2BlockNumList.blockEnds[cBlockIdx] +
                                    1) { // 这个1是提前发射mm1mm2，所以需要额外增加一个轮次
                isLastLoop = (blockInnerIdx == tilingData->s1s2BNGS1S2BlockNumList.blockEnds[cBlockIdx]);
                // 无效块跳过
                if (!isLastLoop && !IsValid(runInfos[taskId & 1], blockInnerIdx)) {
                    loopIdx++;
                    continue;
                }
                if (taskId > 0) {
                    deterPpFlag = 1 - deterPpFlag;
                    WaitMm1Mm2Result();
                    ProcessSoftmaxGrad(runInfos[(taskId + 1) & 1]); // softmaxGrad
                }
                if (!isLastLoop) {
                    SetRunInfo(runInfos[taskId & 1], taskId, blockInnerIdx);
                    // get mm1 mm2 next valid block index and next s2 begin end
                    nextValidBlockInnerIdx = GetNextValidIdx(runInfos[(taskId + 1) & 1], blockInnerIdx + 1);
                    IterateMm1Mm2(runInfos[taskId & 1], nextValidBlockInnerIdx, taskId, maxValidBBLen - taskId);
                    CopyInMaxSum<T2, VECTOR_BASEM>(constInfo, runInfos[taskId & 1], maxSumQue[taskId & 1], softmaxMaxGm,
                                                   softmaxSumGm);
                }
                if (taskId > 0) {
                    ProcessReCompute(runInfos[(taskId + 1) & 1]);
                    GetIsNeedDeter(taskId - 1);
                    IterateMm3Mm4Mm5(runInfos[(taskId + 1) & 1]);
                    if (taskId > 1) {
                        // 确定性计算的基本块滞后于上面的mm345一次
                        WaitMm3Result(maxValidBBLen - remainLoopNum);
                        DeterCompute(false, maxValidBBLen - remainLoopNum, remainLoopNum);
                        remainLoopNum--;
                    }
                }
                if (isLastLoop && taskId > 0) {
                    WaitMm3Result(maxValidBBLen - remainLoopNum);
                    DeterCompute(true, maxValidBBLen - remainLoopNum, remainLoopNum);
                    remainLoopNum--;
                }
                taskId++;
            } else { // 主流程做完的核，后面需要参与到确定性计算中
                int64_t computeLoopIdx = maxValidBBLen - remainLoopNum;
                GetIsNeedDeter(computeLoopIdx);
                WriteOffsetToGM(OUTINDEX, OUTINDEX, OUTINDEX, computeLoopIdx, remainLoopNum);
                deterPpFlag = 1 - deterPpFlag;
                DeterCompute(true, computeLoopIdx, remainLoopNum);
                remainLoopNum--;
            }
        } else { // 没有主流程需要处理的核，只参与确定性计算
            GetIsNeedDeter(loopIdx);
            deterPpFlag = 1 - deterPpFlag;
            DeterCompute(true, loopIdx, remainLoopNum);
            remainLoopNum--;
        }
        loopIdx++;
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::ProcessNormal()
{
    if (tilingData->s1s2BNGS1S2BlockNumList.blockEnds[cBlockIdx] == 0) {
        return;
    }
    int64_t taskId = 0;
    FagRunInfo runInfos[2]; // for ping pong
    int64_t nextValidBlockInnerIdx = 0;
    int64_t blockInnerIdx = 0;
    int64_t dqkvBlockInnerIdx = 0;
    int64_t curLoopIdx = 0; // just for continuous split core
    nextValidBlockInnerIdx =
        GetNextValidIdx(runInfos[0], tilingData->s1s2BNGS1S2BlockNumList.blockStarts[cBlockIdx], curLoopIdx);
    blockInnerIdx = nextValidBlockInnerIdx;
    while (true) {
        isLastLoop = (blockInnerIdx == -1);
        dqkvBlockInnerIdx = blockInnerIdx; // save for dq dk dv next valid block index
        if (taskId > 0) {
            ProcessSoftmaxGrad(runInfos[(taskId + 1) & 1]); // softmaxGrad
            WaitMm1Mm2Result();
        }
        if (!isLastLoop) {
            SetRunInfo(runInfos[taskId & 1], taskId, blockInnerIdx);
            if (tilingData->s1s2BNGS1S2BaseParams.isSplitByBlockIdx) {
                curLoopIdx++;
            } else {
                blockInnerIdx++;
            }
            // get mm1 mm2 next valid block index and next s2 begin end
            nextValidBlockInnerIdx = GetNextValidIdx(runInfos[(taskId + 1) & 1], blockInnerIdx, curLoopIdx);
            IterateMm1Mm2(runInfos[taskId & 1], nextValidBlockInnerIdx);
            CopyInMaxSum<T2, VECTOR_BASEM>(constInfo, runInfos[taskId & 1], maxSumQue[taskId & 1], softmaxMaxGm,
                                           softmaxSumGm);
        }
        if (taskId > 0) {
            ProcessReCompute(runInfos[(taskId + 1) & 1]);
            IterateMm3Mm4Mm5(runInfos[(taskId + 1) & 1], dqkvBlockInnerIdx);
        }
        if (blockInnerIdx == -1) {
            break;
        }
        taskId++;
        blockInnerIdx = nextValidBlockInnerIdx;
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::Process()
{
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        ProcessDeter();
    } else {
        ProcessNormal();
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::WaitMm1Mm2Result()
{
    mm1.WaitIterateAll();
    mm1.WaitIterateAll();
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void 
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::WaitMm3Result(int64_t computeBlockIdx)
{
    if (dqIsNeedDeter[computeBlockIdx & 1] && !dkDvIsNeedDeter[computeBlockIdx & 1]) {
        mm2.WaitIterateAll();
    } else {
        mm3.WaitIterateAll();
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::GetQueryOffset(FagRunInfo &runInfo, bool isMm1Mm2)
{
    int64_t leftMatrixOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    int64_t n2GD = constInfo.commonConstInfo.n2GD;
    int64_t gD = constInfo.commonConstInfo.gD;
    int64_t dSize = constInfo.commonConstInfo.dSize;
    int64_t n2GS1D = constInfo.commonConstInfo.n2GS1D;
    int64_t gS1D = constInfo.commonConstInfo.gS1D;
    int64_t s1D = constInfo.commonConstInfo.s1D;
    int64_t bN2GD = constInfo.commonConstInfo.bN2GD;
    int64_t bOffsetTmp = runInfo.lastBatchTotalS1BOffset;
    if constexpr (IS_ROPE) {
        if (isMm1Mm2) {
            n2GD = (constInfo.commonConstInfo.n2GD / ROPE_D_RATIO) << 1;
            gD = (constInfo.commonConstInfo.gD / ROPE_D_RATIO) << 1;
            dSize = (constInfo.commonConstInfo.dSize / ROPE_D_RATIO) << 1;
            n2GS1D = (constInfo.commonConstInfo.n2GS1D / ROPE_D_RATIO) << 1;
            gS1D = (constInfo.commonConstInfo.gS1D / ROPE_D_RATIO) << 1;
            s1D = (constInfo.commonConstInfo.s1D / ROPE_D_RATIO) << 1;
            bN2GD = (constInfo.commonConstInfo.bN2GD / ROPE_D_RATIO) << 1;
            bOffsetTmp = (bOffsetTmp / ROPE_D_RATIO) << 1;
        }
    }

    if constexpr (IS_TND) {
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        bOffset = bOffsetTmp;
        s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * n2GD;
        n2Offset = runInfo.commonRunInfo.n2oIdx * gD;
        gOffset = runInfo.commonRunInfo.goIdx * dSize;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            bOffset = runInfo.commonRunInfo.boIdx * n2GS1D;
            n2Offset = runInfo.commonRunInfo.n2oIdx * gS1D;
            gOffset = runInfo.commonRunInfo.goIdx * s1D;
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * dSize;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * bN2GD;
            bOffset = runInfo.commonRunInfo.boIdx * n2GD;
            n2Offset = runInfo.commonRunInfo.n2oIdx * gD;
            gOffset = runInfo.commonRunInfo.goIdx * dSize;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            bOffset = runInfo.commonRunInfo.boIdx * n2GS1D;
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * n2GD;
            n2Offset = runInfo.commonRunInfo.n2oIdx * gD;
            gOffset = runInfo.commonRunInfo.goIdx * dSize;
        }
    }
    leftMatrixOffset = bOffset + n2Offset + gOffset + s1Offset;
    return leftMatrixOffset;
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::GetQueryRopeOffset(FagRunInfo &runInfo)
{
    int64_t leftMatrixOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    if constexpr (IS_TND) {
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        bOffset = runInfo.lastBatchTotalS1BRopeOffset;
        s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GDr;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gDr;
        gOffset = runInfo.commonRunInfo.goIdx * constInfo.dRopeSize;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GS1Dr;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gS1Dr;
            gOffset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.s1Dr;
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.dRopeSize;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.commonConstInfo.bN2GDr;
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GDr;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gDr;
            gOffset = runInfo.commonRunInfo.goIdx * constInfo.dRopeSize;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GS1Dr;
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GDr;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gDr;
            gOffset = runInfo.commonRunInfo.goIdx * constInfo.dRopeSize;
        }
    }
    leftMatrixOffset = bOffset + n2Offset + gOffset + s1Offset;
    return leftMatrixOffset;
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::GetDxOffset(FagRunInfo &runInfo)
{
    int64_t leftMatrixOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    if constexpr (IS_TND) {
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        bOffset = runInfo.lastBatchTotalS1BOffsetForDv;
        s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GDv;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gDv;
        gOffset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.dSizeV;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GS1Dv;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gS1Dv;
            gOffset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.s1Dv;
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.commonConstInfo.dSizeV;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.commonConstInfo.bN2GDv;
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GDv;
            n2Offset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.dSizeV;
            gOffset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gDv;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GS1Dv;
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GDv;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gDv;
            gOffset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.dSizeV;
        }
    }
    leftMatrixOffset = bOffset + n2Offset + gOffset + s1Offset;
    return leftMatrixOffset;
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::GetKeyOffset(FagRunInfo &runInfo, bool isMm1Mm2)
{
    int64_t rightMatrixOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;

    int64_t n2D = constInfo.commonConstInfo.n2D;
    int64_t n2S2D = constInfo.commonConstInfo.n2S2D;
    int64_t dSize = constInfo.commonConstInfo.dSize;
    int64_t bN2D = constInfo.commonConstInfo.bN2D;
    int64_t s2D = constInfo.commonConstInfo.s2D;
    int64_t s1D = constInfo.commonConstInfo.s1D;
    int64_t bN2GD = constInfo.commonConstInfo.bN2GD;
    int64_t bOffsetTmp = runInfo.lastBatchTotalS2BOffset;
    if constexpr (IS_ROPE) {
        if (isMm1Mm2) {
            n2D = (constInfo.commonConstInfo.n2D / ROPE_D_RATIO) << 1;
            n2S2D = (constInfo.commonConstInfo.n2S2D / ROPE_D_RATIO) << 1;
            dSize = (constInfo.commonConstInfo.dSize / ROPE_D_RATIO) << 1;
            bN2D = (constInfo.commonConstInfo.bN2D / ROPE_D_RATIO) << 1;
            s2D = (constInfo.commonConstInfo.s2D / ROPE_D_RATIO) << 1;
            s1D = (constInfo.commonConstInfo.s1D / ROPE_D_RATIO) << 1;
            bN2GD = (constInfo.commonConstInfo.bN2GD / ROPE_D_RATIO) << 1;
            bOffsetTmp = (bOffsetTmp / ROPE_D_RATIO) << 1;
        }
    }

    if constexpr (IS_TND) {
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        bOffset = bOffsetTmp;
        s2Offset = runInfo.s2CvBegin * n2D;
        n2Offset = runInfo.commonRunInfo.n2oIdx * dSize;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            bOffset = runInfo.commonRunInfo.boIdx * n2S2D;
            n2Offset = runInfo.commonRunInfo.n2oIdx * s2D;
            s2Offset = runInfo.s2CvBegin * dSize;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            s2Offset = runInfo.s2CvBegin * bN2D;
            bOffset = runInfo.commonRunInfo.boIdx * n2D;
            n2Offset = runInfo.commonRunInfo.n2oIdx * dSize;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            bOffset = runInfo.commonRunInfo.boIdx * n2S2D;
            s2Offset = runInfo.s2CvBegin * n2D;
            n2Offset = runInfo.commonRunInfo.n2oIdx * dSize;
        }
    }
    rightMatrixOffset = bOffset + n2Offset + s2Offset;
    return rightMatrixOffset;
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::GetKeyRopeOffset(FagRunInfo &runInfo)
{
    int64_t rightMatrixOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    if constexpr (IS_TND) {
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        bOffset = runInfo.lastBatchTotalS2BRopeOffset;
        s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.n2Dr;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.dRopeSize;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2S2Dr;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.s2Dr;
            s2Offset = runInfo.s2CvBegin * constInfo.dRopeSize;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.bN2Dr;
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2Dr;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.dRopeSize;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2S2Dr;
            s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.n2Dr;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.dRopeSize;
        }
    }
    rightMatrixOffset = bOffset + n2Offset + s2Offset;
    return rightMatrixOffset;
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::GetValueOffset(FagRunInfo &runInfo)
{
    int64_t rightMatrixOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    if constexpr (IS_TND) {
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        bOffset = runInfo.lastBatchTotalS2BOffsetForDv;
        s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.n2Dv;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.dSizeV;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2S2Dv;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.s2Dv;
            s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.dSizeV;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.bN2Dv;
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2Dv;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.dSizeV;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2S2Dv;
            s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.n2Dv;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.dSizeV;
        }
    }
    rightMatrixOffset = bOffset + n2Offset + s2Offset;
    return rightMatrixOffset;
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::GetDeqScaleQOffset(FagRunInfo &runInfo)
{
    int64_t scaleOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    int64_t scaleNumPerS1 = Ceil<int64_t>(constInfo.commonConstInfo.s1Size, CUBE_BASEM);
    bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2G * scaleNumPerS1;
    n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gSize * scaleNumPerS1;
    gOffset = runInfo.commonRunInfo.goIdx * scaleNumPerS1;
    s1Offset = runInfo.commonRunInfo.s1oIdx;
    scaleOffset = bOffset + n2Offset + gOffset + s1Offset;
    return scaleOffset;
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::GetDeqScaleKOffset(FagRunInfo &runInfo)
{
    int64_t scaleOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    int64_t scaleNumPerS2 = Ceil<int64_t>(constInfo.commonConstInfo.s2Size, CUBE_BASEN);
    bOffset = runInfo.commonRunInfo.boIdx * constInfo.n2Size * scaleNumPerS2;
    n2Offset = runInfo.commonRunInfo.n2oIdx * scaleNumPerS2;
    s2Offset = runInfo.s2oIdx;
    scaleOffset = bOffset + n2Offset + s2Offset;
    return scaleOffset;
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::WriteOffsetToGM(
    int64_t queryGmOffset, int64_t keyGmOffset, int64_t valueGmOffset, int64_t computeBlockIdx, int64_t remainLoopNum)
{
    if (constInfo.deterConstInfo.noNeedDeter) {
        return;
    }

    if (vSubBlockIdx == 1) {
        return;
    }
    LocalTensor<int64_t> deterOffsetTensor = deterOffsetBuf.Get<int64_t>();
    if (computeBlockIdx > 0) {
        WaitFlag<HardEvent::MTE3_S>(constInfo.deterConstInfo.eventIDMte3ToScalar);
    }
    deterOffsetTensor.SetValue(0, queryGmOffset);
    deterOffsetTensor.SetValue(4, keyGmOffset);
    deterOffsetTensor.SetValue(8, valueGmOffset);

    SetFlag<HardEvent::S_MTE3>(constInfo.deterConstInfo.eventIDScalarToMte3);
    WaitFlag<HardEvent::S_MTE3>(constInfo.deterConstInfo.eventIDScalarToMte3);
    DataCopy(deterOffsetGm[(computeBlockIdx * constInfo.deterConstInfo.usedCubeCoreNum + cBlockIdx) * 4 * 3],
                deterOffsetTensor, {1, 3, 0, 0});
    if (remainLoopNum > 1) {
        SetFlag<HardEvent::MTE3_S>(constInfo.deterConstInfo.eventIDMte3ToScalar);
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::GetNextValidIdx(FagRunInfo &runInfo,
                                                                                                int64_t blockInnerIdx,
                                                                                                int64_t curLoopIdx)
{
    if (!tilingData->s1s2BNGS1S2BaseParams.isSplitByBlockIdx) {
        int64_t nextValidBlockInnerIdx = blockInnerIdx;
        while (!IsValid(runInfo, nextValidBlockInnerIdx)) {
            if (nextValidBlockInnerIdx >= tilingData->s1s2BNGS1S2BlockNumList.blockEnds[cBlockIdx]) {
                return -1;
            }
            nextValidBlockInnerIdx++;
        }
        if (nextValidBlockInnerIdx >= tilingData->s1s2BNGS1S2BlockNumList.blockEnds[cBlockIdx]) {
            return -1;
        }
        return nextValidBlockInnerIdx;
    } else {
        return GetNextValidIdxFromFormula(curLoopIdx);
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::GetNextValidIdxFromFormula(int64_t loopIdx)
{
    int64_t continuousBlockNum = tilingData->s1s2BNGS1S2SplitCoreParams.maxValidBBLen > MAX_CONTINUOUS_BLOCK_NUM ?
                                     MAX_CONTINUOUS_BLOCK_NUM :
                                     tilingData->s1s2BNGS1S2SplitCoreParams.maxValidBBLen;
    int64_t blockGroupIdx = loopIdx / continuousBlockNum;      // 第几组
    int64_t blockGroupInnerIdx = loopIdx % continuousBlockNum; // 组内第几个

    int64_t globalIdx = (tilingData->s1s2BNGS1S2BaseParams.coreNum >> 1) * continuousBlockNum * blockGroupIdx +
                        cBlockIdx * continuousBlockNum + blockGroupInnerIdx;
    int64_t totalPerBatchNum = 0;
    if constexpr (!IS_ATTEN_MASK) {
        totalPerBatchNum = constInfo.s1Outer * constInfo.s2Outer;
    } else {
        if (constInfo.s1Token >= constInfo.commonConstInfo.s1Size &&
            constInfo.s2Token >= constInfo.commonConstInfo.s2Size) {
            totalPerBatchNum = constInfo.s1Outer * constInfo.s2Outer;
        } else {
            totalPerBatchNum = (((constInfo.s1Outer << 1) - constInfo.s2Outer + 1) * constInfo.s2Outer) >> 1;
        }
    }

    int64_t bIdx = globalIdx / totalPerBatchNum;
    if (bIdx >= constInfo.bSize * constInfo.n2Size * constInfo.commonConstInfo.gSize) {
        return -1;
    }

    int64_t gDimTail = globalIdx % totalPerBatchNum;
    int64_t s2Idx = 0;
    int64_t s1Idx = 0;
    if constexpr (!IS_ATTEN_MASK) {
        s2Idx = gDimTail / constInfo.s1Outer;
        s1Idx = gDimTail % constInfo.s1Outer;
    } else {
        if (constInfo.s1Token >= constInfo.commonConstInfo.s1Size &&
            constInfo.s2Token >= constInfo.commonConstInfo.s2Size) {
            s2Idx = gDimTail / constInfo.s1Outer;
            s1Idx = gDimTail % constInfo.s1Outer;
        } else {
            float sqrt_delta = sqrt(((constInfo.s1Outer << 1) - 1) * (((constInfo.s1Outer << 1) - 1)) +
                                         ((constInfo.s1Outer - 1 - gDimTail) << 3));
            s2Idx = Ceil<int64_t>(((constInfo.s1Outer << 1) - 1) - sqrt_delta, 2);
            s1Idx = gDimTail - ((((constInfo.s1Outer << 1) - 1 - s2Idx) * s2Idx) >> 1);
        }
    }

    s2CvBegin = s2Idx * CUBE_BASEN;
    s2CvEnd = s2CvBegin + CUBE_BASEN;     // 非尾块s2按照+CUBE_BASEN处理
    if (s2Idx == constInfo.s2Outer - 1) { // 默认s2 cv tail相等
        s2CvEnd = s2CvBegin + constInfo.s2Tail;
    }
    return bIdx * constInfo.s1Outer * constInfo.s2Outer + s2Idx * constInfo.s1Outer + s1Idx;
} 

FAG_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::SetTscmPreloadFlag(
    FagRunInfo &runInfo, FagTscmFlagData &flag, int64_t nextIndex, uint64_t& nextDxOffset, uint64_t& nextQueryOffset)
{
    if constexpr (IS_TSCM_PRELOAD) {
        flag.copyNext = !(nextIndex == -1);
        flag.copyCurrent = (runInfo.commonRunInfo.taskId == 0);
        if (!flag.copyNext) {
            return;
        }

        int64_t nextBoIdx = 0;
        int64_t bDimTail = 0;
        int64_t nextN2oIdx = 0;
        int64_t n2DimTail = 0;
        int64_t nextGoIdx = 0;
        int64_t gDimTail = 0;
        int64_t nextS1oIdx = 0;
 
        int64_t bOffset = 0;
        int64_t n2Offset = 0;
        int64_t gOffset = 0;
        int64_t s1Offset = 0;
        int64_t bOffsetDv = 0;
        int64_t n2OffsetDv = 0;
        int64_t gOffsetDv = 0;
        int64_t s1OffsetDv = 0;
        
        if constexpr (IS_TND && IS_DETER_NEW(DETER_SPARSE_TYPE)) {
            int64_t actualS1Len = coordinateInfos[nextIndex].actualS1Len;
            int64_t s1OuterTmp = coordinateInfos[nextIndex].s1Outer;
            int64_t bIdx = coordinateInfos[nextIndex].batchId;
            int64_t seqQLenPrefix = bIdx == 0 ? 0 : ((__gm__ int64_t *)actualSeqQlenAddr)[bIdx - 1];
            int64_t lastBatchTotalS1BOffset = seqQLenPrefix * constInfo.commonConstInfo.n2GD;
            int64_t lastBatchTotalS1BOffsetForDv = seqQLenPrefix * constInfo.commonConstInfo.n2GDv;
            int64_t s1CvTail = actualS1Len - (s1OuterTmp - 1) * CUBE_BASEM;

            nextS1oIdx = coordinateInfos[nextIndex].s1Idx;
            nextN2oIdx = coordinateInfos[nextIndex].n2Idx;
            nextGoIdx = coordinateInfos[nextIndex].gIdx;
            bOffset = lastBatchTotalS1BOffset;
            s1Offset = nextS1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GD;
            n2Offset = nextN2oIdx * constInfo.commonConstInfo.gD;
            gOffset = nextGoIdx * constInfo.commonConstInfo.dSize;
            if constexpr (IS_D_NO_EQUAL) {
                bOffsetDv = lastBatchTotalS1BOffsetForDv;
                s1OffsetDv = nextS1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GDv;
                n2OffsetDv = nextN2oIdx * constInfo.commonConstInfo.gDv;
                gOffsetDv = nextGoIdx * constInfo.commonConstInfo.dSizeV;
            }
            flag.nextMorN = (nextS1oIdx == s1OuterTmp - 1) ? s1CvTail : CUBE_BASEM;
        } else if constexpr (IS_TND) {
            int64_t lastBatchTotalS1BOffset = runInfo.lastBatchTotalS1BOffset;
            int64_t lastBatchTotalS1BOffsetForDv = runInfo.lastBatchTotalS1BOffsetForDv;
            int64_t lastBatchTotalBaseIdx = runInfo.lastBatchTotalBaseIdx;
            int64_t resbaseIdx = nextIndex - lastBatchTotalBaseIdx;
            int64_t actualS1Len = 0;
            int64_t actualS2Len = 0;
            int64_t s1CvTail = 0;
            int64_t s1OuterTmp = 0;
            int64_t s2OuterTmp = 0;
            for (int64_t bIdx = runInfo.lastBatchIdx; bIdx < constInfo.bSize; bIdx++) {
                GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
                s1OuterTmp = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
                s2OuterTmp = (actualS2Len + VECTOR_BASEN - 1) / VECTOR_BASEN;
                int64_t totalBaseIdx = constInfo.commonConstInfo.n2G * s1OuterTmp * s2OuterTmp;
                if (resbaseIdx < totalBaseIdx) {
                    s1CvTail = actualS1Len - (s1OuterTmp - 1) * CUBE_BASEM;
                    nextBoIdx = bIdx;
                    bDimTail = resbaseIdx;
                    nextN2oIdx = bDimTail / (constInfo.commonConstInfo.gSize * s1OuterTmp * s2OuterTmp);
                    n2DimTail = bDimTail % (constInfo.commonConstInfo.gSize * s1OuterTmp * s2OuterTmp);
                    nextGoIdx = n2DimTail / (s1OuterTmp * s2OuterTmp);
                    gDimTail = n2DimTail % (s1OuterTmp * s2OuterTmp);
                    nextS1oIdx = gDimTail % s1OuterTmp;
                    break;
                } else {
                    lastBatchTotalBaseIdx += totalBaseIdx;
                    resbaseIdx = nextIndex - lastBatchTotalBaseIdx;
                    lastBatchTotalS1BOffset += actualS1Len * constInfo.commonConstInfo.n2GD;
                    lastBatchTotalS1BOffsetForDv += actualS1Len * constInfo.commonConstInfo.n2GDv;
                }
            }
            bOffset = lastBatchTotalS1BOffset;
            s1Offset = nextS1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GD;
            n2Offset = nextN2oIdx * constInfo.commonConstInfo.gD;
            gOffset = nextGoIdx * constInfo.commonConstInfo.dSize;
            if constexpr (IS_D_NO_EQUAL) {
                bOffsetDv = lastBatchTotalS1BOffsetForDv;
                s1OffsetDv = nextS1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GDv;
                n2OffsetDv = nextN2oIdx * constInfo.commonConstInfo.gDv;
                gOffsetDv = nextGoIdx * constInfo.commonConstInfo.dSizeV;
            }
            flag.nextMorN = (nextS1oIdx == s1OuterTmp - 1) ? s1CvTail : CUBE_BASEM;
        } else {
            nextBoIdx = nextIndex / constInfo.n2GS1oS2o;
            bDimTail = nextIndex % constInfo.n2GS1oS2o;
            nextN2oIdx = bDimTail / constInfo.gS1oS2o;
            n2DimTail = bDimTail % constInfo.gS1oS2o;
            nextGoIdx = n2DimTail / constInfo.s1oS2o;
            gDimTail = n2DimTail % constInfo.s1oS2o;
            nextS1oIdx = gDimTail % constInfo.s1Outer;
            if (constInfo.commonConstInfo.layoutType == BNGSD) {
                if constexpr (IS_D_NO_EQUAL) {
                    bOffsetDv = nextBoIdx * constInfo.commonConstInfo.n2GS1Dv;
                    n2OffsetDv = nextN2oIdx * constInfo.commonConstInfo.gS1Dv;
                    gOffsetDv = nextGoIdx * constInfo.commonConstInfo.s1Dv;
                    s1OffsetDv = nextS1oIdx * CUBE_BASEM * constInfo.commonConstInfo.dSizeV;
                }

                bOffset = nextBoIdx * constInfo.commonConstInfo.n2GS1D;
                n2Offset = nextN2oIdx * constInfo.commonConstInfo.gS1D;
                gOffset = nextGoIdx * constInfo.commonConstInfo.s1D;
                s1Offset = nextS1oIdx * CUBE_BASEM * constInfo.commonConstInfo.dSize;
            } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                if constexpr (IS_D_NO_EQUAL) {
                    s1OffsetDv = nextS1oIdx * CUBE_BASEM * constInfo.commonConstInfo.bN2GDv;
                    bOffsetDv = nextBoIdx * constInfo.commonConstInfo.n2GDv;
                    n2OffsetDv = nextGoIdx * constInfo.commonConstInfo.dSizeV;
                    gOffsetDv = nextN2oIdx * constInfo.commonConstInfo.gDv;
                }

                s1Offset = nextS1oIdx * CUBE_BASEM * constInfo.commonConstInfo.bN2GD;
                bOffset = nextBoIdx * constInfo.commonConstInfo.n2GD;
                n2Offset = nextGoIdx * constInfo.commonConstInfo.dSize;
                gOffset = nextN2oIdx * constInfo.commonConstInfo.gD;
            } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                if constexpr (IS_D_NO_EQUAL) {
                    bOffsetDv = nextBoIdx * constInfo.commonConstInfo.n2GS1Dv;
                    s1OffsetDv = nextS1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GDv;
                    n2OffsetDv = nextN2oIdx * constInfo.commonConstInfo.gDv;
                    gOffsetDv = nextGoIdx * constInfo.commonConstInfo.dSizeV;
                }
 
                bOffset = nextBoIdx * constInfo.commonConstInfo.n2GS1D;
                s1Offset = nextS1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GD;
                n2Offset = nextN2oIdx * constInfo.commonConstInfo.gD;
                gOffset = nextGoIdx * constInfo.commonConstInfo.dSize;
            }
            flag.nextMorN = (nextS1oIdx == constInfo.s1Outer - 1) ? constInfo.s1CvTail : CUBE_BASEM;
        }
        nextQueryOffset = bOffset + n2Offset + gOffset + s1Offset;
        if constexpr (IS_ROPE) {
            nextQueryOffset = (nextQueryOffset / 3) << 1;
        }
        if constexpr (IS_D_NO_EQUAL) {
            nextDxOffset = bOffsetDv + n2OffsetDv + gOffsetDv + s1OffsetDv;
        } else {
            nextDxOffset = nextQueryOffset;
        }
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::IterateMm1Mm2(
    FagRunInfo &runInfo, int64_t nextIndex, int64_t computeBlockIdx, int64_t remainLoopNum)
{
    ///////////////////////////////////////////////////////////////
    // MM1: dx@v
    ///////////////////////////////////////////////////////////////
    int64_t queryGmOffset = GetQueryOffset(runInfo);
    int64_t keyGmOffset = GetKeyOffset(runInfo);
    int64_t dxGmOffset = queryGmOffset;
    int64_t valueGmOffset = keyGmOffset;

    if constexpr (IS_D_NO_EQUAL) {
        dxGmOffset = GetDxOffset(runInfo);
        valueGmOffset = GetValueOffset(runInfo);
    }
    uint64_t nextDxOffset = 0;
    uint64_t nextQueryOffset = 0;
 
    FagTscmFlagData flag{0};
    FagTscmRopeFlagData flagRope{0};
    // loop reuse
    if (runInfo.isS2IdxNoChange) {
        flag.kvNeedCopy = 0;
    } else {
        flag.kvNeedCopy = 1;
        kvPingPong = 1 - kvPingPong;
    }
    runInfo.kvPingPong = kvPingPong;
    flag.leftMatrixEncodingTableIdx = GET_Q_DX_ENCODING_TABLE_IDX(0, runInfo.qDxPingPongIdx); // 0 means the 0th mm
    flag.rightMatrixEncodingTableIdx = GET_K_V_ENCODING_TABLE_IDX(0, 0); // 0 means 0th mm, 0 means v no ping pong
    SetTscmPreloadFlag(runInfo, flag, nextIndex, nextDxOffset, nextQueryOffset);
    flag.offsetSign = nextDxOffset > dxGmOffset ? 1 : 0;
    flag.nextAddr = flag.offsetSign ? nextDxOffset - dxGmOffset : dxGmOffset - nextDxOffset;
    if constexpr (IS_ROPE) {
        flagRope.flagCommon = flag;
        flagRope.mmIdx = 0;
        mm1.SetSelfDefineData(flagRope);
    } else {
        mm1.SetSelfDefineData(flag);
    }
 
    if constexpr (IS_TND) {
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        GetSeqQlenKvlenByBidx(runInfo.commonRunInfo.boIdx, actualS1Len, actualS2Len);
        mm1.SetOrgShape(actualS1Len, constInfo.commonConstInfo.s2Size, constInfo.commonConstInfo.mm1Ka,
                        constInfo.commonConstInfo.mm1Kb, CUBE_BASEN);
    } else {
        mm1.SetOrgShape(constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.s2Size,
                        constInfo.commonConstInfo.mm1Ka, constInfo.commonConstInfo.mm1Kb, CUBE_BASEN);
    }
    
    mm1.SetTail(runInfo.commonRunInfo.s1RealSize, runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.dSizeV);
    mm1.SetTensorA(dxGm[dxGmOffset]);
    mm1.SetTensorB(valueGm[valueGmOffset], true);
    LocalTensor<T2> mm1ResQueInTensor = mm1ResBuf[runInfo.commonRunInfo.taskIdMod2].Get<T2>();
    mm1.template IterateAll<false>(mm1ResQueInTensor, 0, false, true);

 
    ///////////////////////////////////////////////////////////////
    // MM2: q@k
    ///////////////////////////////////////////////////////////////
    flag.leftMatrixEncodingTableIdx = GET_Q_DX_ENCODING_TABLE_IDX(1, runInfo.qDxPingPongIdx); // 1 means the 1th mm
    flag.rightMatrixEncodingTableIdx = GET_K_V_ENCODING_TABLE_IDX(1, runInfo.kvPingPong);     // 1 means the 1th mm
    flag.offsetSign = nextQueryOffset > queryGmOffset ? 1 : 0;
    flag.nextAddr = flag.offsetSign ? nextQueryOffset - queryGmOffset : queryGmOffset - nextQueryOffset;
     if constexpr(IS_ROPE) {
        flagRope.flagCommon = flag;
        int64_t queryRopeGmOffset = GetQueryRopeOffset(runInfo);
        int64_t keyRopeGmOffset = GetKeyRopeOffset(runInfo);
        flagRope.aRopeAddr = (uint64_t)queryRopeGm[queryRopeGmOffset].GetPhyAddr();
        flagRope.bRopeAddr = (uint64_t)keyRopeGm[keyRopeGmOffset].GetPhyAddr();
        flagRope.mmIdx = 1;
        mm1.SetSelfDefineData(flagRope);
    } else {
        mm1.SetSelfDefineData(flag);
    }
    if constexpr (IS_D_NO_EQUAL) {
        if constexpr (IS_TND) {
            int64_t actualS1Len = 0;
            int64_t actualS2Len = 0;
            GetSeqQlenKvlenByBidx(runInfo.commonRunInfo.boIdx, actualS1Len, actualS2Len);
            mm1.SetOrgShape(actualS1Len, constInfo.commonConstInfo.s2Size, constInfo.mm2Ka,
                            constInfo.mm2Kb, CUBE_BASEN);
        } else {
            mm1.SetOrgShape(constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.s2Size,
                            constInfo.mm2Ka, constInfo.mm2Kb, CUBE_BASEN);
        }
        
        mm1.SetTail(runInfo.commonRunInfo.s1RealSize, runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.dSize);
    }
    mm1.SetTensorA(queryGm[queryGmOffset]);
    mm1.SetTensorB(keyGm[keyGmOffset], true);
    LocalTensor<T2> mm2ResQueInTensor = mm2ResBuf[runInfo.commonRunInfo.taskIdMod2].Get<T2>();
    mm1.template IterateAll<false>(mm2ResQueInTensor, 0, false, true);
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        queryGmOffset = GetQueryOffset(runInfo, false);
        keyGmOffset = GetKeyOffset(runInfo, false);
        WriteOffsetToGM(queryGmOffset, keyGmOffset, valueGmOffset, computeBlockIdx, remainLoopNum);
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::ProcessSoftmaxGrad(FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF1: Cast + SoftmaxGradFront
    ///////////////////////////////////////////////////////////////
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    LocalTensor<T2> softmaxGradResTensor = softmaxGradResBuf.Get<T2>();
    if constexpr (HEAD_DIM_ALIGN <= VECTOR_BASEN) {
        CopyInSoftmaxGrad<OUTDTYPE, T2, VECTOR_BASEM, HEAD_DIM_ALIGN, IS_D_NO_EQUAL>(
            constInfo, runInfo, 0, runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.halfS1RealSize,
            attenMaskOrYInQue, pseOrDyInQue, dxGm, yGm);
        CalculateCastSoftmaxGrad<OUTDTYPE, T2, VECTOR_BASEM, HEAD_DIM_ALIGN>(
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
            CopyInSoftmaxGrad<OUTDTYPE, T2, VECTOR_BASEM, HEAD_DIM_ALIGN, IS_D_NO_EQUAL>(constInfo, runInfo, loopIdx, curLoopSize, loopSize,
                                                                    attenMaskOrYInQue, pseOrDyInQue, dxGm, yGm);
            CalculateCastSoftmaxGrad<OUTDTYPE, T2, VECTOR_BASEM, HEAD_DIM_ALIGN>(
                constInfo, curLoopSize, attenMaskOrYInQue, pseOrDyInQue, softmaxGradResTensor[loopSize * loopIdx]);
        }
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::ProcessReCompute(FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF2: pse + attenMask + muls + simpleSoftmax copyIn+calculate
    ///////////////////////////////////////////////////////////////
    LocalTensor<T2> mm2ResQueInTensor = mm2ResBuf[runInfo.commonRunInfo.taskIdMod2].Get<T2>();
    LocalTensor<T2> mm1ResQueInTensor = mm1ResBuf[runInfo.commonRunInfo.taskIdMod2].Get<T2>();
    if constexpr (IS_FP8_INPUT) {
        Muls(mm1ResQueInTensor, mm1ResQueInTensor, runInfo.quantScaleInfo.deqScaleDyValue * runInfo.quantScaleInfo.deqScaleVValue, VECTOR_BASEM * VECTOR_BASEN);
        Muls(mm2ResQueInTensor, mm2ResQueInTensor, runInfo.quantScaleInfo.deqScaleQValue * runInfo.quantScaleInfo.deqScaleKValue, VECTOR_BASEM * VECTOR_BASEN);
    }
    CopyInAttenMask<IS_ATTEN_MASK, VECTOR_BASEM, VECTOR_BASEN>(constInfo, runInfo, attenMaskInfo, attenMaskOrYInQue,
                                                               pseOrDyInQue, attenMaskU8Gm);
    CopyInPse<OUTDTYPE, T2, IS_PSE>(constInfo, runInfo, pseInfo, pseOrDyInQue, pseGm);
    CalculatePseMulsSelSimpleSoftMax<OUTDTYPE, T2, false, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD(DETER_SPARSE_TYPE), VECTOR_BASEM, VECTOR_BASEN>(
        constInfo, runInfo, pseInfo, attenMaskInfo, maxSumQue[runInfo.commonRunInfo.taskIdMod2], attenMaskOrYInQue,
        pseOrDyInQue, mm2ResQueInTensor, mm2ResQueInTensor, pseSlope);
    if (dropInfo.dropMaskOuter) {
        if (dropInfo.boolMode) {
            CopyInDropOuter<IS_DROP>(dropMaskBuf, attenMaskOrYInQue, dropMaskWorkspaceGm, runInfo.commonRunInfo, constInfo.commonConstInfo, 
                dropInfo);
        } else {
            CopyInDropOuter<IS_DROP>(dropMaskBuf, attenMaskOrYInQue, dropMaskGm, runInfo.commonRunInfo, constInfo.commonConstInfo, 
                dropInfo);
        }
    } else {
        GenDropMask<IS_DROP>(dropMaskBuf, dropmaskIndexVecBuf, runInfo.commonRunInfo, constInfo.commonConstInfo, dropInfo);
    }
    CalculateDropout<T2, IS_DROP, VECTOR_BASEN>(constInfo, runInfo, dropInfo, mm1ResQueInTensor, mm1ResQueInTensor,
                                                dropMaskBuf);
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::IterateMm3Mm4Mm5(FagRunInfo &runInfo,
                                                                                                 int64_t nextIndex)
{
    ///////////////////////////////////////////////////////////////
    // VF3: sub + mul
    // VF4: dq dk cast + nd2nz
    ///////////////////////////////////////////////////////////////
    LocalTensor<T2> mm1ResQueInTensor = mm1ResBuf[runInfo.commonRunInfo.taskIdMod2].Get<T2>();
    LocalTensor<T2> mm2ResQueInTensor = mm2ResBuf[runInfo.commonRunInfo.taskIdMod2].Get<T2>();
    LocalTensor<T2> softmaxGradResTensor = softmaxGradResBuf.Get<T2>();
    LocalTensor<T1> vecOutBuffer = dSOutQue.AllocTensor<T1>();
    if (runInfo.commonRunInfo.s2RealSize > 64) {
        BroadcastSubMul<T2, 128, 0>(mm1ResQueInTensor, mm1ResQueInTensor, softmaxGradResTensor, mm2ResQueInTensor,
                                 runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
    } else {
        if (constInfo.deterConstInfo.noNeedDeter) {
            BroadcastSubMul<T2, 64, 0>(mm1ResQueInTensor, mm1ResQueInTensor, softmaxGradResTensor, mm2ResQueInTensor,
                                        runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
        } else { // 64~128的脏数据需要清零，避免后面的mm有脏数据参与计算
            BroadcastSubMul<T2, 64, IS_DETER_OLD(DETER_SPARSE_TYPE)>(mm1ResQueInTensor, mm1ResQueInTensor, softmaxGradResTensor, mm2ResQueInTensor,
                                        runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
        }
    }

    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) { // 确定性计算尾块脏数据补零
        if (!constInfo.deterConstInfo.noNeedDeter) { 
            if (runInfo.commonRunInfo.halfS1RealSize != VECTOR_BASEM) {
                Duplicate<T2>(mm1ResQueInTensor[runInfo.commonRunInfo.halfS1RealSize * VECTOR_BASEN], 0,
                            (VECTOR_BASEM - runInfo.commonRunInfo.halfS1RealSize) * VECTOR_BASEN);
            }
        }
    }

    // input type fp32, no post, mov muls here
    if constexpr (IS_FP32_INPUT) {
        Muls(mm1ResQueInTensor, mm1ResQueInTensor, constInfo.scaleValue, VECTOR_BASEM * VECTOR_BASEN);
    }

    LocalTensor<uint8_t> selrIndexesTensor;
    float qScaleDs = 1.0;
    if constexpr (IS_FP8_INPUT) {
        LocalTensor<float> dsAmaxTensor = dsAmaxOutQue.AllocTensor<float>();
        if (runInfo.commonRunInfo.s2RealSize > 64) {
            DsAbsReduceMax<T2, 128>(dsAmaxTensor, mm1ResQueInTensor, runInfo.commonRunInfo.halfS1RealSize, 
                                    runInfo.commonRunInfo.s2RealSize);
        } else {
            DsAbsReduceMax<T2, 64>(dsAmaxTensor, mm1ResQueInTensor, runInfo.commonRunInfo.halfS1RealSize, 
                                    runInfo.commonRunInfo.s2RealSize);
        }
        event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDVToS);
        WaitFlag<HardEvent::V_S>(eventIDVToS);
        float curDsMax = dsAmaxTensor.GetValue(0);

		dsAmaxOutQue.EnQue(dsAmaxTensor);
		dsAmaxOutQue.DeQue<float>();
		DataCopyPad(dsAmaxWorkSpaceGm[vBlockIdx * 128], dsAmaxTensor, {1, 4, 0, 0});

		CrossCoreSetFlag<1, PIPE_MTE3>(10);
		CrossCoreWaitFlag<1, PIPE_MTE3>(10);
		event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);

		int64_t anotherVBlockIdx = vSubBlockIdx == 0 ? (vBlockIdx + 1) : (vBlockIdx - 1);
		DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(dsAmaxWorkSpaceGm[anotherVBlockIdx * 128]);
		float dsAmax = Max<float>(curDsMax, dsAmaxWorkSpaceGm.GetValue(anotherVBlockIdx * 128));

        if (dsAmax > 1e-6f) {
            qScaleDs = FP8_MAX / dsAmax;
            Muls(mm1ResQueInTensor, mm1ResQueInTensor, qScaleDs, VECTOR_BASEM * VECTOR_BASEN);
        }
		dsAmaxOutQue.FreeTensor(dsAmaxTensor);
		selrIndexesTensor = vselrIndexesBuf.Get<uint8_t>();
    }

    CastTransdataDeconflict<T1, T2, VECTOR_BASEN>(vecOutBuffer, mm1ResQueInTensor, selrIndexesTensor, VECTOR_BASEM);
    dSOutQue.EnQue(vecOutBuffer);
    dSOutQue.DeQue<T1>();

    int64_t queryGmOffset = GetQueryOffset(runInfo, false);
    int64_t keyGmOffset = GetKeyOffset(runInfo, false);
    int64_t dxGmOffset = queryGmOffset;
    int64_t valueGmOffset = keyGmOffset;
    if constexpr (IS_D_NO_EQUAL) {
        dxGmOffset = GetDxOffset(runInfo);
        valueGmOffset = GetValueOffset(runInfo);
    }
    int64_t nextS2oIdx = (nextIndex / constInfo.s1Outer) % constInfo.s2Outer;
    int64_t nextN2oIdx = (nextIndex % constInfo.n2GS1oS2o) / constInfo.gS1oS2o;
    int64_t nextBoIdx = nextIndex / constInfo.n2GS1oS2o;
    bool isNextS2IdxNoChange = (nextS2oIdx == runInfo.s2oIdx && nextN2oIdx == runInfo.commonRunInfo.n2oIdx &&
                                nextBoIdx == runInfo.commonRunInfo.boIdx);

    ///////////////////////////////////////////////////////////////
    // Matmal3 dq
    // left [B, N2, G, S1, s2] right [B, N2, 1, S2, D] output [B, N2, G, S1, D]
    ///////////////////////////////////////////////////////////////
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        if (likely(constInfo.deterConstInfo.noNeedDeter)) {
            IterateMm3(runInfo, vecOutBuffer, queryGmOffset, keyGmOffset);
        } else {
            if (dqIsNeedDeter[deterPpFlag]) {
                IterateMm3Deter(runInfo, vecOutBuffer, queryGmOffset, keyGmOffset);
            } else {
                IterateMm3(runInfo, vecOutBuffer, queryGmOffset, keyGmOffset);
            }
        }
    } else {
        IterateMm3(runInfo, vecOutBuffer, queryGmOffset, keyGmOffset, qScaleDs);
    }

    ///////////////////////////////////////////////////////////////
    // Matmal4 dk
    // left [B, N2, G, S1, S2] right [B, N2, 1, S1, D] output [B, N2, G, S2, D]
    ///////////////////////////////////////////////////////////////
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        if (likely(constInfo.deterConstInfo.noNeedDeter)) {
            IterateMm4(runInfo, vecOutBuffer, queryGmOffset, keyGmOffset, nextIndex, isNextS2IdxNoChange);
        } else {
            if (dkDvIsNeedDeter[deterPpFlag]) {
                IterateMm4Deter(runInfo, vecOutBuffer, queryGmOffset, keyGmOffset);
            } else {
                IterateMm4(runInfo, vecOutBuffer, queryGmOffset, keyGmOffset, nextIndex, isNextS2IdxNoChange);
            }
        }
    } else {
        IterateMm4(runInfo, vecOutBuffer, queryGmOffset, keyGmOffset, nextIndex, isNextS2IdxNoChange, qScaleDs);
    }
    dSOutQue.FreeTensor(vecOutBuffer);
    ///////////////////////////////////////////////////////////////
    // VF5: cast + nd2nz
    ///////////////////////////////////////////////////////////////
    LocalTensor<T1> vecOutBuffer1 = pOutQue.AllocTensor<T1>();
    CalculateDropout<T2, IS_DROP, VECTOR_BASEN>(constInfo, runInfo, dropInfo, mm2ResQueInTensor, mm2ResQueInTensor,
                                                dropMaskBuf);
    CastTransdataDeconflict<T1, T2, VECTOR_BASEN>(vecOutBuffer1, mm2ResQueInTensor, selrIndexesTensor, VECTOR_BASEM);
    pOutQue.EnQue(vecOutBuffer1);
    pOutQue.DeQue<T1>();

    ///////////////////////////////////////////////////////////////
    // Matmal5 dv
    // left [B, N2, G, S1, S2] right [B, N2, G, S1, D] output [B, N2, 1, S2, D]
    ///////////////////////////////////////////////////////////////
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        if (likely(constInfo.deterConstInfo.noNeedDeter)) {
            IterateMm5(runInfo, vecOutBuffer1, dxGmOffset, valueGmOffset, nextIndex, isNextS2IdxNoChange);
        } else {
            if (dkDvIsNeedDeter[deterPpFlag]) {
                IterateMm5Deter(runInfo, vecOutBuffer1, dxGmOffset, valueGmOffset);
            } else {
                IterateMm5(runInfo, vecOutBuffer1, dxGmOffset, valueGmOffset, nextIndex, isNextS2IdxNoChange);
            }
        }
    } else {
        IterateMm5(runInfo, vecOutBuffer1, dxGmOffset, valueGmOffset, nextIndex, isNextS2IdxNoChange);
    }
    pOutQue.FreeTensor(vecOutBuffer1);
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::IterateMm3Deter(
    FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer, int64_t dxOrQueryGmOffset, int64_t keyOrValueGmOffset)
{
    uint32_t deterGmOffset = 0;
    deterGmOffset = cBlockIdx * BASE_DQ_SIZE +
                    deterPpFlag * (BASE_DQ_SIZE + 2 * BASE_DKV_SIZE) * constInfo.deterConstInfo.usedCubeCoreNum;
    FagTscmFlagData flag{0};
    flag.rightMatrixEncodingTableIdx = GET_K_V_ENCODING_TABLE_IDX(2, runInfo.kvPingPong); // 2 means the 2th mm
    mm2.SetSelfDefineData(flag);

    if constexpr (IS_TND) {
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        GetSeqQlenKvlenByBidx(runInfo.commonRunInfo.boIdx, actualS1Len, actualS2Len);
        mm2.SetOrgShape(actualS1Len, constInfo.commonConstInfo.n2D, actualS2Len, actualS2Len, HEAD_DIM_ALIGN);
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            mm2.SetOrgShape(constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.dSize,
                            constInfo.commonConstInfo.s2Size, constInfo.commonConstInfo.s2Size, HEAD_DIM_ALIGN);
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            mm2.SetOrgShape(constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.bN2D,
                            constInfo.commonConstInfo.s2Size, constInfo.commonConstInfo.s2Size, HEAD_DIM_ALIGN);
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            mm2.SetOrgShape(constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.n2D,
                            constInfo.commonConstInfo.s2Size, constInfo.commonConstInfo.s2Size, HEAD_DIM_ALIGN);
        }
    }
    LocalTensor<T1> dsScmTensor = dsScm.AllocTensor<T1>();
    CopyUB2L1Deter(runInfo, dsScmTensor, vecOutBuffer);
    dsScm.EnQue(dsScmTensor);
    dsScm.DeQue<T1>();
    mm2.SetTail(CUBE_BASEM, constInfo.commonConstInfo.dSize, runInfo.commonRunInfo.s2RealSize);
    mm2.SetTensorA(dsScmTensor);
    mm2.SetTensorB(keyGm[keyOrValueGmOffset]);
    if (dqIsNeedDeter[deterPpFlag] && !dkDvIsNeedDeter[deterPpFlag]) {
        mm2.template IterateAll<false>(deterGm[deterGmOffset], false, false, true);
    } else {
        mm2.template IterateAll<false>(deterGm[deterGmOffset], false, false, false);
    }
    mm2.End();
    dsScm.FreeTensor(dsScmTensor);
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::IterateMm3(
    FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer, int64_t dxOrQueryGmOffset, int64_t keyOrValueGmOffset, float qScaleDs)
{
    FagTscmFlagData flag{0};
    flag.rightMatrixEncodingTableIdx = GET_K_V_ENCODING_TABLE_IDX(2, runInfo.kvPingPong); // 2 means the 2th mm
    mm2.SetSelfDefineData(flag);

    if constexpr (IS_TND) {
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        GetSeqQlenKvlenByBidx(runInfo.commonRunInfo.boIdx, actualS1Len, actualS2Len);
        mm2.SetOrgShape(actualS1Len, constInfo.commonConstInfo.n2D, actualS2Len, actualS2Len,
                        constInfo.commonConstInfo.n2GD);
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            mm2.SetOrgShape(constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.dSize,
                            constInfo.commonConstInfo.s2Size);
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            mm2.SetOrgShape(constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.bN2D,
                            constInfo.commonConstInfo.s2Size, constInfo.commonConstInfo.s2Size,
                            constInfo.commonConstInfo.bN2GD);
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            mm2.SetOrgShape(constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.n2D,
                            constInfo.commonConstInfo.s2Size, constInfo.commonConstInfo.s2Size,
                            constInfo.commonConstInfo.n2GD);
        }
    }
    LocalTensor<T1> dsScmTensor = dsScm.AllocTensor<T1>();
    CopyUB2L1<true>(runInfo, dsScmTensor, vecOutBuffer);
    dsScm.EnQue(dsScmTensor);
    dsScm.DeQue<T1>();
    mm2.SetTail(runInfo.commonRunInfo.s1RealSize, constInfo.commonConstInfo.dSize,
				runInfo.commonRunInfo.s2RealSize);

    if constexpr (IS_FP8_INPUT) {
        float tmp = runInfo.quantScaleInfo.deqScaleKValue / qScaleDs;
        uint64_t ans = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
        mm2.SetQuantScalar(ans);
    }
    mm2.SetTensorA(dsScmTensor);
    mm2.SetTensorB(keyGm[keyOrValueGmOffset]);
    if constexpr (IS_DETER_NEW(DETER_SPARSE_TYPE)) {
        mm2.template IterateAll<false>(dqWorkSpaceGm[dxOrQueryGmOffset], true, false, isMm3NeedWait);
    } else {
        mm2.template IterateAll<false>(dqWorkSpaceGm[dxOrQueryGmOffset], true, false, false);
    }
    mm2.End();
    dsScm.FreeTensor(dsScmTensor);
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::IterateMm4Deter(
    FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer, int64_t dxOrQueryGmOffset, int64_t keyOrValueGmOffset)
{
    uint32_t deterGmOffset =
        cBlockIdx * BASE_DKV_SIZE + BASE_DQ_SIZE * constInfo.deterConstInfo.usedCubeCoreNum +
        deterPpFlag * (BASE_DQ_SIZE + 2 * BASE_DKV_SIZE) * constInfo.deterConstInfo.usedCubeCoreNum;
    FagTscmFlagData flag{0};
    flag.rightMatrixEncodingTableIdx = GET_Q_DX_ENCODING_TABLE_IDX(3, runInfo.qDxPingPongIdx); // 3 means the 3th mm
    mm3.SetSelfDefineData(flag);
    if constexpr (IS_TND) {
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        GetSeqQlenKvlenByBidx(runInfo.commonRunInfo.boIdx, actualS1Len, actualS2Len);
        mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.n2GD, actualS1Len, actualS1Len,
                        HEAD_DIM_ALIGN);
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.dSize,
                            constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.s1Size, HEAD_DIM_ALIGN);
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.bN2GD,
                            constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.s1Size, HEAD_DIM_ALIGN);
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.n2GD,
                            constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.s1Size, HEAD_DIM_ALIGN);
        }
    }
    LocalTensor<T1> dsScmTensordq = dsScm.AllocTensor<T1>();
    if (dqIsNeedDeter[deterPpFlag]) {
        dsScmTensordq.SetAddrWithOffset(dsScmTensordq, CUBE_BASEM * CUBE_BASEN);
        CopyUB2L1(runInfo, dsScmTensordq, vecOutBuffer);
    }
    dsScm.EnQue(dsScmTensordq);
    dsScm.DeQue<T1>();
    mm3.SetTail(CUBE_BASEN, constInfo.commonConstInfo.dSize, runInfo.commonRunInfo.s1RealSize);
    mm3.SetTensorA(dsScmTensordq, true);
    mm3.SetTensorB(queryGm[dxOrQueryGmOffset]); // sameB
    mm3.template IterateAll<false>(deterGm[deterGmOffset], false, false, false);
    mm3.End();
    dsScm.FreeTensor(dsScmTensordq);
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::IterateMm4(
    FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer, int64_t dxOrQueryGmOffset, int64_t keyOrValueGmOffset,
    int64_t nextIndex, bool isNextS2IdxNoChange, float qScaleDs)
{
    FagTscmFlagData flag{0};
    flag.rightMatrixEncodingTableIdx = GET_Q_DX_ENCODING_TABLE_IDX(3, runInfo.qDxPingPongIdx); // 3 means the 3th mm
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
    LocalTensor<T1> dsScmTensordq = dsScm.AllocTensor<T1>();
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        if (dqIsNeedDeter[deterPpFlag]) {
            dsScmTensordq.SetAddrWithOffset(dsScmTensordq, CUBE_BASEM * CUBE_BASEN);
            CopyUB2L1(runInfo, dsScmTensordq, vecOutBuffer);
        }
    }
    if constexpr (IS_FP8_INPUT) {
        dsScmTensordq.SetAddrWithOffset(dsScmTensordq, CUBE_BASEM * CUBE_BASEN);
        CopyUB2L1(runInfo, dsScmTensordq, vecOutBuffer);
    }
    dsScm.EnQue(dsScmTensordq);
    dsScm.DeQue<T1>();
    mm3.SetTail(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.dSize,
            runInfo.commonRunInfo.s1RealSize);

    if constexpr (IS_FP8_INPUT) {
        float tmp = runInfo.quantScaleInfo.deqScaleQValue / qScaleDs;
        uint64_t ans = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
        mm3.SetQuantScalar(ans);
    }

    mm3.SetTensorA(dsScmTensordq, true);
    mm3.SetTensorB(queryGm[dxOrQueryGmOffset]); // sameB
    if constexpr (!IS_L0C_REUSE) {
        mm3.SetSelfDefineData(flag);
        mm3.template IterateAll<false>(dkWorkSpaceGm[keyOrValueGmOffset], true, false, false);
    } else {
        flag.nextMorN = L0C_BUF_NUM - DK_DV_L0C_BUF_NUM; // for dk l0c buffer idx
        mm3.SetSelfDefineData(flag);
        mm3.template Iterate<false>(runInfo.isS2IdxNoChange);
        if (nextIndex != -1) {
            // 当前s2列的最后一个有效基本块
            if (!isNextS2IdxNoChange) {
                mm3.template GetTensorC<false>(dkWorkSpaceGm[keyOrValueGmOffset], true, false);
            }
        } else {
            mm3.template GetTensorC<false>(dkWorkSpaceGm[keyOrValueGmOffset], true, false);
        }
    }
    mm3.End();
    dsScm.FreeTensor(dsScmTensordq);
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::IterateMm5Deter(
    FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer1, int64_t dxOrQueryGmOffset, int64_t keyOrValueGmOffset)
{
    uint32_t deterGmOffset =
        cBlockIdx * BASE_DKV_SIZE + (BASE_DQ_SIZE + BASE_DKV_SIZE) * constInfo.deterConstInfo.usedCubeCoreNum +
        deterPpFlag * (BASE_DQ_SIZE + 2 * BASE_DKV_SIZE) * constInfo.deterConstInfo.usedCubeCoreNum;
    FagTscmFlagData flag{0};
    flag.rightMatrixEncodingTableIdx = GET_Q_DX_ENCODING_TABLE_IDX(4, runInfo.qDxPingPongIdx); // 4 means the 4th mm
    if constexpr (IS_D_NO_EQUAL) {
        if constexpr (IS_TND) {
            int64_t actualS1Len = 0;
            int64_t actualS2Len = 0;
            GetSeqQlenKvlenByBidx(runInfo.commonRunInfo.boIdx, actualS1Len, actualS2Len);
            mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.n2GDv, actualS1Len, actualS1Len,
                            HEAD_DIM_ALIGN);
        } else {
            if (constInfo.commonConstInfo.layoutType == BNGSD) {
                mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.dSizeV,
                                constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.s1Size, HEAD_DIM_ALIGN);
            } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.bN2GDv,
                                constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.s1Size, HEAD_DIM_ALIGN);
            } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.n2GDv,
                                constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.s1Size, HEAD_DIM_ALIGN);
            }
        }
    }
    mm3.SetSelfDefineData(flag);
    LocalTensor<T1> pScmTensor = pScm.AllocTensor<T1>();
    CopyUB2L1(runInfo, pScmTensor, vecOutBuffer1);
    pScm.EnQue(pScmTensor);
    pScm.DeQue<T1>();
    mm3.SetTail(CUBE_BASEN, constInfo.commonConstInfo.dSizeV, runInfo.commonRunInfo.s1RealSize);
    mm3.SetTensorA(pScmTensor, true);
    mm3.SetTensorB(dxGm[dxOrQueryGmOffset]); // sameB
    mm3.template IterateAll<false>(deterGm[deterGmOffset], false, false, true);
    mm3.End();
    pScm.FreeTensor(pScmTensor);
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::IterateMm5(
    FagRunInfo &runInfo, LocalTensor<T1> &vecOutBuffer1, int64_t dxOrQueryGmOffset, int64_t keyOrValueGmOffset,
    int64_t nextIndex, bool isNextS2IdxNoChange)
{
    FagTscmFlagData flag{0};
    flag.rightMatrixEncodingTableIdx = GET_Q_DX_ENCODING_TABLE_IDX(4, runInfo.qDxPingPongIdx); // 4 means the 4th mm
    if constexpr (IS_D_NO_EQUAL) {
        if constexpr (IS_TND) {
            int64_t actualS1Len = 0;
            int64_t actualS2Len = 0;
            GetSeqQlenKvlenByBidx(runInfo.commonRunInfo.boIdx, actualS1Len, actualS2Len);
            mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.n2GDv, actualS1Len, actualS1Len,
                            constInfo.commonConstInfo.n2Dv);
        } else {
            if (constInfo.commonConstInfo.layoutType == BNGSD) {
                mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.dSizeV,
                                constInfo.commonConstInfo.s1Size);
            } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.bN2GDv,
                                constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.s1Size,
                                constInfo.commonConstInfo.bN2Dv);
            } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                mm3.SetOrgShape(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.n2GDv,
                                constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.s1Size,
                                constInfo.commonConstInfo.n2Dv);
            }
        }
    }
    LocalTensor<T1> pScmTensor = pScm.AllocTensor<T1>();
    CopyUB2L1(runInfo, pScmTensor, vecOutBuffer1);
    pScm.EnQue(pScmTensor);
    pScm.DeQue<T1>();
    if constexpr (IS_D_NO_EQUAL) {
		mm3.SetTail(runInfo.commonRunInfo.s2RealSize, constInfo.commonConstInfo.dSizeV,
					runInfo.commonRunInfo.s1RealSize);
    }
    
    if constexpr (IS_FP8_INPUT) {
        uint64_t ans = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&runInfo.quantScaleInfo.deqScaleDyValue));
        mm3.SetQuantScalar(ans);
    }

    mm3.SetTensorA(pScmTensor, true);
    mm3.SetTensorB(dxGm[dxOrQueryGmOffset]); // sameB or norm
    if constexpr (!IS_L0C_REUSE) {
        mm3.SetSelfDefineData(flag);
        if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
            if (likely(constInfo.deterConstInfo.noNeedDeter)) {
                mm3.template IterateAll<false>(dvWorkSpaceGm[keyOrValueGmOffset], true, false, true);
            } else {
                if (!dqIsNeedDeter[deterPpFlag] && !dkDvIsNeedDeter[deterPpFlag]) {
                    // 如果dq和dk dv都不需要做确定性计算，那么最后一个参数需要设置为true与waitIterateAll匹配
                    mm3.template IterateAll<false>(dvWorkSpaceGm[keyOrValueGmOffset], true, false, true);
                } else {
                    if (isLastLoop) {
                        mm3.template IterateAll<true>(dvWorkSpaceGm[keyOrValueGmOffset], true, false, false);
                    } else {
                        mm3.template IterateAll<false>(dvWorkSpaceGm[keyOrValueGmOffset], true, false, false);
                    }
                }
            }
        } else {
            if (isLastLoop) {
                mm3.template IterateAll<true>(dvWorkSpaceGm[keyOrValueGmOffset], true, false, false);
            } else {
                mm3.template IterateAll<false>(dvWorkSpaceGm[keyOrValueGmOffset], true, false, false);
            }
        }
    } else {
        flag.nextMorN = L0C_BUF_NUM - 1; // for dv l0c buffer idx
        mm3.SetSelfDefineData(flag);
        if (nextIndex != -1) {
            mm3.template Iterate<false>(runInfo.isS2IdxNoChange);
            // 当前s2列的最后一个有效基本块
            if (!isNextS2IdxNoChange) {
                mm3.template GetTensorC<false>(dvWorkSpaceGm[keyOrValueGmOffset], true, false);
            }
        } else {
            mm3.template Iterate<true>(runInfo.isS2IdxNoChange);
            mm3.template GetTensorC<true>(dvWorkSpaceGm[keyOrValueGmOffset], true, false);
        }
    }
    mm3.End();
    pScm.FreeTensor(pScmTensor);
}

FAG_FUNCTION_TEMPLATE
template <const bool IS_DQ>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::CopyUB2L1(
    FagRunInfo &runInfo, LocalTensor<T1> &dstTensor, LocalTensor<T1> &srcTensor)
{
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    uint32_t scmOffset = vSubBlockIdx == 0 ? 0 : runInfo.commonRunInfo.firstHalfS1RealSize * FRACTAL_NZ_C0_SIZE;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = VECTOR_BASEN / FRACTAL_NZ_C0_SIZE;
    dataCopyParams.blockLen = (uint16_t)(runInfo.commonRunInfo.halfS1RealSize * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM);
    dataCopyParams.srcStride =
        (uint16_t)((VECTOR_BASEM + 1 - runInfo.commonRunInfo.halfS1RealSize) * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM);
    if constexpr (IS_FP8_INPUT) {
        if constexpr (IS_DQ) {
            uint32_t s1RealSizeAlignTo16 = AlignTo16(runInfo.commonRunInfo.s1RealSize);
            dataCopyParams.dstStride =
                (s1RealSizeAlignTo16 - runInfo.commonRunInfo.halfS1RealSize) * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM;
        } else {
            uint32_t s1RealSizeAlignTo32 = AlignTo32(runInfo.commonRunInfo.s1RealSize);
            dataCopyParams.dstStride =
                (s1RealSizeAlignTo32 - runInfo.commonRunInfo.halfS1RealSize) * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM;
        }
    } else {
        uint32_t s1RealSizeAlignTo16 = AlignTo16(runInfo.commonRunInfo.s1RealSize);
        dataCopyParams.dstStride =
            (s1RealSizeAlignTo16 - runInfo.commonRunInfo.halfS1RealSize) * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM;
    }
    DataCopy(dstTensor[scmOffset], srcTensor, dataCopyParams);
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::CopyUB2L1Deter(
    FagRunInfo &runInfo, LocalTensor<T1> &dstTensor, LocalTensor<T1> &srcTensor)
{
    uint32_t scmOffset = (vSubBlockIdx == 0 ? 0 : runInfo.commonRunInfo.firstHalfS1RealSize * FRACTAL_NZ_C0_SIZE);
    DataCopyParams dataCopyParams;
    if (runInfo.commonRunInfo.halfS1RealSize != 0) {
        dataCopyParams.blockCount = VECTOR_BASEN / FRACTAL_NZ_C0_SIZE;
        dataCopyParams.blockLen = (uint16_t)(runInfo.commonRunInfo.halfS1RealSize * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM);
        dataCopyParams.srcStride =
            (uint16_t)((VECTOR_BASEM - runInfo.commonRunInfo.halfS1RealSize + 1) * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM);
        dataCopyParams.dstStride =
            (uint16_t)(CUBE_BASEM - runInfo.commonRunInfo.halfS1RealSize) * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM;
        DataCopy(dstTensor[scmOffset], srcTensor, dataCopyParams);
    }
    if (runInfo.commonRunInfo.halfS1RealSize != VECTOR_BASEM) {
        // copy 补零的数据
        scmOffset = (vSubBlockIdx == 0 ? runInfo.commonRunInfo.s1RealSize * FRACTAL_NZ_C0_SIZE
                                        : (VECTOR_BASEM + runInfo.commonRunInfo.halfS1RealSize) * FRACTAL_NZ_C0_SIZE);
        dataCopyParams.blockCount = VECTOR_BASEN / FRACTAL_NZ_C0_SIZE;
        dataCopyParams.blockLen =
            (uint16_t)((VECTOR_BASEM - runInfo.commonRunInfo.halfS1RealSize) * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM);
        dataCopyParams.srcStride =
            (uint16_t)((runInfo.commonRunInfo.halfS1RealSize + 1) * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM);
        dataCopyParams.dstStride =
            (uint16_t)(VECTOR_BASEM + runInfo.commonRunInfo.halfS1RealSize) * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM;
        DataCopy(dstTensor[scmOffset], srcTensor[runInfo.commonRunInfo.halfS1RealSize * FRACTAL_NZ_C0_SIZE], dataCopyParams);
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline bool
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::CheckIsValidBlock(FagRunInfo &runInfo,
                                                                                                  int64_t baseIdx,
                                                                                                  int64_t s1oDimIdx,
                                                                                                  int64_t s2oDimIdx)
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
        s2CvBegin = s2IdxLeft;
        s2CvEnd = s2CvBegin + CUBE_BASEN;         // 非尾块s2按照+CUBE_BASEN处理
        if (s2oDimIdx == constInfo.s2Outer - 1) { // 默认s2 cv tail相等
            s2CvEnd = s2CvBegin + constInfo.s2Tail;
        }
    }
    return isValid;
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline bool
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::IsValid(FagRunInfo &runInfo,
                                                                                        int64_t index)
{
    if constexpr (IS_TND) {
        int64_t resbaseIdx = index - runInfo.lastBatchTotalBaseIdx;
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        for (int64_t bIdx = runInfo.lastBatchIdx; bIdx < constInfo.bSize; bIdx++) {
            GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
            int64_t s1OuterTmp = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
            int64_t s2OuterTmp = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
            int64_t totalBaseIdx = constInfo.n2Size * constInfo.commonConstInfo.gSize * s1OuterTmp * s2OuterTmp;
            if (resbaseIdx < totalBaseIdx) {
                int64_t gDimTail = resbaseIdx % (s1OuterTmp * s2OuterTmp);
                int64_t s2oDimIdx = gDimTail / s1OuterTmp;
                int64_t s1oDimIdx = gDimTail % s1OuterTmp;
                int64_t s2IdxLeft = s2oDimIdx * CUBE_BASEN;
                int64_t s2IdxRight = Min((s2oDimIdx + 1) * CUBE_BASEN, actualS2Len);
                if constexpr (SPLIT_AXIS == 5) {
                    if (curS2oIdx == -1 || curS2oIdx != s2oDimIdx) {
                        curS2oIdx = s2oDimIdx;
                        curS2InvalidTotalNum = 0;
                    }
                }
                if constexpr (IS_ATTEN_MASK) {
                    if (constInfo.sparseMode == PREFIX_COMPRESS) {
                        int64_t s2IdxLeft = s2oDimIdx * CUBE_BASEN;
                        int64_t s2IdxRight = (s2oDimIdx + 1) * CUBE_BASEN;
                        int64_t s2IgnoredEndLen = actualS1Len - static_cast<int64_t>(CUBE_BASEM * (s1oDimIdx + 1));
                        int64_t s2EndLen = 0;
                        if (actualS2Len > s2IgnoredEndLen) {
                            s2EndLen = actualS2Len - s2IgnoredEndLen;
                        }
                        if (constInfo.sparseMode == PREFIX || constInfo.sparseMode == PREFIX_COMPRESS) {
                            s2EndLen = Min(Max(s2EndLen, ((__gm__ int64_t *)prefixNAddr)[bIdx]), actualS2Len);
                        }
                        bool isValid = s2IdxLeft < s2EndLen;
                        if (isValid) {
                            s2CvBegin = s2IdxLeft;
                            s2CvEnd = s2CvBegin + CUBE_BASEN; // 非尾块s2按照+CUBE_BASEN处理
                            if (s2oDimIdx == s2OuterTmp - 1) {
                                s2CvEnd = s2CvBegin + actualS2Len - s2oDimIdx * CUBE_BASEN;
                            }
                        }
                        if constexpr (SPLIT_AXIS == 5) {
                            if (!isValid) {
                                curS2InvalidTotalNum += 1;
                            }
                            if (curS2InvalidTotalNum * CUBE_BASEM >= actualS1Len) {
                                return true;
                            }
                        }
                        return isValid;
                    }

                    UpdateToken(runInfo, bIdx);
                    int64_t s2SparseLeft = Max(CUBE_BASEM * s1oDimIdx - actualCalcS1Token, 0);
                    s2SparseLeft = s2SparseLeft >> 6 << 6;
                    int64_t s2SparseRight = AlignTo64(
                        Min(CUBE_BASEM * (s1oDimIdx + 1), constInfo.commonConstInfo.s1Size) + actualCalcS2Token);
                    s2SparseRight = Min(s2SparseRight, actualS2Len);
                    bool isValid = s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft;
                    s2CvBegin = s2IdxLeft;
                    s2CvEnd = s2CvBegin + CUBE_BASEN;  // 非尾块s2按照+CUBE_BASEN处理
                    if (s2oDimIdx == s2OuterTmp - 1) { // 默认s2 cv tail相等
                        s2CvEnd = s2CvBegin + actualS2Len - s2oDimIdx * CUBE_BASEN;
                    }
                    if constexpr (SPLIT_AXIS == 5) {
                        if (!isValid) {
                            curS2InvalidTotalNum += 1;
                        }
                        if (curS2InvalidTotalNum * CUBE_BASEM >= actualS1Len) {
                            return true;
                        }
                    }
                    return isValid;
                } else {
                    s2CvBegin = s2IdxLeft;
                    s2CvEnd = s2IdxRight;
                    return true;
                }
            } else {
                resbaseIdx -= totalBaseIdx;
            }
        }
        return false;
    } else {
        int64_t gDimTail = index % constInfo.s1oS2o;
        int64_t s2oDimIdx = gDimTail / constInfo.s1Outer;
        int64_t s1oDimIdx = gDimTail % constInfo.s1Outer;
        int64_t s2IdxLeft = s2oDimIdx * CUBE_BASEN;
        int64_t s2IdxRight = Min((s2oDimIdx + 1) * CUBE_BASEN, constInfo.commonConstInfo.s2Size);
        if constexpr (IS_ATTEN_MASK) {
            if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL || constInfo.sparseMode == PREFIX ||
                constInfo.sparseMode == PREFIX_COMPRESS) {
                return CheckIsValidBlock(runInfo, index, s1oDimIdx, s2oDimIdx);
            } else {
                int64_t s2SparseLeft = Max(CUBE_BASEM * s1oDimIdx - constInfo.s1Token, 0);
                s2SparseLeft = s2SparseLeft >> 6 << 6;
                int64_t s2SparseRight =
                    AlignTo64(Min(CUBE_BASEM * (s1oDimIdx + 1), constInfo.commonConstInfo.s1Size) + constInfo.s2Token);
                s2SparseRight = Min(s2SparseRight, constInfo.commonConstInfo.s2Size);
                bool isValid = s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft;
                s2CvBegin = s2IdxLeft;
                s2CvEnd = s2CvBegin + CUBE_BASEN;         // 非尾块s2按照+CUBE_BASEN处理
                if (s2oDimIdx == constInfo.s2Outer - 1) { // 默认s2 cv tail相等
                    s2CvEnd = s2CvBegin + constInfo.s2Tail;
                }
                return isValid;
            }
        } else {
            s2CvBegin = s2IdxLeft;
            s2CvEnd = s2IdxRight;
            return true;
        }
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::UpdateToken(FagRunInfo &runInfo, 
                                                                                            int64_t bIdx)
{
    // sparse_mode == band 或者 RIGHT_DOWN_CASUAL时，token以右下角为基本，需要校正
    if constexpr (IS_ATTEN_MASK) {
        int64_t actualS1Len;
        int64_t actualS2Len;
        if (constInfo.sparseMode == RIGHT_DOWN_CASUAL_BAND && bIdx != attenMaskInfo.bandIndex) {
            GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
            actualCalcS1Token = static_cast<int64_t>(INT32_MAX) + actualS1Len - actualS2Len;
            actualCalcS2Token = static_cast<int64_t>(0) - actualS1Len + actualS2Len;
        } else if (constInfo.sparseMode == BAND_LEFT_UP_CASUAL && bIdx != attenMaskInfo.bandIndex) {
            actualCalcS1Token = INT32_MAX;
            actualCalcS2Token = 0;
        } else if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL || constInfo.sparseMode == BAND || (constInfo.sparseMode == RIGHT_DOWN_CASUAL_BAND 
                    && bIdx == attenMaskInfo.bandIndex) || (constInfo.sparseMode == BAND_LEFT_UP_CASUAL && bIdx == attenMaskInfo.bandIndex)) {
            GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
            actualCalcS1Token = constInfo.s1Token + actualS1Len - actualS2Len;
            actualCalcS2Token = constInfo.s2Token - actualS1Len + actualS2Len;
        }
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::DeterComputeDq(
    bool isLastSort, int64_t computeBlockIdx, int64_t remainLoopNum)
{
    // 卡指定流水的全核同步
    SetFlag<HardEvent::MTE3_MTE2>(constInfo.deterConstInfo.eventIDMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(constInfo.deterConstInfo.eventIDMte3ToMte2);
    SyncAll<true, syncAllConfigMte2ToMte2>();

    LocalTensor<int64_t> deterOffsetTensor = deterOffsetBuf.Get<int64_t>();
    if (!isFirstDeter) {
        WaitFlag<HardEvent::S_MTE2>(constInfo.deterConstInfo.eventIDScalarToMte2);
    }
    DataCopy(deterOffsetTensor, deterOffsetGm[computeBlockIdx * constInfo.deterConstInfo.usedCubeCoreNum * INT64_BLOCK_NUM * 3],
             {1, static_cast<uint16_t>(constInfo.deterConstInfo.usedCubeCoreNum * 3), 0, 0});
    SetFlag<HardEvent::MTE2_S>(constInfo.deterConstInfo.eventIDMte2ToScalar);
    int16_t bufId = 0;
    // 非last基本块处理的都是上一轮mm的结果， 最后一次循环需要补充处理本轮的mm结果
    if (unlikely(isLastSort)) {
        bufId = deterPpFlag;
    } else {
        bufId = 1 - deterPpFlag;
    }
    uint32_t dqSrcOfs = bufId * (BASE_DQ_SIZE + BASE_DKV_SIZE * 2) * constInfo.deterConstInfo.usedCubeCoreNum;
    dqSrcOfs += vBlockIdx * constInfo.deterConstInfo.dqEachVectorSize;

    // dq deter
    DataCopyParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.deterConstInfo.dqEachVectorSize / FLOAT_BLOCK_SIZE;
    dataCopyParams.srcStride = BASE_DQ_SIZE / FLOAT_BLOCK_SIZE - dataCopyParams.blockLen;
    dataCopyParams.dstStride = 0;
    DataCopyExtParams dataCopyPadParams;
    dataCopyPadParams.blockCount = static_cast<uint16_t>(constInfo.deterConstInfo.eachVecCoreS1Offset);
    dataCopyPadParams.blockLen = static_cast<uint32_t>(constInfo.commonConstInfo.dSize * sizeof(T2));
    dataCopyPadParams.srcStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDqkSrcStride);
    dataCopyPadParams.dstStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDqDstStride);

    uint32_t allSize =
        constInfo.deterConstInfo.usedCubeCoreNum * constInfo.deterConstInfo.dqEachVectorSize * sizeof(T2);
    uint8_t loopTimes = static_cast<uint8_t>((allSize + DETER_DQ_UB_SIZE - 1) / DETER_DQ_UB_SIZE);
    uint8_t eachLoopBlockCount =
        static_cast<uint8_t>(Min(DETER_DQ_UB_SIZE / (constInfo.deterConstInfo.dqEachVectorSize * sizeof(T2)),
                                 constInfo.deterConstInfo.usedCubeCoreNum));
    uint8_t eachLoopStart = 0;
    uint8_t eachLoopEnd = 0;
    AscendC::SetAtomicAdd<T2>();
    WaitFlag<HardEvent::MTE2_S>(constInfo.deterConstInfo.eventIDMte2ToScalar);
    for (uint8_t loopIdx = 0; loopIdx < loopTimes; loopIdx++) {
        LocalTensor<T2> dqDeterBuf = deterInOutQue.AllocTensor<T2>();
        dataCopyParams.blockCount = (loopIdx < loopTimes - 1)
                                        ? eachLoopBlockCount
                                        : constInfo.deterConstInfo.usedCubeCoreNum - loopIdx * eachLoopBlockCount;
        if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum) {
            DataCopy(dqDeterBuf, deterGm[dqSrcOfs], dataCopyParams);
        }      
        dqSrcOfs += eachLoopBlockCount * BASE_DQ_SIZE;
        deterInOutQue.EnQue(dqDeterBuf);
        deterInOutQue.DeQue<T2>();

        eachLoopStart = loopIdx * eachLoopBlockCount;
        eachLoopEnd = Min((loopIdx + 1) * eachLoopBlockCount, constInfo.deterConstInfo.usedCubeCoreNum);
        for (uint16_t cIx = eachLoopStart; cIx < eachLoopEnd; cIx++) {
            dqOffset[cIx] = deterOffsetTensor.GetValue(cIx * INT64_BLOCK_NUM * 3);
        }
        // dq 每个V核需要处理所有C核的dq结果
        for (uint16_t cIx = eachLoopStart; cIx < eachLoopEnd; cIx++) {
            if (dqOffset[cIx] == OUTINDEX) {
                continue;
            }
            dqOffset[cIx] += constInfo.deterConstInfo.deterVecCoreS1Offset;
            if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum) {
                AscendC::DataCopyPad(dqWorkSpaceGm[dqOffset[cIx]],
                                 dqDeterBuf[(cIx - eachLoopStart) * constInfo.deterConstInfo.dqEachVectorSize],
                                 dataCopyPadParams);
            }
            PipeBarrier<PIPE_MTE3>();
        }
        deterInOutQue.FreeTensor(dqDeterBuf);
    }
    AscendC::SetAtomicNone();

    if (remainLoopNum > 0) {
        SetFlag<HardEvent::S_MTE2>(constInfo.deterConstInfo.eventIDScalarToMte2);
    }

    if (remainLoopNum > 2) { // 最后两轮不需要卡
        SyncAll<true, syncAllConfigMte3ToMte3>();
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::DeterComputeDkv(
    bool isLastSort, int64_t computeBlockIdx, int64_t remainLoopNum)
{
    SyncAll<>();
    LocalTensor<int64_t> deterOffsetTensor = deterOffsetBuf.Get<int64_t>();
    if (!isFirstDeter) {
        WaitFlag<HardEvent::S_MTE2>(constInfo.deterConstInfo.eventIDScalarToMte2);
    }
    DataCopy(deterOffsetTensor, deterOffsetGm[computeBlockIdx * constInfo.deterConstInfo.usedCubeCoreNum * INT64_BLOCK_NUM * 3],
             {1, static_cast<uint16_t>(constInfo.deterConstInfo.usedCubeCoreNum * 3), 0, 0});
    SetFlag<HardEvent::MTE2_S>(constInfo.deterConstInfo.eventIDMte2ToScalar);
    WaitFlag<HardEvent::MTE2_S>(constInfo.deterConstInfo.eventIDMte2ToScalar);
    for (uint16_t cIx = 0; cIx < constInfo.deterConstInfo.usedCubeCoreNum; cIx++) {
        dkOffset[cIx] = deterOffsetTensor.GetValue(cIx * INT64_BLOCK_NUM * 3 + INT64_BLOCK_NUM);
        dvOffset[cIx] = deterOffsetTensor.GetValue(cIx * INT64_BLOCK_NUM * 3 + INT64_BLOCK_NUM * 2);
    }
    int16_t bufId = 0;
    // 非last基本块处理的都是上一轮mm的结果， 最后一次循环需要补充处理本轮的mm结果
    if (isLastSort) {
        bufId = deterPpFlag;
    } else {
        bufId = 1 - deterPpFlag;
    }
    uint32_t dqSrcOfs = bufId * (BASE_DQ_SIZE + BASE_DKV_SIZE * 2) * constInfo.deterConstInfo.usedCubeCoreNum;
    uint32_t dkSrcOfs = dqSrcOfs + BASE_DQ_SIZE * constInfo.deterConstInfo.usedCubeCoreNum;
    uint32_t dvSrcOfs = dkSrcOfs + BASE_DKV_SIZE * constInfo.deterConstInfo.usedCubeCoreNum;
    dqSrcOfs += vBlockIdx * constInfo.deterConstInfo.dqEachVectorSize;
    dkSrcOfs += vBlockIdx * constInfo.deterConstInfo.dkvEachVectorSize;
    dvSrcOfs += vBlockIdx * constInfo.deterConstInfo.dkvEachVectorSize;

    // dk dv deter
    DataCopyParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.deterConstInfo.dkvEachVectorSize / FLOAT_BLOCK_SIZE;
    dataCopyParams.srcStride = BASE_DKV_SIZE / FLOAT_BLOCK_SIZE - dataCopyParams.blockLen;
    dataCopyParams.dstStride = 0;

    DataCopyExtParams dataCopyPadParams;
    dataCopyPadParams.blockCount = static_cast<uint16_t>(constInfo.deterConstInfo.eachVecCoreS2Offset);
    dataCopyPadParams.blockLen = static_cast<uint32_t>(constInfo.commonConstInfo.dSize * sizeof(T2));
    dataCopyPadParams.srcStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDqkSrcStride);
    dataCopyPadParams.dstStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDkDstStride);

    DataCopyExtParams dataCopyDvPadParams;
    dataCopyDvPadParams.blockCount = static_cast<uint16_t>(constInfo.deterConstInfo.eachVecCoreS2Offset);
    dataCopyDvPadParams.blockLen = static_cast<uint32_t>(constInfo.commonConstInfo.dSizeV * sizeof(T2));
    dataCopyDvPadParams.srcStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDvSrcStride);
    dataCopyDvPadParams.dstStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDvDstStride);

    uint32_t allSize = constInfo.deterConstInfo.usedCubeCoreNum * constInfo.deterConstInfo.dkvEachVectorSize * sizeof(T2);
    uint8_t loopTimes = static_cast<uint8_t>((allSize + DETER_DKV_UB_SIZE - 1) / DETER_DKV_UB_SIZE);
    uint8_t eachLoopBlockCount =
        static_cast<uint8_t>(Min(DETER_DKV_UB_SIZE / (constInfo.deterConstInfo.dkvEachVectorSize * sizeof(T2)),
                                 constInfo.deterConstInfo.usedCubeCoreNum));
    uint8_t eachLoopStart = 0;
    uint8_t eachLoopEnd = 0;
    AscendC::SetAtomicAdd<T2>();
    for (uint8_t loopIdx = 0; loopIdx < loopTimes; loopIdx++) {
        LocalTensor<T2> dkDeterBuf = mm1ResBuf[1 - bufId].Get<T2>();
        LocalTensor<T2> dvDeterBuf = mm2ResBuf[1 - bufId].Get<T2>();
        dataCopyParams.blockCount = (loopIdx < loopTimes - 1)
                                        ? eachLoopBlockCount
                                        : constInfo.deterConstInfo.usedCubeCoreNum - loopIdx * eachLoopBlockCount;
        if (loopIdx > 0) {
            WaitFlag<HardEvent::MTE3_MTE2>(constInfo.deterConstInfo.eventIDMte3ToMte2);
        }
        if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum) {
            DataCopy(dkDeterBuf, deterGm[dkSrcOfs], dataCopyParams);
            DataCopy(dvDeterBuf, deterGm[dvSrcOfs], dataCopyParams);
        }
        dkSrcOfs += eachLoopBlockCount * BASE_DKV_SIZE;
        dvSrcOfs += eachLoopBlockCount * BASE_DKV_SIZE;

        SetFlag<HardEvent::MTE2_MTE3>(constInfo.deterConstInfo.eventIDMte2ToMte3);
        WaitFlag<HardEvent::MTE2_MTE3>(constInfo.deterConstInfo.eventIDMte2ToMte3);

        eachLoopStart = loopIdx * eachLoopBlockCount;
        eachLoopEnd = Min((loopIdx + 1) * eachLoopBlockCount, constInfo.deterConstInfo.usedCubeCoreNum);
        // dq 每个V核需要处理所有C核的dk,dv结果
        for (uint16_t cIx = eachLoopStart; cIx < eachLoopEnd; cIx++) {
            if (dkOffset[cIx] == OUTINDEX && dvOffset[cIx] == OUTINDEX) {
                continue;
            }
            dkOffset[cIx] += constInfo.deterConstInfo.deterDkVecCoreS2Offset;
            dvOffset[cIx] += constInfo.deterConstInfo.deterDvVecCoreS2Offset;
            if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum) {
                AscendC::DataCopyPad(dkWorkSpaceGm[dkOffset[cIx]],
                                    dkDeterBuf[(cIx - eachLoopStart) * constInfo.deterConstInfo.dkvEachVectorSize],
                                    dataCopyPadParams);
                AscendC::DataCopyPad(dvWorkSpaceGm[dvOffset[cIx]],
                                    dvDeterBuf[(cIx - eachLoopStart) * constInfo.deterConstInfo.dkvEachVectorSize],
                                    dataCopyDvPadParams);
            }
            PipeBarrier<PIPE_MTE3>();
        }
        if (loopIdx < loopTimes - 1) {
            SetFlag<HardEvent::MTE3_MTE2>(constInfo.deterConstInfo.eventIDMte3ToMte2);
        }
    }
    AscendC::SetAtomicNone();

    if (remainLoopNum > 0) {
        SetFlag<HardEvent::S_MTE2>(constInfo.deterConstInfo.eventIDScalarToMte2);
    }
    if (remainLoopNum > 2) { // 最后两轮不需要卡
        SyncAll(); // 由于复用了mm1mm2的在ub中的buf，所以为了防止确定性计算还没有做完，后面的mm已经做完的情况，踩数据，所以只能卡scalar
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::DeterComputeDqkv(
    bool isLastSort, int64_t computeBlockIdx, int64_t remainLoopNum)
{
    SyncAll<>();
    LocalTensor<int64_t> deterOffsetTensor = deterOffsetBuf.Get<int64_t>();
    if (!isFirstDeter) {
        WaitFlag<HardEvent::S_MTE2>(constInfo.deterConstInfo.eventIDScalarToMte2);
    }
    DataCopy(deterOffsetTensor, deterOffsetGm[computeBlockIdx * constInfo.deterConstInfo.usedCubeCoreNum * INT64_BLOCK_NUM * 3],
             {1, static_cast<uint16_t>(constInfo.deterConstInfo.usedCubeCoreNum * 3), 0, 0});
    SetFlag<HardEvent::MTE2_S>(constInfo.deterConstInfo.eventIDMte2ToScalar);
    WaitFlag<HardEvent::MTE2_S>(constInfo.deterConstInfo.eventIDMte2ToScalar);
    for (uint16_t cIx = 0; cIx < constInfo.deterConstInfo.usedCubeCoreNum; cIx++) {
        dqOffset[cIx] = deterOffsetTensor.GetValue(cIx * INT64_BLOCK_NUM * 3);
        dkOffset[cIx] = deterOffsetTensor.GetValue(cIx * INT64_BLOCK_NUM * 3 + INT64_BLOCK_NUM);
        dvOffset[cIx] = deterOffsetTensor.GetValue(cIx * INT64_BLOCK_NUM * 3 + INT64_BLOCK_NUM * 2);
    }
    int16_t bufId = 0;
    // 非last基本块处理的都是上一轮mm的结果， 最后一次循环需要补充处理本轮的mm结果
    if (isLastSort) {
        bufId = deterPpFlag;
    } else {
        bufId = 1 - deterPpFlag;
    }
    uint32_t dqSrcOfs = bufId * (BASE_DQ_SIZE + BASE_DKV_SIZE * 2) * constInfo.deterConstInfo.usedCubeCoreNum;
    uint32_t dkSrcOfs = dqSrcOfs + BASE_DQ_SIZE * constInfo.deterConstInfo.usedCubeCoreNum;
    uint32_t dvSrcOfs = dkSrcOfs + BASE_DKV_SIZE * constInfo.deterConstInfo.usedCubeCoreNum;
    dqSrcOfs += vBlockIdx * constInfo.deterConstInfo.dqEachVectorSize;
    dkSrcOfs += vBlockIdx * constInfo.deterConstInfo.dkvEachVectorSize;
    dvSrcOfs += vBlockIdx * constInfo.deterConstInfo.dkvEachVectorSize;

    // dq deter
    DataCopyParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.deterConstInfo.dqEachVectorSize / FLOAT_BLOCK_SIZE;
    dataCopyParams.srcStride = BASE_DQ_SIZE / FLOAT_BLOCK_SIZE - dataCopyParams.blockLen;
    dataCopyParams.dstStride = 0;
    DataCopyExtParams dataCopyPadParams;
    dataCopyPadParams.blockCount = static_cast<uint16_t>(constInfo.deterConstInfo.eachVecCoreS1Offset);
    dataCopyPadParams.blockLen = static_cast<uint32_t>(constInfo.commonConstInfo.dSize * sizeof(T2));
    dataCopyPadParams.srcStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDqkSrcStride);
    dataCopyPadParams.dstStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDqDstStride);

    uint32_t allSize =
        constInfo.deterConstInfo.usedCubeCoreNum * constInfo.deterConstInfo.dqEachVectorSize * sizeof(T2);
    uint8_t loopTimes = static_cast<uint8_t>((allSize + DETER_DQ_UB_SIZE - 1) / DETER_DQ_UB_SIZE);
    uint8_t eachLoopBlockCount =
        static_cast<uint8_t>(Min(DETER_DQ_UB_SIZE / (constInfo.deterConstInfo.dqEachVectorSize * sizeof(T2)),
                                 constInfo.deterConstInfo.usedCubeCoreNum));
    uint8_t eachLoopStart = 0;
    uint8_t eachLoopEnd = 0;
    AscendC::SetAtomicAdd<T2>();
    for (uint8_t loopIdx = 0; loopIdx < loopTimes; loopIdx++) {
        LocalTensor<T2> dqDeterBuf = deterInOutQue.AllocTensor<T2>();
        dataCopyParams.blockCount = (loopIdx < loopTimes - 1)
                                        ? eachLoopBlockCount
                                        : constInfo.deterConstInfo.usedCubeCoreNum - loopIdx * eachLoopBlockCount;
        if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum) {
            DataCopy(dqDeterBuf, deterGm[dqSrcOfs], dataCopyParams);
        }
        dqSrcOfs += eachLoopBlockCount * BASE_DQ_SIZE;
        deterInOutQue.EnQue(dqDeterBuf);
        deterInOutQue.DeQue<T2>();

        eachLoopStart = loopIdx * eachLoopBlockCount;
        eachLoopEnd = Min((loopIdx + 1) * eachLoopBlockCount, constInfo.deterConstInfo.usedCubeCoreNum);
        // dq 每个V核需要处理所有C核的dq结果
        for (uint16_t cIx = eachLoopStart; cIx < eachLoopEnd; cIx++) {
            if (dqOffset[cIx] == OUTINDEX) {
                continue;
            }
            dqOffset[cIx] += constInfo.deterConstInfo.deterVecCoreS1Offset;
            if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum) {
                AscendC::DataCopyPad(dqWorkSpaceGm[dqOffset[cIx]],
                                    dqDeterBuf[(cIx - eachLoopStart) * constInfo.deterConstInfo.dqEachVectorSize],
                                    dataCopyPadParams);
            }
            PipeBarrier<PIPE_MTE3>();
        }
        deterInOutQue.FreeTensor(dqDeterBuf);
    }

    // dk dv deter
    dataCopyParams.blockLen = constInfo.deterConstInfo.dkvEachVectorSize / FLOAT_BLOCK_SIZE;
    dataCopyParams.srcStride = BASE_DKV_SIZE / FLOAT_BLOCK_SIZE - dataCopyParams.blockLen;
    dataCopyParams.dstStride = 0;

    dataCopyPadParams.blockCount = static_cast<uint16_t>(constInfo.deterConstInfo.eachVecCoreS2Offset);
    dataCopyPadParams.blockLen = static_cast<uint32_t>(constInfo.commonConstInfo.dSize * sizeof(T2));
    dataCopyPadParams.srcStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDqkSrcStride);
    dataCopyPadParams.dstStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDkDstStride);

    DataCopyExtParams dataCopyDvPadParams;
    dataCopyDvPadParams.blockCount = static_cast<uint16_t>(constInfo.deterConstInfo.eachVecCoreS2Offset);
    dataCopyDvPadParams.blockLen = static_cast<uint32_t>(constInfo.commonConstInfo.dSizeV * sizeof(T2));
    dataCopyDvPadParams.srcStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDvSrcStride);
    dataCopyDvPadParams.dstStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDvDstStride);

    allSize = constInfo.deterConstInfo.usedCubeCoreNum * constInfo.deterConstInfo.dkvEachVectorSize * sizeof(T2);
    loopTimes = static_cast<uint8_t>((allSize + DETER_DKV_UB_SIZE - 1) / DETER_DKV_UB_SIZE);
    eachLoopBlockCount =
        static_cast<uint8_t>(Min(DETER_DKV_UB_SIZE / (constInfo.deterConstInfo.dkvEachVectorSize * sizeof(T2)),
                                 constInfo.deterConstInfo.usedCubeCoreNum));
    eachLoopStart = 0;
    eachLoopEnd = 0;

    for (uint8_t loopIdx = 0; loopIdx < loopTimes; loopIdx++) {
        LocalTensor<T2> dkDeterBuf = mm1ResBuf[1 - bufId].Get<T2>();
        LocalTensor<T2> dvDeterBuf = mm2ResBuf[1 - bufId].Get<T2>();
        dataCopyParams.blockCount = (loopIdx < loopTimes - 1)
                                        ? eachLoopBlockCount
                                        : constInfo.deterConstInfo.usedCubeCoreNum - loopIdx * eachLoopBlockCount;
        if (loopIdx > 0) {
            WaitFlag<HardEvent::MTE3_MTE2>(constInfo.deterConstInfo.eventIDMte3ToMte2);
        }
        if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum) {
            DataCopy(dkDeterBuf, deterGm[dkSrcOfs], dataCopyParams);
            DataCopy(dvDeterBuf, deterGm[dvSrcOfs], dataCopyParams);
        }
        dkSrcOfs += eachLoopBlockCount * BASE_DKV_SIZE;
        dvSrcOfs += eachLoopBlockCount * BASE_DKV_SIZE;

        SetFlag<HardEvent::MTE2_MTE3>(constInfo.deterConstInfo.eventIDMte2ToMte3);
        WaitFlag<HardEvent::MTE2_MTE3>(constInfo.deterConstInfo.eventIDMte2ToMte3);

        eachLoopStart = loopIdx * eachLoopBlockCount;
        eachLoopEnd = Min((loopIdx + 1) * eachLoopBlockCount, constInfo.deterConstInfo.usedCubeCoreNum);
        // dq 每个V核需要处理所有C核的dk,dv结果
        for (uint16_t cIx = eachLoopStart; cIx < eachLoopEnd; cIx++) {
            if (dkOffset[cIx] == OUTINDEX && dvOffset[cIx] == OUTINDEX) {
                continue;
            }
            dkOffset[cIx] += constInfo.deterConstInfo.deterDkVecCoreS2Offset;
            dvOffset[cIx] += constInfo.deterConstInfo.deterDvVecCoreS2Offset;
            if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum) {
                AscendC::DataCopyPad(dkWorkSpaceGm[dkOffset[cIx]],
                                    dkDeterBuf[(cIx - eachLoopStart) * constInfo.deterConstInfo.dkvEachVectorSize],
                                    dataCopyPadParams);
                AscendC::DataCopyPad(dvWorkSpaceGm[dvOffset[cIx]],
                                    dvDeterBuf[(cIx - eachLoopStart) * constInfo.deterConstInfo.dkvEachVectorSize],
                                    dataCopyDvPadParams);
            }
            PipeBarrier<PIPE_MTE3>();
        }
        if (loopIdx < loopTimes - 1) {
            SetFlag<HardEvent::MTE3_MTE2>(constInfo.deterConstInfo.eventIDMte3ToMte2);
        }
    }
    AscendC::SetAtomicNone();

    if (remainLoopNum > 0) {
        SetFlag<HardEvent::S_MTE2>(constInfo.deterConstInfo.eventIDScalarToMte2);
    }
    if (remainLoopNum > 2) { // 最后两轮不需要卡
        SyncAll(); // 由于复用了mm1mm2的在ub中的buf，所以为了防止确定性计算还没有做完，后面的mm已经做完的情况，踩数据，所以只能卡scalar
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::DeterCompute(
    bool isLastSort, int64_t computeBlockIdx, int64_t remainLoopNum)
{
    int32_t pingpangIdx = computeBlockIdx & 1;
    if (likely(constInfo.deterConstInfo.noNeedDeter)) {
        if (remainLoopNum > 1) {
            SyncAll<true>();
        }
    } else {
        if (dqIsNeedDeter[pingpangIdx] && !dkDvIsNeedDeter[pingpangIdx]) {
            // 该轮次只有dq需要做deter
            DeterComputeDq(isLastSort, computeBlockIdx, remainLoopNum);
            isFirstDeter = false;
        } else if (!dqIsNeedDeter[pingpangIdx] && dkDvIsNeedDeter[pingpangIdx]) {
            // 该轮次只有dk dv需要做deter
            DeterComputeDkv(isLastSort, computeBlockIdx, remainLoopNum);
            isFirstDeter = false;
        } else if (dqIsNeedDeter[pingpangIdx] && dkDvIsNeedDeter[pingpangIdx]) {
            // 该轮次dq dk dv都需要做deter
            DeterComputeDqkv(isLastSort, computeBlockIdx, remainLoopNum);
            isFirstDeter = false;
        } else {
            // 该伦茨不需要做确定性计算，只需要加上全核同步
            if (remainLoopNum > 1) {
                SyncAll();
            }
        }
    }
}

FAG_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2StaticRegbase<FAG_FUNCTION_PARAMS_TEMPLATE>::SyncALLCores()
{
    SyncAll();
}

#endif // _FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2GS1S2_REGBASE_H_
