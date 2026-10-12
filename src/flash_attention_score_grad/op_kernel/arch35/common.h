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
 * \file common.h
 * \brief
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../../common/op_kernel/arch35/util_regbase.h"
#include <cstdint>

using namespace regbaseutil;
using AscendC::Cast;
using AscendC::RoundMode;

namespace commondef {
// 自主管理L1后，计算所需要的L1空间大小，分为preload和非preload两种模式
#define GET_L1_WITH_PRELOAD_USED_SIZE(S1_BASE, D_ALIGN, S2_BASE)                                                       \
    ((S1_BASE) * (D_ALIGN) * 20 + (S1_BASE) * (S2_BASE) * 4)
#define GET_L1_NO_PRELOAD_USED_SIZE(S1_BASE, D_ALIGN, S2_BASE) ((S1_BASE) * (D_ALIGN) * 16 + (S1_BASE) * (S2_BASE) * 4)
// 是否开启自主管理L1
#define IS_TSCM_REUSE(HEAD_DIM_ALIGN, T1, IS_DETER_OLD, FP8_OPEN_TSCM)                                                                              \
    ((((!(IS_DETER_OLD) && (HEAD_DIM_ALIGN) <= (uint32_t)DTemplateType::Aligned256) ||                                           \
     ((IS_DETER_OLD) && (HEAD_DIM_ALIGN) <= (uint32_t)DTemplateType::Aligned192)) && (!IsSameType<T1, float>::value) && (!IsSameType<T1, fp8_e5m2_t>::value)) \
        && (!IsSameType<T1, fp8_e4m3fn_t>::value) && (!IsSameType<T1, hifloat8_t>::value) || (FP8_OPEN_TSCM && (IsSameType<T1, fp8_e5m2_t>::value || IsSameType<T1, fp8_e4m3fn_t>::value || IsSameType<T1, hifloat8_t>::value)))
// 是否开启L1 preload，限制条件为D <= 192，非BN2模板，非确定性计算，非TND
#define IS_TSCM_PRELOAD_ROPE(HEAD_DIM_ALIGN, T1, SPLIT_AXIS, IS_DETER_OLD, IS_TND, FP8_OPEN_TSCM, IS_ROPE)                                              \
    ((((HEAD_DIM_ALIGN) <= (uint32_t)DTemplateType::Aligned192) && (((SPLIT_AXIS) == 0) || ((SPLIT_AXIS) == 5)) && ((IS_DETER_OLD) == 0 || ((IS_DETER_OLD) == 1 && IS_ROPE)) &&    \
     (!IsSameType<T1, float>::value) && (!IsSameType<T1, fp8_e5m2_t>::value) && (!IsSameType<T1, fp8_e4m3fn_t>::value) && (!IsSameType<T1, hifloat8_t>::value)) ||       \
     (FP8_OPEN_TSCM && ((HEAD_DIM_ALIGN) <= (uint32_t)DTemplateType::Aligned256) && (IsSameType<T1, fp8_e5m2_t>::value || IsSameType<T1, fp8_e4m3fn_t>::value || IsSameType<T1, hifloat8_t>::value)))
// 是否开启L1 preload，限制条件为D <= 192，非BN2模板，非确定性计算，非TND
#define IS_TSCM_PRELOAD(HEAD_DIM_ALIGN, T1, SPLIT_AXIS, IS_DETER_OLD, IS_TND)                                          \
    (IS_TSCM_PRELOAD_ROPE(HEAD_DIM_ALIGN, T1, SPLIT_AXIS, IS_DETER_OLD, IS_TND, false, false))
// 计算所需要的L0C空间大小
#define GET_SHARED_C1_BUFFER_SZIE(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN)                                              \
    ((CUBE_BASEN) > (HEAD_DIM_ALIGN)                                                                                   \
         ? ((CUBE_BASEM) > (HEAD_DIM_ALIGN) ? (CUBE_BASEM) * (CUBE_BASEN) * sizeof(float)                              \
                                            : (CUBE_BASEN) * (HEAD_DIM_ALIGN) * sizeof(float))                         \
         : ((CUBE_BASEM) > (CUBE_BASEN) ? (CUBE_BASEM) * (HEAD_DIM_ALIGN) * sizeof(float)                              \
                                        : (CUBE_BASEN) * (HEAD_DIM_ALIGN) * sizeof(float)))
// L0C自主管理相关
constexpr uint32_t MIN_L0C_BUF_NUM = 4;
constexpr int32_t DK_DV_L0C_BUF_NUM = 2;
constexpr uint32_t L0C_MAX_SIZE = 256 * 1024;
#define GET_MM1_MM2_MM3_MAX_L0C_SIZE(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN)                                           \
    ((CUBE_BASEN) > (HEAD_DIM_ALIGN) ? (CUBE_BASEM) * (CUBE_BASEN) * sizeof(float)                                     \
                                     : (CUBE_BASEM) * (HEAD_DIM_ALIGN) * sizeof(float))
#define GET_MAX_REMAIN_L0C_NUM(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN)                                                 \
    ((L0C_MAX_SIZE - (CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(float) * DK_DV_L0C_BUF_NUM)) /                              \
     (GET_MM1_MM2_MM3_MAX_L0C_SIZE(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN)))

#define GET_L0C_BUF_NUM(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN)                                                        \
    (DK_DV_L0C_BUF_NUM + (int32_t)GET_MAX_REMAIN_L0C_NUM(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN) <= 0                  \
         ? 1                                                                                                           \
         : DK_DV_L0C_BUF_NUM + (int32_t)GET_MAX_REMAIN_L0C_NUM(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN))

#define IS_L0C_REUSE(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN, IS_DETER_OLD, T1, IS_TND)                                     \
    ((GET_L0C_BUF_NUM(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN) >= MIN_L0C_BUF_NUM) && !(IS_DETER_OLD) &&                    \
     (!IsSameType<T1, float>::value) && (!IS_TND) &&                                                                   \
     (!(IsSameType<T1, fp8_e5m2_t>::value || IsSameType<T1, fp8_e4m3fn_t>::value || IsSameType<T1, hifloat8_t>::value)))

constexpr uint32_t L0_MAX_SIZE = 64 * 1024;
constexpr uint32_t L1_MAX_SIZE = 512 * 1024;
// 当前判断仅在FP32场景生效，后续需考虑FP16/BF16并结合L0DB开关
#define IS_L0_EXCEED(M, N, K, T1) (M * K * sizeof(T1) > L0_MAX_SIZE || K * N * sizeof(T1) > L0_MAX_SIZE);

#define FagOldTilingType                                                                                                  \
    const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<NEED_DETER_PREFIX(DETER_SPARSE_TYPE, IS_TND), IS_TND, false> \
        *__restrict

#define FagTilingType                                                                                                  \
    const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<NEED_DETER_PREFIX(DETER_SPARSE_TYPE, IS_TND), IS_TND, IS_TND_SWIZZLE> \
        *__restrict


constexpr uint32_t RESERVED_WORKSPACE_SIZE = 64 * 1024;
constexpr bool INPUT_DISABLE = 0;
constexpr bool INPUT_ENABLE = 1;

// SPLIT_AXIS Enum
constexpr uint8_t BN2GS1S2 = 0;
constexpr uint8_t BN2 = 1;
constexpr uint8_t BN2S2 = 5;

// D_TYPE
constexpr uint8_t FLOAT16_PRECISION = 3;
constexpr uint8_t BFLOAT16 = 2;

constexpr uint32_t BSNGD = 1;
constexpr uint32_t SBNGD = 2;
constexpr uint32_t BNGSD = 3;
constexpr uint32_t TND = 4;

constexpr uint32_t PREFIX_LENGTH = 64;
constexpr uint32_t SEQ_ARR_LENGTH = 256;
constexpr uint32_t ADDR_ALIGN_SIZE = 512;
constexpr uint32_t INPUT_NUMS = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t FLOAT_BLOCK_SIZE = 32 / sizeof(float);
constexpr uint32_t MAX_BASIC_BLOCK_SIZE = 1024;
constexpr uint32_t PSE_PERFORMANCE_MODE = 0x12;
constexpr uint32_t MAX_CUBE_CORE_NUM = 36;
constexpr uint32_t MAX_SUM_REDUCE_AXIS_SIZE = 32;
constexpr uint32_t C0_SIZE = 16;
constexpr uint32_t CV_CORE_RATIO = 2;
constexpr uint32_t GROUP_TSCM_MASK = 0x4;
constexpr uint32_t TSCM_BUF_NUM = 9;
// rope
constexpr uint32_t ROPE_D_RATIO = 3;
constexpr uint32_t ROPE_D_128 = 128;
constexpr uint32_t ROPE_D_64 = 64;

// sparse mode
constexpr uint32_t NO_MASK = 0; // 未传入attenmask，不做mask操作
constexpr uint32_t ALL_MASK = 1;
constexpr uint32_t LEFT_UP_CAUSAL = 2;    // 左上角点划分的三角部分
constexpr uint32_t RIGHT_DOWN_CAUSAL = 3; // 右下角点划分的下三角部分
constexpr uint32_t BAND = 4;
constexpr uint32_t PREFIX = 5;
constexpr uint32_t PREFIX_COMPRESS = 6;
constexpr uint32_t RIGHT_DOWN_CASUAL_BAND = 7;
constexpr uint32_t BAND_LEFT_UP_CASUAL = 8;

// deter sparse type
// 非确定性
constexpr uint8_t NO_DETER = 0;
// 确定性老实现方案
constexpr uint8_t DETER_OLD = 1;
constexpr uint8_t DETER_DENSE = 2;
constexpr uint8_t DETER_CAUSAL = 3;
constexpr uint8_t DETER_BAND = 4;

constexpr uint16_t ALIGN_OFFSET_16 = 15;
constexpr uint16_t ALIGN_OFFSET_32 = 31;
constexpr uint16_t ALIGN_OFFSET_64 = 63;
constexpr uint16_t ALIGN_OFFSET_128 = 127;
constexpr uint16_t ALIGN_OFFSET_512 = 511;

constexpr uint16_t OFFSET_BITS_4 = 4;
constexpr uint16_t OFFSET_BITS_5 = 5;
constexpr uint16_t OFFSET_BITS_6 = 6;
constexpr uint16_t OFFSET_BITS_7 = 7;
constexpr uint16_t OFFSET_BITS_9 = 9;

// pse shape type same as tiling
constexpr uint32_t PSE_SHAPE_TYPE_BNSS = 0;
constexpr uint32_t PSE_SHAPE_TYPE_BN1S = 1;
constexpr uint32_t PSE_SHAPE_TYPE_1NSS = 2;
constexpr uint32_t PSE_SHAPE_TYPE_BNHS = 3;
constexpr uint32_t PSE_SHAPE_TYPE_1NHS = 4;
constexpr uint32_t PSE_B_N2_G_SLOPE = 5;
constexpr uint32_t PSE_1_N2_G_SLOPE = 6;

constexpr uint32_t PSE_COMPRESS_H = 1024;
constexpr uint32_t VREG_SIZE = 256;
constexpr uint32_t MAX_CONTINUOUS_BLOCK_NUM = 6;
constexpr uint16_t UNROLL_FACTOR = 2;

struct DeterConstInfo {
    uint8_t usedCubeCoreNum;
    uint8_t usedVectorCoreNum;
    uint8_t eachVecCoreS1Offset;
    uint8_t eachVecCoreS2Offset;
    uint32_t dqEachVectorSize;
    uint32_t dkvEachVectorSize;
    // 确定性计算MTE3的stride
    int64_t deterBStride;
    int64_t deterN2Stride;
    int64_t deterGStride;
    int64_t deterS1oStride;
    uint32_t deterDqkSrcStride;
    uint32_t deterDvSrcStride;
    uint32_t deterDqDstStride;
    uint32_t deterDkDstStride;
    uint32_t deterDvDstStride;
    uint32_t deterVecCoreS1Offset;
    uint32_t deterDkVecCoreS2Offset;
    uint32_t deterDvVecCoreS2Offset;
    bool noNeedDeter;
    // event_id
    event_t eventIDScalarToMte2;
    event_t eventIDMte2ToScalar;
    event_t eventIDScalarToMte3;
    event_t eventIDMte3ToScalar;
    event_t eventIDMte3ToMte2;
    event_t eventIDMte2ToMte3;
};

struct FagConstInfo {
    ConstInfo<false, false> commonConstInfo{0};
    DeterConstInfo deterConstInfo{0};
    int64_t bSize;
    int64_t n2Size;
    float scaleValue;
    float attenMaskMinValue;

    // quant
    float pScale; // 量化参数
    float dsScale; // 量化参数
    float pScaleD; // 反量化参数
    float dsScaleD; // 反量化参数
    float pScaleLog; // log(pScale)

    int64_t copyOutDStride;

    // 轴的乘积
    int64_t gS1o;
    int64_t n2GS1o;
    int64_t n2GS1oS2o;
    int64_t gS1oS2o;
    int64_t s1oS2o;

    __gm__ uint8_t *seqS1_addr;
    __gm__ uint8_t *seqS2_addr;

    int64_t s1Token;
    int64_t s2Token;
    int64_t sparseMode;
    int64_t s1Outer;
    int64_t s2Outer;
    int64_t s1CvTail;
    int64_t s1Tail;
    int64_t s2Tail;
    uint32_t sfmgMaxLoopSize;
    uint32_t dAlignToBlock;
    uint32_t dAlignToBlockForFp8;
    int64_t mm2Ka;
    int64_t mm2Kb;
    int64_t mm3Ka;
    int64_t mm4Kb;
    int64_t dRopeSize = 64; // rope旋转的维度
    uint32_t continuousBlockNum = 0; // 核内连续块数量
    // swizzle相关
    int64_t mSwizzleBlockNum = 0;
    int64_t mSwizzleBlockNumTail = 0;
    int64_t nSwizzleBlockNum = 0;
    int64_t nSwizzleBlockNumTail = 0;
    int64_t leftUpTotalRound = 0;
    int64_t leftDownTotalRound = 0;
    int64_t rightUpTotalRound = 0;
    int64_t rightDownTotalRound = 0;
    int64_t leftSingleColTotalRound = 0;
    int64_t leftTotalRound = 0;
    int64_t batchTotalRound = 0;
    // 核数
    uint32_t aicCoreNum = 0;
};

// fp8反量化因子
struct QuantScaleInfo {
	float deqScaleQValue = 1.0f;
	float deqScaleKValue = 1.0f;
	float deqScaleVValue = 1.0f;
	float deqScaleDyValue = 1.0f;
	float deqScaleOValue = 1.0f;
};
// FP8场景字段
struct QuantRunInfo {
    uint32_t s1ProcessSize;
    uint32_t s2ProcessSize;
    uint32_t s1Idx;
    uint32_t s2Idx;
    bool isDqFixOut;
    bool isDkFixOut;
    bool isDvFixOut;

    float qkDScale; // qDScale * kDScale
    float vdyDScale; // dyDScale * vDScale
    float dsScaleDMulDeqScaleK; // dsScaleD * deqScaleKValue
    float dsScaleDMulDeqScaleQ; // pScaleD * deqScaleDyValue
    float pScaleDMulDeqScaleDy; // dsScaleD * deqScaleQValue

    bool isDkvCompleted = true;
    bool isDqCompleted = true;

    // 512基本块内部，再切128
    int64_t innerS1RealSize[4];
    int64_t innerS2RealSize[4];
    int64_t innerS1LoopNum;
    int64_t innerS2LoopNum;

    int64_t qInnerOffset[4];
    int64_t kvInnerOffset[4];
};
struct FagRunInfo {
    RunInfo<false> commonRunInfo{0};
	QuantScaleInfo quantScaleInfo;
    QuantRunInfo quantRunInfo;
    int64_t s2oIdx;
    int64_t s2CvBegin;
    int64_t s2CvEnd;
    int64_t kGmS2SplitOffset = 0;
    int64_t vGmS2SplitOffset = 0;
    uint8_t kvPingPong;
    int32_t s1RealSizeAlign2;
    int32_t s2RealSizeAlign2;
    int64_t dAlign16;
    int32_t halfS2RealSize; // vector侧实际的s2基本块大小，如果Cube基本块=128，那么halfS2RealSize=64
    int32_t
        firstHalfS2RealSize; // 当s2RealSize不是2的整数倍时，v0比v1少计算一行，计算subblock偏移的时候需要使用v0的s2 size
    uint8_t qDxPingPongIdx;
    uint8_t isS2IdxNoChange; // s2Idx是否变化
    uint8_t isNextS2IdxNoChange; // 下一个基本块的s2Idx是否变化（是否切换了列）
    // BN2模板使用
    bool isLastS1Outer = false; // 标记BN2扩展模板中是否是S1轴要处理的最后一个s1outer
    bool isFirstS1Outer = false; // 标记BN2扩展模板中是否是S1轴要处理的第一个s1outer

    // TND需要记录上一次的基本块的信息，用于优化scalar
    int64_t lastBatchIdx = 0;
    int64_t lastBatchTotalBaseIdx = 0;
    int64_t lastBatchTotalS1BOffset = 0;
    int64_t lastBatchTotalS1BRopeOffset = 0;
    int64_t lastBatchTotalS1BOffsetForDv = 0;
    int64_t lastBatchTotalS2BOffset = 0;
    int64_t lastBatchTotalS2BRopeOffset = 0;
    int64_t lastBatchTotalS2BOffsetForDv = 0;
    int64_t lastBatchTotalS1S2SizeAlign = 0;
    int64_t lastBatchTotalS1S2Size = 0;
    int64_t lastBatchTotalS2Size = 0;
    // 只有确定性计算使用
    bool completed = true;
    int64_t dyOffset;
    int64_t queryOffsetWithRope;
    int64_t keyOffsetWithRope;
    int64_t queryOffsetWithRopeForMm12;
    int64_t keyOffsetWithRopeForMm12;
    int8_t specialS2Index = -1;

    bool isFirstBlock = true;
    bool isKeyReuse = false;
    bool isValueReuse = false;
    bool isNextKeyReuse = false;
    bool isLastProcessBlock = false;
    bool isFirstProcessBlock = false;
    
    int64_t maxsumOffset;
};

constexpr SyncAllConfig syncAllConfigMte2ToMte2 = {PIPE_MTE2, PIPE_MTE2};
constexpr SyncAllConfig syncAllConfigMte3ToMte2 = {PIPE_MTE3, PIPE_MTE2};
constexpr SyncAllConfig syncAllConfigMte3ToMte3 = {PIPE_MTE3, PIPE_MTE3};
constexpr SyncAllConfig syncAllConfigMte2ToMte3 = {PIPE_MTE2, PIPE_MTE3};

struct LoopInfo {
    int64_t bIdx{0};
    int64_t n2Idx{0};
    int64_t gIdx{0};
    int64_t s1oIdx{0};
    int64_t s2oIdx{0};
};

struct Bn2MultiBlkInfo {
    int64_t s2oDimIdx{0};
    int64_t s2OuterTmp{0}; // TND场景此值不可直接使用constinfo中的S2Outer
    int64_t s2SparseLeft{0};
    int64_t s2SparseRight{0};
};

__aicore__ inline uint32_t AlignTo(uint32_t num1, uint32_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

__aicore__ inline int64_t AlignTo16(int64_t num) { return (num + ALIGN_OFFSET_16) >> OFFSET_BITS_4 << OFFSET_BITS_4; }

__aicore__ inline int64_t AlignTo32(int64_t num) { return (num + ALIGN_OFFSET_32) >> OFFSET_BITS_5 << OFFSET_BITS_5; }

__aicore__ inline int64_t AlignTo64(int64_t num) { return (num + ALIGN_OFFSET_64) >> OFFSET_BITS_6 << OFFSET_BITS_6; }

__aicore__ inline int64_t AlignTo128(int64_t num) { return (num + ALIGN_OFFSET_128) >> OFFSET_BITS_7 << OFFSET_BITS_7; }

__aicore__ inline int64_t AlignTo512(int64_t num) { return (num + ALIGN_OFFSET_512) >> OFFSET_BITS_9 << OFFSET_BITS_9; }

__aicore__ constexpr bool IS_DETER_OLD(const uint8_t deterSparseType) 
{
    return deterSparseType == DETER_OLD;
}
 
__aicore__ constexpr bool IS_DETER_NEW(const uint8_t deterSparseType) 
{
    return deterSparseType != NO_DETER && deterSparseType != DETER_OLD;
}
 
__aicore__ constexpr bool NEED_DETER_PREFIX(const uint8_t deterSparseType, bool isTnd) 
{
    return isTnd && IS_DETER_NEW(deterSparseType);
}
}
