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
 * \file flash_attention_score_grad_tiling_common.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <vector>
#include <register/tilingdata_base.h>
#include <register/op_impl_registry.h>
#include <tiling/tiling_api.h>

namespace optiling {
namespace fag {

constexpr int64_t BYTE_PER_BLOCK = 32; // 32 B in block
constexpr int HIGH_PRECISION = 0;
constexpr int HIGH_PERFORMANCE = 1;

constexpr const char *BSH_STR = "BSH";
constexpr const char *SBH_STR = "SBH";
constexpr const char *BNSD_STR = "BNSD";
constexpr const char *BSND_STR = "BSND";
constexpr const char *TND_STR = "TND";
constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;
constexpr size_t DIM_4 = 4;
constexpr size_t LAST_AXIS_IDX = 1;
constexpr size_t SEC_LAST_AXIS_IDX = 2;
constexpr uint32_t MULT_BASE = 2;
constexpr int64_t SINGLE_VEC_INST_DATASIZE = 256;
constexpr uint32_t DEFAULT_DATA_TYPE_SIZE = 4;
constexpr uint32_t DEFAULT_MASK = 64;
constexpr uint32_t FP16_DATA_TYPE_SIZE = 2;
constexpr uint32_t BF16_DATA_TYPE_SIZE = 2;
constexpr uint32_t FP16_MASK = 128;
constexpr uint32_t BF16_MASK = 64;
constexpr int64_t FRACTAL_NUM = 16;     // 16 is 分形大小
constexpr uint32_t CUBE_ALIGN_NUM = 16; // 16 is cube align num
constexpr uint32_t BYTE_BLOCK = 32;     // 32 B in block
constexpr uint32_t BATCH_MAX_SIZE = 2048;
constexpr uint32_t PREFIX_COMPRESS_S1_SIZE = 3072;
constexpr uint32_t ATTEN_MASK_COMPRESS_LIMIT = 2048;
constexpr uint32_t BOOL_BLOCK_NUMS = 32;
constexpr uint32_t DROPOUT4BIT_LEN = 16;
const int64_t SAB_TND_SIZE = 1024;

enum class TilingDataType : uint32_t {
    FP16 = 1,
    BF16 = 2,
    FP32 = 3,
    INHP
};

enum class InputLayout : uint32_t {
    BSH = 0,
    SBH = 1,
    BNSD = 2,
    BSND = 3,
    TND
};

enum class AxisIdx : uint32_t {
    B = 0,
    S = 1,
    H = 2
};

enum class Axis4Idx : uint32_t {
    AXIS4_B = static_cast<uint32_t>(AxisIdx::B),
    AXIS4_S = static_cast<uint32_t>(AxisIdx::S),
    AXIS4_N = 2,
    AXIS4_D = 3
};

/* layout和b,s,h三根轴的位置关系映射 */
const std::vector<std::vector<size_t>> LAYOUT_TO_AXIS{
    // 3根轴对应dimid
    {0, 1, 2}, // BSH
    {1, 0, 2}, // SBH
    // 4根轴对应dimid
    {0, 2, 1, 3}, // BNSD
    {0, 1, 2, 3}  // BSND
};

enum class InputIndex : uint32_t {
    QUERY = 0,
    KEY,
    VALUE,
    DY,
    PSE_SHIFT,
    DROP_MASK,
    PADDING_MASK,
    ATTEN_MASK,
    SOFTMAX_MAX,
    SOFTMAX_SUM,
    SOFTMAX_IN,
    ATTENTION_IN,
    PREFIX_N,
    ACTUAL_SEQ_Q_LEN,
    ACTUAL_SEQ_KV_LEN,
    Q_START_IDX,
    KV_START_IDX,
    D_SCALE_Q,
    D_SCALE_K,
    D_SCALE_V,
    D_SCALE_DY,
    D_SCALE_O,
    QUERY_ROPE_IDX,
    KEY_ROPE_IDX,
    SINK_IDX,
    D_SCALE_DS_IDX,
    D_SCALE_P_IDX
};

enum class AttenMaskCompressMode : uint8_t {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE,
    RIGHT_DOWN_CAUSAL_MODE,
    BAND_EQUAL_S_MODE,
    PREFIX_COMPRESS_MODE,
    RIGHT_DOWN_CASUAL_BAND_MODE,
    BAND_LEFT_UP_CASUAL_MODE
};

enum class AttrIndex : uint32_t {
    SCALE_VALUE = 0,
    KEEP_PROB,
    PRE_TOKENS,
    NEXT_TOKENS,
    HEAD_NUM,
    INPUT_LAYOUT,
    INNER_PRECISE,
    SPARSE_MODE,
    PSETYPE,
    SEED,
    OFFSET,
    OUT_TYPE,
    TND_SOFTMAX_IN
};

enum class PseShapeType : uint32_t {
    PSE_SHAPE_TYPE_BNSS,
    PSE_SHAPE_TYPE_BN1S,
    PSE_SHAPE_TYPE_1NSS,
    PSE_SHAPE_TYPE_BNHS,
    PSE_SHAPE_TYPE_1NHS,
    PSE_B_N2_G_SLOPE,
    PSE_1_N2_G_SLOPE
};

enum class PseType : uint8_t {
    PSE_OUTER_MUL_ADD_TYPE = 0,
    PSE_OUTER_ADD_MUL_TYPE, // default
    PSE_INNER_MUL_ADD_TYPE,
    PSE_INNER_MUL_ADD_SQRT_TYPE,
    PSE_INVALID_TYPE
};

enum class AttenDataType : uint32_t {
    ATTEN_MASK_TYPE_SAME = 0,   // 0 表示 AttenMask 数据类型与 qkv 一致
    ATTEN_MASK_TYPE_U8_BOOL = 1 // 1 表示 AttenMask 数据类型为 u8 bool
};

enum class AttenShapeType : uint32_t {
    ATTEN_MASK_SHAPE_TYPE_SS,
    ATTEN_MASK_SHAPE_TYPE_B1SS,
    ATTEN_MASK_SHAPE_TYPE_BNSS
};

enum class SparseMode : uint32_t {
    NO_MASK = 0, // 未传入attenmask，不做mask操作
    ALL_MASK,
    LEFT_UP_CAUSAL,        // 左上角点划分的三角部分
    RIGHT_DOWN_CAUSAL = 3, // 右下角点划分的下三角部分
    BAND = 4,
    PREFIX = 5,
    PREFIX_COMPRESS = 6,
    RIGHT_DOWN_CASUAL_BAND = 7,
    BAND_LEFT_UP_CASUAL = 8
};

constexpr uint32_t ATTEN_MASK_SHAPE_TEMP_DIMS = 0; // 0 是 B1SS 及 SS差异轴索引, S不可能为 1

struct TempParams { // 频繁使用的中间态临时变量
    uint32_t usedUBSize;
    uint32_t tilingKey;
    uint32_t apiClcQueueSize = 0;
};

inline int64_t CeilCommon(int64_t num1, int64_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

inline int64_t Align(const int64_t n)
{
    return (n + BYTE_PER_BLOCK - 1) & (~(BYTE_PER_BLOCK - 1));
}

inline uint32_t AlignData(const uint32_t a, const uint32_t b)
{
    if (b == static_cast<uint32_t>(0)) {
        return a;
    }
    return (a + b - static_cast<uint32_t>(1)) / b * b;
}

template <class T>
inline auto AlignTo(const T n, const T alignSize) -> T
{
    if (alignSize == 0) {
        return 0;
    }
    return (n + alignSize - 1) & (~(alignSize - 1));
}

template <typename T>
auto AlignUp(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

inline int64_t AbsCeil(int64_t num1, int64_t num2)
{ 
    bool isNegative = (num1 < 0) || (num2 < 0);
    int64_t result = (std::abs(num1) + std::abs(num2) - 1) / std::abs(num2);
    return isNegative ? -result : result;
}
 
inline int64_t Gcd(int64_t a, int64_t b)
{
    int64_t r;
    while (b > 0) {
        r = a % b;
        a = b;
        b = r;
    }
    return a;
}

ge::graphStatus CheckSoftmaxMaxShape(gert::TilingContext *context, int64_t b, int64_t n1, int64_t s1, bool isQuant);
ge::graphStatus CheckSoftmaxSumShape(gert::TilingContext *context, int64_t b, int64_t n1, int64_t s1, bool isQuant);
ge::graphStatus CheckAttentionInShape(gert::TilingContext *context);
ge::graphStatus CheckShapeValid(gert::TilingContext *context, int64_t b, int64_t n1, int64_t s1, int64_t d);
ge::graphStatus CheckTndShapeValid(gert::TilingContext *context, int64_t t1, int64_t n1, int64_t d);

}
} // namespace optiling
