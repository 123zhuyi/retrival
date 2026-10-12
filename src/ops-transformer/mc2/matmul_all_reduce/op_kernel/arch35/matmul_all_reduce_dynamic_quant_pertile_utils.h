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
 * \file matmul_all_reduce_dynamic_quant_pertile_utils.h
 * \brief
 */

#ifndef MATMUL_ALL_REDUCE_DYNAMIC_QUANT_PERTILE_UTILS_H
#define MATMUL_ALL_REDUCE_DYNAMIC_QUANT_PERTILE_UTILS_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/pad/broadcast.h"
#include "adv_api/reduce/reduce.h"

namespace AscendC {

constexpr uint32_t TILELEN = 128;
constexpr uint32_t BROADCAST_DIM = 2;
constexpr uint32_t UB_DATABLOCK = 32;
constexpr uint32_t COMPARE_ALIGN_LEN = 256;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3_MAX_VALUE = 448.0f;

template <class T>
__aicore__ inline void DynamicQuant(uint32_t curScaleCnt, uint32_t padCalCnt, LocalTensor<float> tilesLocal,
                                    LocalTensor<T> curOutTiles, LocalTensor<float> curScale,
                                    LocalTensor<float> tempWorkTiles, LocalTensor<float> tempScale,
                                    LocalTensor<uint8_t> tempMskBuf)
{
    float xTypeMax = std::is_same<T, fp8_e4m3fn_t>::value ? FP8_E4M3_MAX_VALUE : FP8_E5M2_MAX_VALUE;
    const uint32_t broadCastDst[BROADCAST_DIM] = {curScaleCnt, static_cast<uint32_t>(TILELEN)};
    const uint32_t broadCastSrc[BROADCAST_DIM] = {curScaleCnt, 1};
    uint32_t compareCnt = (Ceil(curScaleCnt * sizeof(float), COMPARE_ALIGN_LEN) * COMPARE_ALIGN_LEN) / sizeof(float);
    Abs(tempWorkTiles, tilesLocal, padCalCnt);
    Duplicate(tempScale, 0.0f, padCalCnt);
    ReduceMax<float, AscendC::Pattern::Reduce::AR, true>(tempScale, tempWorkTiles, broadCastDst, false);
    CompareScalar(tempMskBuf, tempScale, 0.0f, AscendC::CMPMODE::NE, compareCnt);
    // 量化参数为0的位置不做量化
    Select(tempScale, tempMskBuf, tempScale, xTypeMax, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, curScaleCnt);
    Duplicate(tempWorkTiles, xTypeMax, curScaleCnt);
    Div(curScale, tempWorkTiles, tempScale, curScaleCnt);
    CompareScalar(tempMskBuf, curScale, 0.0f, AscendC::CMPMODE::NE, compareCnt);
    Select(curScale, tempMskBuf, curScale, 1.0f, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, curScaleCnt);
    // Broadcast到{curScaleCnt, 128}，随后Div一次算出所有量化后的数据
    Broadcast<float, BROADCAST_DIM, 1, false>(tempScale, curScale, broadCastDst, broadCastSrc);
    Mul(tilesLocal, tilesLocal, tempScale, padCalCnt);
    Cast(curOutTiles, tilesLocal, RoundMode::CAST_RINT, padCalCnt);
}

template <class T, class U>
__aicore__ inline void DynamicDequant(uint32_t curScaleCnt, uint32_t padCalCnt, LocalTensor<U> outLocal,
                                      LocalTensor<T> tilesLocal, LocalTensor<float> scalesLocal,
                                      LocalTensor<float> tempOut, LocalTensor<float> tempScale)
{
    const uint32_t broadCastDst[BROADCAST_DIM] = {curScaleCnt, static_cast<uint32_t>(TILELEN)};
    const uint32_t broadCastSrc[BROADCAST_DIM] = {curScaleCnt, 1};
    Cast(tempOut, tilesLocal, RoundMode::CAST_NONE, padCalCnt);
    Broadcast<float, BROADCAST_DIM, 1, false>(tempScale, scalesLocal, broadCastDst, broadCastSrc);
    if constexpr (!std::is_same<U, float>::value) {
        Div(tempOut, tempOut, tempScale, padCalCnt);
        Cast(outLocal, tempOut, RoundMode::CAST_RINT, padCalCnt);
    } else {
        Div(outLocal.template ReinterpretCast<float>(), tempOut, tempScale, padCalCnt);
    }
}

template <class XType, class YType>
__aicore__ inline uint32_t GetMaxProcRows(bool isQuant, uint32_t calBuffSize)
{
    uint32_t curUbSize = isQuant ?
                             TOTAL_UB_SIZE - calBuffSize - (COMPARE_ALIGN_LEN - 1 + UB_DATABLOCK - 1) * DOUBLE_BUFFER :
                             TOTAL_UB_SIZE - calBuffSize - (UB_DATABLOCK - 1) * DOUBLE_BUFFER;
    // 将ub空间按照各个buf所占比例切分
    uint32_t ubDenom =
        isQuant ? 3 * TILELEN * sizeof(float) + TILELEN * sizeof(XType) + sizeof(float) + sizeof(uint8_t) :
                  TILELEN * sizeof(XType) + sizeof(float) + TILELEN * sizeof(YType) + 2 * TILELEN * sizeof(float);
    ubDenom *= DOUBLE_BUFFER;
    return curUbSize / ubDenom;
}

template <class XType>
__aicore__ inline uint32_t GetMixedMaxProcRows(uint32_t calBuffSize)
{
    uint32_t curUbSize = TOTAL_UB_SIZE - calBuffSize - (COMPARE_ALIGN_LEN - 1 + UB_DATABLOCK - 1) * DOUBLE_BUFFER;
    uint32_t tilesBuffRatio = TILELEN * sizeof(XType);
    uint32_t scalesBuffRatio = sizeof(float);
    uint32_t calcBuffRatio = 3 * TILELEN * sizeof(float) + sizeof(uint8_t);
    uint32_t ubDenom = tilesBuffRatio + scalesBuffRatio + calcBuffRatio;
    ubDenom *= DOUBLE_BUFFER;
    return curUbSize / ubDenom;
}
} // MATMUL_ALL_REDUCE_DYNAMIC_QUANT_PERTILE_UTILS_H

#endif
