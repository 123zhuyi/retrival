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
 * \file log_space_tiling_data.h
 * \brief LogSpace TilingData 结构体定义
 */
#ifndef _LOG_SPACE_TILING_DATA_H_
#define _LOG_SPACE_TILING_DATA_H_

struct LogSpaceTilingData {
    uint64_t totalLen = 0;    // steps
    uint32_t coreNum = 1;     // 参与计算的 AI Core 数
    uint32_t tileLen = 0;     // 每个常规核处理的元素数
    uint32_t tailCoreIdx = 0; // 处理尾块的核 id
    uint32_t tailTileLen = 0; // 尾核处理的元素数
    uint32_t ubChunk = 0;     // 单次搬运粒度（fp32 元素数）
    uint32_t reserved = 0;    // 对齐占位（保证 float 字段 4B 对齐）
    float startF = 0.0f;
    float stepF = 0.0f;     // (end-start)/(steps-1), steps==1 时为 0
    float logBase = 0.0f;   // ln(base)
    float startValF = 0.0f; // host 用 double pow 算好的 base^start，供 kernel 端点精确覆写
    float endValF = 0.0f;   // host 用 double pow 算好的 base^end
    // 整型路径用 double-float(fp32 pair)在 arg 空间(arg = x*ln(base))算准指数：arg = argStart + g*stepLn。
    // host 用 double 算 argStart=start*ln(base)、stepLn=step*ln(base)，各拆成 hi+lo 两个 fp32。
    float argStartHi = 0.0f;
    float argStartLo = 0.0f;
    float stepLnHi = 0.0f;
    float stepLnLo = 0.0f;
    // 整数幂精确覆写表：落在整数指数(网格点)的元素，df 的 exp 会下溢到整数下方(如 10^1->9.9999,
    // 整型 TRUNC 成 9)。host 用 double pow 精确算 base^m 经 tiling 下发，kernel 对这些元素 SetValue 覆写。
    float stepLoX = 0.0f;     // x 空间 step 的 double-float 低位（定位网格点用，大 case 需高精度）
    int32_t nmin = 0;         // 幂表起始指数 m
    int32_t nCount = 0;       // 幂表条目数（<=96，0 表示禁用整数幂覆写）
    float baseN[96] = {0.0f}; // baseN[k] = (float)pow(base, nmin+k)
    // [df-V] 大值 int8/uint8：用向量单元的 double-float exp 算高精度参考 expBase（标量精度不够）。
    int32_t useDfV = 0;      // 1=启用 df-V（仅 int8/uint8 且值 ∈ (2^24,2^32)）
    float rfHi[12] = {0.0f}; // 1/k! 的 fp32 高位（k=0..11，向量泰勒 exp 用）
    float rfLo[12] = {0.0f}; // 1/k! 的 double-float 低位
    // df-V 递推：expBase[c]=expBase[c-1]·const，const=base^(ubChunk·step) → df-exp 整核只算 1 次
    float constHi = 0.0f;
    float constLo = 0.0f;
};
#endif
