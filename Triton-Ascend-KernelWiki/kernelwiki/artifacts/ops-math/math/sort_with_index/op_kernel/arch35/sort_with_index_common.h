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
 * \file sort_with_index_common.h
 * \brief sort with index common utilities
 */
#ifndef SORT_WITH_INDEX_COMMON_H
#define SORT_WITH_INDEX_COMMON_H
#include "kernel_operator.h"
#include "constant_var_simd.h"

namespace SortWithIndex {

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

template <typename T>
__aicore__ inline T SortGetMin(T left, T right)
{
    return (left > right ? right : left);
}

#ifdef TOP_K_V2_APT_H
struct SortWithIndexTilingDataSimt {
    int32_t isDescend;
    uint32_t sortLoopTimes;
    uint32_t unsortedDimParallel;
    uint64_t unsortedDimNum;
    uint32_t lastDimTileNum;
    uint32_t lastDimNeedCore;
    uint32_t numTileDataSize;
    uint32_t sortAcApiNeedBufferSize;
    uint32_t mergSortAcApiNeedBufferSize;
    uint32_t oneCoreRowNum;
    uint32_t outputLastDimValue;
    uint32_t isInInt32Range;
    int64_t lastAxisNum;
    uint32_t keyParams0;
    uint32_t keyParams1;
    uint32_t keyParams2;    
    uint32_t keyParams3;
    uint32_t keyParams4;
    uint32_t keyParams5;
    uint32_t tmpUbSize;
    uint32_t modeType;        
};
#endif

}
#endif