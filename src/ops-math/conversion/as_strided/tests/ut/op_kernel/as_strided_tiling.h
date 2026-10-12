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
 * \file as_strided_tiling.h
 * \brief
 */

#ifndef __AS_STRIDED_TILING_H__
#define __AS_STRIDED_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct AsStridedTilingData {
  int64_t storageOffset = 0;
  uint32_t blockNum = 0;
  uint32_t loopsTailCore = 0;
  uint32_t tilingAxisIdx = 0;
  uint32_t outerAxisFactor = 0;
  uint32_t innerAxisFactor = 0;
  uint32_t outerAxisNum = 0;
  uint32_t innerAxisNum = 0;
  uint32_t loopsPerCore = 0;
  uint32_t ubFactor = 0;
  uint32_t ubFactorTail = 0;
  uint32_t ubSize = 0;
  uint32_t innerAxisFactorTail = 0;
  uint32_t axisOutTotalFactor = 0;
  int32_t stride_arr[10] = {};
  int32_t size_arr[10] = {};
  int32_t inner_axis[10] = {};
  int32_t outStrideArr[10] = {};
  int32_t outLoopArr[10] = {};
  uint32_t nddmaLoop[5] = {};
  uint32_t nddmaTailLoop[5] = {};
  uint64_t nddmaSrcStride[5] = {};
  uint32_t nddmaDstStride[5] = {};
};
#pragma pack()

#ifdef __NPU_TILING__
inline [aicore] void InitTilingData(const __gm__ uint8_t* tiling, AsStridedTilingData* const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(AsStridedTilingData) / 4; i++) *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, AsStridedTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(AsStridedTilingData));
}
#endif


#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
tiling_struct tiling_data; \
InitTilingData(tiling_arg, &tiling_data)


#define GET_TILING_DATA(tiling_data, tiling_arg) \
AsStridedTilingData tiling_data; \
InitTilingData(tiling_arg, &tiling_data)

#define DTYPE_X int32_t
#define DTYPE_INDICES int32_t

#endif