/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _FAST_OP_TEST_HISTOGRAM_V2_H_
#define _FAST_OP_TEST_HISTOGRAM_V2_H_

#include "kernel_tiling/kernel_tiling.h"
struct HistogramV2TilingData {
  int64_t bins;
  int64_t ubBinsLength;

  int64_t formerNum;
  int64_t formerLength;
  int64_t formerLengthAligned;
  int64_t tailLength;
  int64_t tailLengthAligned;

  int64_t formerTileNum;
  int64_t formerTileDataLength;
  int64_t formerTileLeftDataLength;
  int64_t formerTileLeftDataLengthAligned;

  int64_t tailTileNum;
  int64_t tailTileDataLength;
  int64_t tailTileLeftDataLength;
  int64_t tailTileLeftDataLengthAligned;
};

#define DTYPE_X int64_t

#pragma pack(1)

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  __ubuf__ tilingStruct* tilingDataPointer =                                \
      reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                                        \
  HistogramV2TilingData tilingData;                                                                         \
  INIT_TILING_DATA(HistogramV2TilingData, tilingDataPointer, tilingPointer);                                \
  (tilingData).bins = tilingDataPointer->bins;                                                            \
  (tilingData).ubBinsLength = tilingDataPointer->ubBinsLength;                                              \
  (tilingData).formerNum = tilingDataPointer->formerNum;                                                  \
  (tilingData).formerLength = tilingDataPointer->formerLength;                                            \     
  (tilingData).formerLengthAligned = tilingDataPointer->formerLengthAligned;                              \
  (tilingData).tailLength = tilingDataPointer->tailLength;                                                \     
  (tilingData).tailLengthAligned = tilingDataPointer->tailLengthAligned;                                  \
  (tilingData).formerTileNum = tilingDataPointer->formerTileNum;                                          \
  (tilingData).formerTileDataLength = tilingDataPointer->formerTileDataLength;                            \
  (tilingData).formerTileLeftDataLength = tilingDataPointer->formerTileLeftDataLength;                    \
  (tilingData).formerTileLeftDataLengthAligned = tilingDataPointer->formerTileLeftDataLengthAligned;      \
  (tilingData).tailTileNum = tilingDataPointer->tailTileNum;                                              \     
  (tilingData).tailTileDataLength = tilingDataPointer->tailTileDataLength;                                \
  (tilingData).tailTileLeftDataLength = tilingDataPointer->tailTileLeftDataLength;                        \     
  (tilingData).tailTileLeftDataLengthAligned = tilingDataPointer->tailTileLeftDataLengthAligned;
#endif // _FAST_OP_TEST_HISTOGRAM_V2_H_
