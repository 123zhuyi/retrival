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
 * \file quant_grouped_matmul_dequant.cpp
 * \brief
 */

#include "quant_matmul_dequant_grouped.h"

extern "C" __global__ __aicore__ void quant_grouped_matmul_dequant(GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale, GM_ADDR group_list,
                                                           GM_ADDR bias, GM_ADDR x_scale, GM_ADDR x_offset, GM_ADDR smooth_scale,
                                                           GM_ADDR y, GM_ADDR usrworkspace, GM_ADDR qmmTiling) {
  SetAtomicNone();
  GET_TILING_DATA(tiling_data, qmmTiling);
  QuantMatmulDequantTilingData* __restrict tilingData = const_cast<QuantMatmulDequantTilingData *> (&tiling_data);
  if (TILING_KEY_IS(10000003)) {
    QuantMatmulDequantGrouped handle;
    handle.Init(x, quantized_weight, weight_scale, group_list, bias, x_scale, x_offset, smooth_scale, y, usrworkspace, tilingData);
    handle.Process();
  } else if (TILING_KEY_IS(10000004)){
    QuantMatmulDequantGrouped handle;
    handle.SetSwift();
    handle.Init(x, quantized_weight, weight_scale, group_list, bias, x_scale, x_offset, smooth_scale, y, usrworkspace, tilingData);
    handle.Process();
  }
  SetMaskNorm();
  ResetMask();
}
