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
 * \file grouped_bias_add_grad_tiling_def.h
 * \brief
 */

#ifndef GROUPED_BIAS_ADD_GRAD_TILING_DEF_H
#define GROUPED_BIAS_ADD_GRAD_TILING_DEF_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GroupedBiasAddGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);          // 使用的核数
TILING_DATA_FIELD_DEF(uint32_t, normalCoreNum);        // 整核个数
TILING_DATA_FIELD_DEF(uint32_t, normalCoreProcessNum); // 整核处理的GH大小
TILING_DATA_FIELD_DEF(uint32_t, tailCoreProcessNum);   // 尾核处理的GH大小
TILING_DATA_FIELD_DEF(int64_t, wsUnitNum);             // workspace上每个核存放ub累加结果个数
TILING_DATA_FIELD_DEF(int64_t, dimG);
TILING_DATA_FIELD_DEF(int64_t, dimC);
TILING_DATA_FIELD_DEF(int64_t, dimH);
TILING_DATA_FIELD_DEF(int64_t, dimGB);
TILING_DATA_FIELD_DEF(uint32_t, baseH);    // 单次ub处理的H方向个数
TILING_DATA_FIELD_DEF(uint32_t, baseC);    // 单次ub处理的C方向个数
TILING_DATA_FIELD_DEF(uint32_t, loopCNum); // 每个核的ub循环次数
TILING_DATA_FIELD_DEF(int32_t, groupIdxType);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupedBiasAddGrad, GroupedBiasAddGradTilingData)
} // namespace optiling
#endif // GROUPED_BIAS_ADD_GRAD_TILING_DEF_H