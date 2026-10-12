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
 * \file kl_div_v2_tiling_data.h
 * \brief kl div v2 tiling data
 */

#ifndef KL_DIV_V2_TILING_DATA_H
#define KL_DIV_V2_TILING_DATA_H

#include "atvoss/reduce/reduce_tiling_data.h"
struct KLDivV2TilingData {
    float emptyValue; // 空Tensor时输出值
    Ops::Base::ReduceOpTilingData reduceTiling;
};
#endif