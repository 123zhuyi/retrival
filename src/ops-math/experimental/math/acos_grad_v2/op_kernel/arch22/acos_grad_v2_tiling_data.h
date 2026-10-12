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
 * \file acos_grad_v2_tiling_data.h
 * \brief AcosGradV2 Tiling Data 结构（arch22 / Ascend910B）
 */

#ifndef ACOS_GRAD_V2_TILING_DATA_H
#define ACOS_GRAD_V2_TILING_DATA_H

#include <cstdint>

struct AcosGradV2TilingData {
    uint64_t totalLength;
    uint32_t blockFormer;
    uint32_t blockNum;
    uint32_t ubFormer;
    // 核内 loop/tail 由 kernel 侧按 blockLength_ 自行推导，无需在 tiling 预计算
};

#endif // ACOS_GRAD_V2_TILING_DATA_H
