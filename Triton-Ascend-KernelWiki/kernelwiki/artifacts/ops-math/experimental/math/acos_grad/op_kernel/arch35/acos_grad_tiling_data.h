/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file acos_grad_tiling_data.h
 * \brief AcosGrad Tiling Data Structure
 */

#ifndef ACOS_GRAD_TILING_DATA_H
#define ACOS_GRAD_TILING_DATA_H

#include <cstdint>

struct AcosGradTilingData {
    uint64_t totalLength;
    uint32_t blockFormer;
    uint32_t blockNum;
    uint32_t ubFormer;
    uint32_t ubLoopOfFormerBlock;
    uint32_t ubTailOfFormerBlock;
    uint32_t ubLoopOfTailBlock;
    uint32_t ubTailOfTailBlock;
};

#endif // ACOS_GRAD_TILING_DATA_H
