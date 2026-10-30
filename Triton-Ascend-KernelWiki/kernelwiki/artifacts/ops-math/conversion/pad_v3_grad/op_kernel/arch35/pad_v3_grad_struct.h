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
 * \file pad_v3_grad_struct.h
 * \brief pad v3 grad struct
 */
#ifndef PAD_V3_GRAD_STRUCT_H_
#define PAD_V3_GRAD_STRUCT_H_

constexpr uint64_t PAD_GRAD_MAX_DIMS_NUM = 8;

struct PadV3GradACTilingData {
    uint8_t dimNum;
    uint8_t ubAxis;
    uint32_t ubFactor;
    uint32_t ubPerCount;
    uint32_t ubTotalCount;
    uint32_t outTileSize; // 元素个数
    uint32_t additionTileSize;

    uint64_t inShape[PAD_GRAD_MAX_DIMS_NUM];
    uint64_t outShape[PAD_GRAD_MAX_DIMS_NUM];
    uint64_t inStride[PAD_GRAD_MAX_DIMS_NUM];
    uint64_t outStride[PAD_GRAD_MAX_DIMS_NUM];
    int64_t leftPad[PAD_GRAD_MAX_DIMS_NUM];
    int64_t rightPad[PAD_GRAD_MAX_DIMS_NUM];
};

#endif // PAD_V3_GRAD_STRUCT_H_
