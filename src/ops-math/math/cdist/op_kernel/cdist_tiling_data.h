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
 * \file cdist_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __CDIST_TILLING_DATA_H__
#define __CDIST_TILLING_DATA_H__

struct CdistTilingData {
    int64_t realCoreNum = 0;
    int64_t blockFactor = 0;
    int64_t blockTailFactor = 0;
    int64_t B = 0;
    int64_t P = 0;
    int64_t R = 0;
    int64_t M = 0;
    int64_t blockMainNumB = 0;
    int64_t blockTailNumB = 0;
    int64_t blockMainFactorB = 0;
    int64_t blockTailFactorB = 0;
    int64_t blockMainNumP = 0;
    int64_t blockTailNumP = 0;
    int64_t blockMainFactorP = 0;
    int64_t blockTailFactorP = 0;
    int64_t blockMainNumR = 0;
    int64_t blockTailNumR = 0;
    int64_t blockMainFactorR = 0;
    int64_t blockTailFactorR = 0;
    int64_t ubLoopNumB = 0;
    int64_t ubFactorB = 0;
    int64_t ubTailFactorB = 0;
    int64_t ubLoopNumP = 0;
    int64_t ubFactorP = 0;
    int64_t ubTailFactorP = 0;
    int64_t ubLoopNumR = 0;
    int64_t ubFactorR = 0;
    int64_t ubTailFactorR = 0;
    int64_t ubLoopNumM = 0;
    int64_t ubFactorM = 0;
    int64_t ubTailFactorM = 0;
    float p = 0.0f;
};
#endif