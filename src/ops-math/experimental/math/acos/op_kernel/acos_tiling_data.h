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
 * \file acos_tiling_data.h
 * \brief
 */

#ifndef _ROTARY_POSITION_EMBEDDING_GRAD_TILING_DATA_H_
#define _ROTARY_POSITION_EMBEDDING_GRAD_TILING_DATA_H_

struct AcosTilingData {
    uint64_t formerCoreNum = 0;
    uint64_t tailCoreNum = 0;
    uint64_t formerCoreDataNum = 0;
    uint64_t tailCoreDataNum = 0;
    uint64_t formerCoreLoopCount = 0;
    uint64_t formerCoreFormerDataNum = 0;
    uint64_t formerCoreTailDataNum = 0;
    uint64_t tailCoreLoopCount = 0;
    uint64_t tailCoreFormerDataNum = 0;
    uint64_t tailCoreTailDataNum = 0;
};
#endif
