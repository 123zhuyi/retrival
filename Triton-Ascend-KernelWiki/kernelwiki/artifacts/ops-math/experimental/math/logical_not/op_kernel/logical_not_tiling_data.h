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
 * \file logical_not_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __LOGICAL_NOT_TILLING_DATA_H__
#define __LOGICAL_NOT_TILLING_DATA_H__

struct LogicalNotTilingData {
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t bigCoreLoopNum;
    uint32_t smallCoreLoopNum;
    uint32_t ubPartDataNum;
    uint32_t smallCoreTailDataNum;
    uint32_t bigCoreTailDataNum;
    uint32_t tailBlockNum;
};
#endif
