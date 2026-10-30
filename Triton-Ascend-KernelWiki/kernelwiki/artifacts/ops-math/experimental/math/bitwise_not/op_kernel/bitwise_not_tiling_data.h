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
 * \file bitwise_not_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __BITWISE_NOT_TILING_DATA_H__
#define __BITWISE_NOT_TILING_DATA_H__

struct BitwiseNotTilingData {
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t bigCoreLoopNum;
    uint32_t smallCoreLoopNum;
    uint32_t ubPartDataNum;
    uint32_t smallCoreTailDataNum;
    uint32_t bigCoreTailDataNum;
    uint32_t tailBlockNum;
    // BOOL 与 INT8 在 kernel 承载类型同为 int8_t；op_host 据 self.dtype==DT_BOOL 置 1，
    // kernel 据此选「逻辑非」分支，否则走「按位取反」分支，杜绝 int8/bool 二义性。
    uint32_t isBool;
    // ↓↓↓ 尾块对齐处理：32B 块对齐切分使各核名义元素数之和 = alignedTotal
    //     （= ceil(inputLength,32)/dataTypeLength），最多比真实 inputDataNum 多 pad(<1 block) 个填充元素。
    //     该 pad 只落在 GM 序最后一个核（必为最后一个 small core），故仅末核最后一个 tile 用真实剩余元素数
    //     lastCoreTailDataNum（= smallCoreTailDataNum - pad，可非 32B 对齐 → DataCopyPad）写回 GM，
    //     避免末核越界写 / 非末核漏写。
    uint32_t lastCoreId;          // GM 序最后一个核的 coreId（= coreNum-1）
    uint32_t lastCoreTailDataNum; // 末核最后一个 tile 的真实剩余元素数（原 dtype 元素数，可非 32B 对齐）
};
#endif
