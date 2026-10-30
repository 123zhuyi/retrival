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
 * \file pad_v3_grad_replication_struct.h
 * \brief pad v3 grad replication tiling data structure
 */

#ifndef PAD_V3_GRAD_REPLICATION_STRUCT_H_
#define PAD_V3_GRAD_REPLICATION_STRUCT_H_

constexpr uint64_t PAD_GRAD_REPLICATION_MAX_DIMS_NUM = 8;
constexpr uint32_t PAD_GRAD_REPLICATION_MAX_PAD_DIMS_NUM = 5; // 后5维可做padding的最大维度数

struct PadV3GradReplicationTilingData {
    uint8_t dimNum;
    uint8_t splitAxis;
    uint32_t splitCount;
    uint32_t splitSize;
    uint32_t usedCoreNum;
    uint32_t tilesPerCore;

    uint64_t inputShape[PAD_GRAD_REPLICATION_MAX_DIMS_NUM];    // 原始tensor（不含padding）
    uint64_t outputShape[PAD_GRAD_REPLICATION_MAX_DIMS_NUM];   // padding后tensor（含padding）
    uint64_t strideAligned[PAD_GRAD_REPLICATION_MAX_DIMS_NUM]; // UB对齐stride
    int64_t leftPad[PAD_GRAD_REPLICATION_MAX_DIMS_NUM];  // 所有维度左padding（实际运行时前N-5维为0）
    int64_t rightPad[PAD_GRAD_REPLICATION_MAX_DIMS_NUM]; // 所有维度右padding（实际运行时前N-5维为0）
};

#endif // PAD_V3_GRAD_REPLICATION_STRUCT_H_
