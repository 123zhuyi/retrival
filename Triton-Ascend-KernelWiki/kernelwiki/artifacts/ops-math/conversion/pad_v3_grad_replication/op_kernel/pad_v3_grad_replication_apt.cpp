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
 * \file pad_v3_grad_replication.cpp
 * \brief pad_v3_grad_replication kernel entry (dispatcher)
 */

#include "arch35/pad_v3_grad_replication_struct.h"
#include "arch35/pad_v3_grad_replication_tilingkey.h"
#include "arch35/pad_v3_grad_replication.h"
#include "arch35/pad_v3_grad_replication_edge_simt.h"

#include <type_traits>

#include "kernel_operator.h"

using namespace AscendC;
using namespace PadV3GradReplication;

// 工厂：根据 (DimNum, SplitAxis) 模板实例化具体 kernel
template <uint8_t DimNum, uint8_t SplitAxis>
__aicore__ inline void LaunchKernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (SplitAxis < DimNum) {
        GET_TILING_DATA_WITH_STRUCT(PadV3GradReplicationTilingData, tilingData, tiling);

        // 切尾轴：纯 SIMT 路径，直接从 GM 读写，不需要 UB buffer
        if constexpr (SplitAxis == DimNum - 1) {
            if constexpr (std::is_same_v<DTYPE_X, half> || std::is_same_v<DTYPE_X, bfloat16_t> ||
                          std::is_same_v<DTYPE_X, float>) {
                PadV3GradReplicationEdgeSimt<DTYPE_X> op;
                op.Init(x, y, &tilingData);
                op.Process<int64_t>();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(uint8_t)) {
                PadV3GradReplicationEdgeSimt<uint8_t> op;
                op.Init(x, y, &tilingData);
                op.Process<int64_t>();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(uint16_t)) {
                PadV3GradReplicationEdgeSimt<uint16_t> op;
                op.Init(x, y, &tilingData);
                op.Process<int64_t>();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(uint32_t)) {
                PadV3GradReplicationEdgeSimt<uint32_t> op;
                op.Init(x, y, &tilingData);
                op.Process<int64_t>();
            } else if constexpr (sizeof(DTYPE_X) == sizeof(uint64_t)) {
                PadV3GradReplicationEdgeSimt<uint64_t> op;
                op.Init(x, y, &tilingData);
                op.Process<int64_t>();
            }
            return;
        }

        // 非切尾轴：原矩形搬移 + 向量累加 kernel
        TPipe pipe;
        if constexpr (std::is_same_v<DTYPE_X, half> || std::is_same_v<DTYPE_X, bfloat16_t> ||
                      std::is_same_v<DTYPE_X, float>) {
            KernelPadV3GradReplication<DTYPE_X, DimNum, SplitAxis> op(&pipe, &tilingData);
            op.Init(x, y);
            op.Process();
        } else if constexpr (sizeof(DTYPE_X) == sizeof(uint8_t)) {
            KernelPadV3GradReplication<uint8_t, DimNum, SplitAxis> op(&pipe, &tilingData);
            op.Init(x, y);
            op.Process();
        } else if constexpr (sizeof(DTYPE_X) == sizeof(uint16_t)) {
            KernelPadV3GradReplication<uint16_t, DimNum, SplitAxis> op(&pipe, &tilingData);
            op.Init(x, y);
            op.Process();
        } else if constexpr (sizeof(DTYPE_X) == sizeof(uint32_t)) {
            KernelPadV3GradReplication<uint32_t, DimNum, SplitAxis> op(&pipe, &tilingData);
            op.Init(x, y);
            op.Process();
        } else if constexpr (sizeof(DTYPE_X) == sizeof(uint64_t)) {
            KernelPadV3GradReplication<uint64_t, DimNum, SplitAxis> op(&pipe, &tilingData);
            op.Init(x, y);
            op.Process();
        }
    }
}

template <uint8_t DimNum, uint8_t SplitAxis>
__global__ __aicore__ void pad_v3_grad_replication(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace,
                                                   GM_ADDR tiling)
{
    SetSysWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(PadV3GradReplicationTilingData);

    LaunchKernel<DimNum, SplitAxis>(x, y, workspace, tiling);
}
