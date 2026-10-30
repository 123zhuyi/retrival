/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file view_copy.cpp
 * \brief ViewCopy kernel entry.
 */

#include "view_copy.h"
#include "view_copy_tiling_key.h"

template <uint32_t schMode>
__global__ __aicore__ void view_copy(GM_ADDR dst, GM_ADDR dst_size, GM_ADDR dst_stride, GM_ADDR dst_storage_offset,
                                     GM_ADDR src, GM_ADDR src_size, GM_ADDR src_stride, GM_ADDR src_storage_offset,
                                     GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(ViewCopyTilingData);
    GET_TILING_DATA_WITH_STRUCT(ViewCopyTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if constexpr (schMode == VIEWCOPY_TPL_SCH_MODE_0) {
        NsViewCopy::ViewCopy<int8_t> op;
        op.Init(dst, dst_size, dst_stride, dst_storage_offset, src, src_size, src_stride, src_storage_offset, y,
                &tilingData);
        op.Process();
    } else if constexpr (schMode == VIEWCOPY_TPL_SCH_MODE_1) {
        NsViewCopy::ViewCopy<int16_t> op;
        op.Init(dst, dst_size, dst_stride, dst_storage_offset, src, src_size, src_stride, src_storage_offset, y,
                &tilingData);
        op.Process();
    } else if constexpr (schMode == VIEWCOPY_TPL_SCH_MODE_2) {
        NsViewCopy::ViewCopy<int32_t> op;
        op.Init(dst, dst_size, dst_stride, dst_storage_offset, src, src_size, src_stride, src_storage_offset, y,
                &tilingData);
        op.Process();
    } else if constexpr (schMode == VIEWCOPY_TPL_SCH_MODE_3) {
        NsViewCopy::ViewCopy<int64_t> op;
        op.Init(dst, dst_size, dst_stride, dst_storage_offset, src, src_size, src_stride, src_storage_offset, y,
                &tilingData);
        op.Process();
    }
}
