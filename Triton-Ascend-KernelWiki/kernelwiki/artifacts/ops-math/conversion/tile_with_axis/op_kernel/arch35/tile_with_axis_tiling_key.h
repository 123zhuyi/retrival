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
 * \file tile_with_axis_tiling_key.h
 * \brief TileWithAxis TilingKey 模板参数定义
 *
 * TilingKey: UB_AXIS (0=outerDim, 1=tiles, 2=rowLength)，共 3 个.
 * dtype 由框架自动注入 DTYPE_X，不纳入 TilingKey.
 * (ref: pad_v3_grad_replication 同模式)
 */

#ifndef __TILE_WITH_AXIS_TILING_KEY_H__
#define __TILE_WITH_AXIS_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(TileWithAxis,
    ASCENDC_TPL_UINT_DECL(UB_AXIS, 8, ASCENDC_TPL_UI_LIST, 0, 1, 2)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB_AXIS, ASCENDC_TPL_UI_LIST, 0, 1, 2)
    )
);

#endif // __TILE_WITH_AXIS_TILING_KEY_H__
