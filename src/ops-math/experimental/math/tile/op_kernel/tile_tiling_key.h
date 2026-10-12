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
 * \file tile_tiling_key.h
 * \brief tile tiling key declare
 */
#ifndef TILE_TILING_KEY_H_
#define TILE_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define TILE_TPL_SCH_MODE_DEFAULT 0
#define TILE_TPL_SCH_MODE_BUILDONCE 2
#define TILE_TPL_SCH_MODE_DMABUILD 3
#define TILE_TPL_SCH_MODE_READBACK 4
#define TILE_TPL_SCH_MODE_VECGATHER 5

ASCENDC_TPL_ARGS_DECL(
    Tile, ASCENDC_TPL_UINT_DECL(
              schMode, 3, ASCENDC_TPL_UI_LIST, TILE_TPL_SCH_MODE_DEFAULT, TILE_TPL_SCH_MODE_BUILDONCE,
              TILE_TPL_SCH_MODE_DMABUILD, TILE_TPL_SCH_MODE_READBACK, TILE_TPL_SCH_MODE_VECGATHER));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(
        schMode, ASCENDC_TPL_UI_LIST, TILE_TPL_SCH_MODE_DEFAULT, TILE_TPL_SCH_MODE_BUILDONCE,
        TILE_TPL_SCH_MODE_DMABUILD, TILE_TPL_SCH_MODE_READBACK, TILE_TPL_SCH_MODE_VECGATHER)), );

#endif
