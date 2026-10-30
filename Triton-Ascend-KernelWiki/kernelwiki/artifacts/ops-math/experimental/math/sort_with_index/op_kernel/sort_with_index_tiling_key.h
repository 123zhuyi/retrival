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
 * \file sort_with_index_tiling_key.h
 * \brief SortWithIndex TilingKey template-argument declaration.
 *
 * The logical TilingKey is 3-D: (VALUE_DT, INDEX_DT, SIZE_MODE).
 *   - VALUE_DT / INDEX_DT are dispatched at compile time through the auto-generated DTYPE_X /
 *     DTYPE_INDEX macros (driven by the op_host DataType list), NOT through ASCENDC_TPL.
 *   - SIZE_MODE is dispatched through the ASCENDC_TPL "schMode" key declared below.
 *
 * SIZE_MODE values:
 *   0 = SINGLE_TILE           (single-tile in-row sort; main path)
 *   1 = MULTI_TILE_MRGSORT    (large-axis core-internal MrgSort; implemented)
 *   2 = EMPTY                 (empty tensor / N==0 / rowNum==0; implemented)
 */

#ifndef SORT_WITH_INDEX_TILING_KEY_H
#define SORT_WITH_INDEX_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define SORT_WITH_INDEX_SIZE_MODE_SINGLE 0
#define SORT_WITH_INDEX_SIZE_MODE_MRGSORT 1
#define SORT_WITH_INDEX_SIZE_MODE_EMPTY 2

// bitWidth must satisfy 2^bitWidth >= number_of_values. 3 SIZE_MODE values -> bitWidth = 2.
ASCENDC_TPL_ARGS_DECL(SortWithIndex,
    ASCENDC_TPL_UINT_DECL(schMode, 2,
        ASCENDC_TPL_UI_LIST,
        SORT_WITH_INDEX_SIZE_MODE_SINGLE,
        SORT_WITH_INDEX_SIZE_MODE_MRGSORT,
        SORT_WITH_INDEX_SIZE_MODE_EMPTY));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode,
            ASCENDC_TPL_UI_LIST,
            SORT_WITH_INDEX_SIZE_MODE_SINGLE,
            SORT_WITH_INDEX_SIZE_MODE_MRGSORT,
            SORT_WITH_INDEX_SIZE_MODE_EMPTY)));

#endif  // SORT_WITH_INDEX_TILING_KEY_H
