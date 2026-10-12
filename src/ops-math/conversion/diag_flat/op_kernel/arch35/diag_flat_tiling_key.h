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
 * \file diag_flat_tiling_key.h
 * \brief DiagFlat TilingKey template parameter definition (arch35, DAV_3510)
 *
 * Single TilingKey 3501: SIMD+SIMT hybrid, covers all 13 dtypes.
 * (ref: diag_v2/op_kernel/arch35/diag_v2_tiling_key.h)
 */

#ifndef __DIAG_FLAT_ARCH35_TILING_KEY_H__
#define __DIAG_FLAT_ARCH35_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(DiagFlat,
    ASCENDC_TPL_UINT_DECL(ARCH35_KEY, 32, ASCENDC_TPL_UI_LIST, 3501)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(ARCH35_KEY, ASCENDC_TPL_UI_LIST, 3501)
    )
);

#endif // __DIAG_FLAT_ARCH35_TILING_KEY_H__
