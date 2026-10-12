/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_TILING_KEY_H
#define GROUPED_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_RINT 4
#define TPL_SCALE_ALG_1 1

namespace GroupedDynamicMxQuantWithDualAxisOp {
ASCENDC_TPL_ARGS_DECL(GroupedDynamicMxQuantWithDualAxis,
                      ASCENDC_TPL_UINT_DECL(roundMode, 1, ASCENDC_TPL_UI_LIST, TPL_RINT),
                      ASCENDC_TPL_UINT_DECL(scaleAlg, 1, ASCENDC_TPL_UI_LIST, TPL_SCALE_ALG_1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(roundMode, ASCENDC_TPL_UI_LIST, TPL_RINT),
                                     ASCENDC_TPL_UINT_SEL(scaleAlg, ASCENDC_TPL_UI_LIST, TPL_SCALE_ALG_1)));
} // namespace GroupedDynamicMxQuantWithDualAxisOp

#endif // GROUPED_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_TILING_KEY_H
