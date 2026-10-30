/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file is_finite_struct.h
 * \brief
 */

#ifndef IS_FINITE_STRUCT_H
#define IS_FINITE_STRUCT_H

#include "ascendc/host_api/tiling/template_argument.h"

namespace IsFiniteNs {

#define IS_FINITE_TPL_BOOL 10
#define IS_FINITE_TPL_FP16 20
#define IS_FINITE_TPL_FP32 30
#define IS_FINITE_TPL_BF16 40

ASCENDC_TPL_ARGS_DECL(IsFinite,
    ASCENDC_TPL_DTYPE_DECL(D_T_X, IS_FINITE_TPL_FP16, IS_FINITE_TPL_BF16, IS_FINITE_TPL_FP32),
    ASCENDC_TPL_DTYPE_DECL(D_T_Y, IS_FINITE_TPL_BOOL),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
     ASCENDC_TPL_DTYPE_SEL(D_T_X, IS_FINITE_TPL_FP16),
     ASCENDC_TPL_DTYPE_SEL(D_T_Y, IS_FINITE_TPL_BOOL),
    ),
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    ASCENDC_TPL_ARGS_SEL(
     ASCENDC_TPL_DTYPE_SEL(D_T_X, IS_FINITE_TPL_BF16),
     ASCENDC_TPL_DTYPE_SEL(D_T_Y, IS_FINITE_TPL_BOOL),
    ),
#endif
    ASCENDC_TPL_ARGS_SEL(
     ASCENDC_TPL_DTYPE_SEL(D_T_X, IS_FINITE_TPL_FP32),
     ASCENDC_TPL_DTYPE_SEL(D_T_Y, IS_FINITE_TPL_BOOL),
    )
);

struct IsFiniteTilingData {
  uint32_t usableUbSize;
  uint32_t needCoreNum;
  uint64_t totalDataCount;
  uint64_t perCoreDataCount;
  uint64_t tailDataCoreNum;
  uint64_t lastCoreDataCount;
};

}
#endif // IS_FINITE_STRUCT_H