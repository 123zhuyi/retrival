/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file cdist_grad_tiling_key.h
 * \brief CdistGrad TilingKey template argument definitions
 *
 * Template parameters:
 *   D_T: Input data type (float, half)
 *   P_MODE: p value mode (0=p1, 1=p2, 2=pinf, 3=general)
 *   SCH_MODE: Schedule mode (0=FullM, 1=SplitM reserved)
 */
#ifndef __CDIST_GRAD_TILING_KEY_H__
#define __CDIST_GRAD_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(
    CdistGrad, ASCENDC_TPL_DATATYPE_DECL(D_T, C_DT_FLOAT, C_DT_FLOAT16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(P_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),
    ASCENDC_TPL_UINT_DECL(SCH_MODE, 8, ASCENDC_TPL_UI_LIST, 0));

ASCENDC_TPL_SEL(
    // float + 4 p modes + FullM
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT), ASCENDC_TPL_UINT_SEL(P_MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),
        ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, 0)),
    // float16 + 4 p modes + FullM
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT16), ASCENDC_TPL_UINT_SEL(P_MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),
        ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, 0)), );

#endif // __CDIST_GRAD_TILING_KEY_H__
