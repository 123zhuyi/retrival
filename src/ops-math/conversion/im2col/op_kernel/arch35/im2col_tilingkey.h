/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file im2col_tilingkey.h
 * \brief
 */

#ifndef _IM2COL_TILING_KEY_H_
#define _IM2COL_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_FORMAT_NCHW 0
#define TPL_FORMAT_NHWC 1

#define TPL_UB_AXIS_NCHW_NC 0
#define TPL_UB_AXIS_NCHW_HW 1
#define TPL_UB_AXIS_NHWC_N 0
#define TPL_UB_AXIS_NHWC_H 1
#define TPL_UB_AXIS_NHWC_W 2
#define TPL_UB_AXIS_NHWC_C 3

ASCENDC_TPL_ARGS_DECL(
    Im2col,
    // data format
    ASCENDC_TPL_FORMAT_DECL(Format, TPL_FORMAT_NCHW, TPL_FORMAT_NHWC),
    // 切哪根轴
    ASCENDC_TPL_UINT_DECL(UbAxis, 8, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
    // 是否有pad
    ASCENDC_TPL_BOOL_DECL(IsPadding, 0, 1),
    // 是否为simt
    ASCENDC_TPL_BOOL_DECL(IsSIMT, 0, 1),
    // 是否为大shape
    ASCENDC_TPL_BOOL_DECL(IsBigShape, 0, 1),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        // NCHW
        ASCENDC_TPL_FORMAT_SEL(Format, TPL_FORMAT_NCHW),
        ASCENDC_TPL_UINT_SEL(UbAxis, ASCENDC_TPL_UI_RANGE, 1, 0, 1),
        ASCENDC_TPL_BOOL_SEL(IsPadding, 0, 1),
        // 忽略 SIMT 参数
        ASCENDC_TPL_BOOL_SEL(IsSIMT, 0),
        ASCENDC_TPL_BOOL_SEL(IsBigShape, 0),
    ),
    ASCENDC_TPL_ARGS_SEL(
        // NHWC
        ASCENDC_TPL_FORMAT_SEL(Format, TPL_FORMAT_NHWC),
        ASCENDC_TPL_UINT_SEL(UbAxis, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
        ASCENDC_TPL_BOOL_SEL(IsPadding, 0, 1),
        // 忽略 SIMT 参数
        ASCENDC_TPL_BOOL_SEL(IsSIMT, 0),
        ASCENDC_TPL_BOOL_SEL(IsBigShape, 0),
    ),
    ASCENDC_TPL_ARGS_SEL(
        // 数据格式
        ASCENDC_TPL_FORMAT_SEL(Format, TPL_FORMAT_NCHW, TPL_FORMAT_NHWC),
        // 忽略
        ASCENDC_TPL_UINT_SEL(UbAxis, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_BOOL_SEL(IsPadding, 0),
        // SIMT 参数
        ASCENDC_TPL_BOOL_SEL(IsSIMT, 1),
        ASCENDC_TPL_BOOL_SEL(IsBigShape, 0, 1),
    ),
);

#endif
