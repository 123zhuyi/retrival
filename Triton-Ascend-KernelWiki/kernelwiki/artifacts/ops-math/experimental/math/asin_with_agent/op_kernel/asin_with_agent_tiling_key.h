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
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file asin_with_agent_tiling_key.h
 * \brief AsinWithAgent 模板参数定义（arch32）
 *
 * TilingKey 对应关系（由输入 dtype 决定）：
 *   TilingKey=0  C_DT_FLOAT   -> float（Group A，直接 AscendC::Asin）
 *   TilingKey=1  C_DT_FLOAT16 -> half（Group A，直接 AscendC::Asin）
 *   TilingKey=2  C_DT_DOUBLE  -> double（Group B，Cast fp64->fp32->Asin->fp64）
 *   TilingKey=3  C_DT_INT8    -> int8（Group C，Cast to fp32->Asin）
 *   TilingKey=4  C_DT_INT16   -> int16（Group C，Cast to fp32->Asin）
 *   TilingKey=5  C_DT_INT32   -> int32（Group C，Cast to fp32->Asin）
 *   TilingKey=6  C_DT_INT64   -> int64（Group C，Cast to fp32->Asin）
 *   TilingKey=7  C_DT_UINT8   -> uint8（Group C，Cast to fp32->Asin）
 *   TilingKey=8  C_DT_BOOL    -> bool（Group C，Cast to fp32->Asin）
 *
 * 迭代一：仅实现 TilingKey=0（fp32），其余在骨架中留桩
 */

#ifndef __ASIN_WITH_AGENT_TILING_KEY_H__
#define __ASIN_WITH_AGENT_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

// 模板参数声明：D_T 为输入 dtype
ASCENDC_TPL_ARGS_DECL(
    AsinWithAgent,
    ASCENDC_TPL_DATATYPE_DECL(
        D_T, C_DT_FLOAT, C_DT_FLOAT16, C_DT_DOUBLE, C_DT_INT8, C_DT_INT16, C_DT_INT32, C_DT_INT64, C_DT_UINT8,
        C_DT_BOOL, ASCENDC_TPL_INPUT(0)), );

// 模板实例化声明：枚举所有支持的 dtype
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT16)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_DOUBLE)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_INT8)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_INT16)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_INT32)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_INT64)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_UINT8)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_BOOL)), );

#endif // __ASIN_WITH_AGENT_TILING_KEY_H__
