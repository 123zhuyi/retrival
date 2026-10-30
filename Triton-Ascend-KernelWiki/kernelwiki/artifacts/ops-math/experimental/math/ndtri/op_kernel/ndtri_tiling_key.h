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
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
/*!
 * \file ndtri_tiling_key.h
 * \brief Ndtri Tiling 模板参数定义
 *
 * 模板参数维度：
 *   - D_T: 输入/输出 Tensor 的数据类型（C_DT_FLOAT / C_DT_FLOAT16 / C_DT_BF16）
 *   - K_ALIGN: 32B 对齐标记（1=对齐, 0=非对齐）
 *
 * 共 6 个 TilingKey：{fp32, fp16, bf16} × {对齐, 非对齐}
 *
 * 迭代一：仅 {C_DT_FLOAT, K_ALIGN=1} 真实运行；其他 5 个 TilingKey 在本骨架阶段
 * 由 Host 派发进入 Kernel 后，Kernel 代码路径统一走 FP32（fp16/bf16 路径暂先
 * 占位——即 Compute 中 IS_FP32 分支），迭代二会替换成真实 Cast 链路。
 */

#ifndef NDTRI_TILING_KEY_H_
#define NDTRI_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(
    Ndtri, ASCENDC_TPL_DATATYPE_DECL(D_T, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(K_ALIGN, 8, ASCENDC_TPL_UI_LIST, 0, 1));

// 所有 6 个组合保留（骨架）：fp32/fp16/bf16 × 对齐/非对齐
// 迭代一仅 fp32+对齐真正跑通，其他 5 个分支使用相同 Kernel 代码路径（后续迭代替换）
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT), ASCENDC_TPL_UINT_SEL(K_ALIGN, ASCENDC_TPL_UI_LIST, 0, 1)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT16), ASCENDC_TPL_UINT_SEL(K_ALIGN, ASCENDC_TPL_UI_LIST, 0, 1)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_BF16), ASCENDC_TPL_UINT_SEL(K_ALIGN, ASCENDC_TPL_UI_LIST, 0, 1)), );

#endif // NDTRI_TILING_KEY_H_
