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
 * \file adds_tiling_data.h
 * \brief Adds 算子 TilingData 结构体定义（atvoss 框架 - Elewise 模式）
 */

#ifndef ADDS_TILING_DATA_H_
#define ADDS_TILING_DATA_H_

#include "atvoss/elewise/elewise_base_struct.h"

/**
 * \struct AddsTilingData
 * \brief 自定义 TilingData 包装结构体（Elewise 模式必须使用 struct，不能用 using 别名）
 * 
 * 注意：
 * 1. baseTiling 必须是第一个成员（框架要求）
 * 2. scalarValue 用于存储标量值，通过 Kernel 入口的 SetVar 注入到 DAG
 */
struct AddsTilingData {
    ::Ops::Base::EleBaseTilingData baseTiling;  // 框架基础部分（必须第一个成员）
    float scalarValue;                          // 标量参数（运行时注入）
};

#endif  // ADDS_TILING_DATA_H_