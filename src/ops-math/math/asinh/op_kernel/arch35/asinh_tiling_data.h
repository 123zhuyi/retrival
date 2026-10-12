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
 * \file asinh_tiling_data.h
 * \brief Asinh 算子 TilingData 结构体定义
 *
 * 与 DESIGN.md v1.1 §3.4 对齐：
 *   ✅ 必须使用标准 C++ struct 定义 TilingData
 *   ❌ 禁止使用废弃的 BEGIN_TILING_DATA_DEF 宏（REQUIREMENTS §8.4 强约束）
 *
 *   - totalNum：总元素数（input/out 一致）
 *   - blockFactor：单核处理元素数（按 32B 对齐）
 *   - ubFactor：单次 UB 循环处理元素数
 *     **v1.1 / DESIGN-API-COMPARE-1 修复：按 64 元素 (FP32 视角 256B) 对齐**
 *     满足 Compare.md line 114/125 的 count 256B 对齐强约束
 */
#ifndef _ASINH_TILING_DATA_H_
#define _ASINH_TILING_DATA_H_

struct AsinhTilingData {
    int64_t totalNum = 0;     // 总元素数（input/out 一致）
    int64_t blockFactor = 0;  // 单核处理元素数（按 32B 对齐，避免相邻核 CopyOut 写入越界）
    int64_t ubFactor = 0;     // 单次 UB 循环处理元素数（**FP32 视角 64 元素 = 256B 对齐**，
                              //   满足 Compare API 的 count 256B 对齐强约束）
};

#endif // _ASINH_TILING_DATA_H_
