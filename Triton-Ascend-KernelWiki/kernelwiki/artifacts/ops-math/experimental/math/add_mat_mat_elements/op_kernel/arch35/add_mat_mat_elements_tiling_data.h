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
 * \file add_mat_mat_elements_tiling_data.h
 * \brief AddMatMatElements TilingData 结构体定义
 *
 * 使用标准 C++ struct，禁止使用废弃宏 BEGIN_TILING_DATA_DEF
 */

#ifndef ADD_MAT_MAT_ELEMENTS_TILING_DATA_H_
#define ADD_MAT_MAT_ELEMENTS_TILING_DATA_H_

struct AddMatMatElementsTilingData {
    uint64_t totalLength;     // 总元素数（= a/b/c 所有维度之积）
    uint32_t tileLength;      // 单次 UB 处理的元素数（满足32B对齐）
    uint32_t blockNum;        // 实际使用的 AI Core 数
    uint32_t blockLength;     // 每个 Core 负责的元素数
    uint32_t lastBlockLength; // 最后一个 Core 的元素数（可能 < blockLength）
    float alphaVal;           // alpha 标量值（统一以 float 存储，Kernel 内按需转换）
    float betaVal;            // beta 标量值（统一以 float 存储）
};

#endif // ADD_MAT_MAT_ELEMENTS_TILING_DATA_H_
