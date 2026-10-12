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
 * \file fused_mul_add_n_tiling.h
 * \brief op_kernel UT 专用的 TilingData 反序列化注入头。
 *
 * 作用机制：
 *   cmake/ut.cmake 在编译 op_kernel UT 时，若本目录存在 <op>_tiling.h，则用
 *   `-include` 强制注入到 kernel 翻译单元。本文件复用 op_kernel/fused_mul_add_n_tiling_data.h
 *   中的真实 plain-struct FusedMulAddNTilingData，并把 GET_TILING_DATA_WITH_STRUCT /
 *   GET_TILING_DATA 重定义为「从一段扁平内存逐字拷贝」的实现，使 A2 kernel 在 CPU UT 下
 *   直接从 UT 填充的扁平 tiling 缓冲反序列化出 TilingData，从而让 op_kernel UT 完全自包含，
 *   不依赖 host tiling .so 的产出。
 */
#ifndef UT_FUSED_MUL_ADD_N_TILING_H_
#define UT_FUSED_MUL_ADD_N_TILING_H_

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "../../../op_kernel/fused_mul_add_n_tiling_data.h"

// UT 在测试中按字段顺序填充并 memcpy 到 tiling GM 的 POD；与真实 FusedMulAddNTilingData 同布局。
typedef FusedMulAddNTilingData FusedMulAddNTilingDataUT;

#ifdef __NPU_TILING__
inline [aicore] void InitTilingData(const __gm__ uint8_t* tiling, FusedMulAddNTilingData* constData)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)constData;
    for (size_t i = 0; i < sizeof(FusedMulAddNTilingData) / 4; i++) {
        *(dst + i) = *(src + i);
    }
}
#else
inline void InitTilingData(uint8_t* tiling, FusedMulAddNTilingData* constData)
{
    memcpy(constData, tiling, sizeof(FusedMulAddNTilingData));
}
#endif // __NPU_TILING__

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer))

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA(tilingData, tilingArg) \
    FusedMulAddNTilingData tilingData;         \
    InitTilingData(tilingArg, &tilingData)

#endif // UT_FUSED_MUL_ADD_N_TILING_H_
