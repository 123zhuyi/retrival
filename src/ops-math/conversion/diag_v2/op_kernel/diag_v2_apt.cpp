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
 * \file diag_v2_apt.cpp
 * \brief DiagV2 kernel entry for arch35 (DAV_3510, __NPU_ARCH__=3101)
 *
 * Design: DESIGN.md v2.4 Sec 3.4.1
 *
 * Dual dispatch on IS_1D_INPUT:
 *   IS_1D_INPUT=0 → 2D→1D diagonal extraction (DiagV2Simd kernel)
 *   IS_1D_INPUT=1 → 1D→2D diagonal matrix construction (DiagFlatSimd kernel, shared with diag_flat)
 */

#include "arch35/diag_v2_tiling_key.h"
#include "arch35/diag_v2.h"
#include "../diag_flat/arch35/diag_flat_simd.h"

#include <type_traits>
#include "kernel_operator.h"

using namespace AscendC;

template <int IS_1D_INPUT>
__aicore__ inline void LaunchKernel(GM_ADDR x, GM_ADDR y,
                                     const DiagV2Arch35TilingData* td)
{
    if constexpr (IS_1D_INPUT == 0) {
        // 2D→1D: diagonal extraction
        // 8-byte types (complex64/double/int64/uint64): alias through int64_t
        // because SIMT cannot dereference __gm__ double* / __gm__ Complex<float>*,
        // and DataCopyPad does not support complex64.
        // bool (1 byte): alias through uint8_t because DataCopyPad does not support bool.
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            DiagV2Simd<int64_t> op;
            op.Init(x, y, td);
            op.Process();
        } else if constexpr (std::is_same_v<DTYPE_X, bool>) {
            DiagV2Simd<uint8_t> op;
            op.Init(x, y, td);
            op.Process();
        } else {
            DiagV2Simd<DTYPE_X> op;
            op.Init(x, y, td);
            op.Process();
        }
    } else {
        // 1D→2D: construct diagonal matrix, delegate to diag_flat kernel
        DiagFlatArch35TilingData flatTd;
        flatTd.numInput    = td->numInput;
        flatTd.diagonal    = td->diagonal;
        flatTd.outWidth    = td->outWidth;
        flatTd.outTotal    = td->outTotal;
        flatTd.outPerCore  = td->outPerCore;
        flatTd.tileLength  = td->tileLength;
        flatTd.realCoreNum = td->realCoreNum;

        // 8-byte types (complex64/double/int64/uint64): alias through int64_t
        // bool (1 byte): alias through uint8_t (DataCopyPad does not support bool)
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            DiagFlatSimd<int64_t> op;
            op.Init(x, y, &flatTd);
            op.Process();
        } else if constexpr (std::is_same_v<DTYPE_X, bool>) {
            DiagFlatSimd<uint8_t> op;
            op.Init(x, y, &flatTd);
            op.Process();
        } else {
            DiagFlatSimd<DTYPE_X> op;
            op.Init(x, y, &flatTd);
            op.Process();
        }
    }
}

template <int IS_1D_INPUT>
__global__ __aicore__ void diag_v2(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    SetSysWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(DiagV2Arch35TilingData);
    GET_TILING_DATA_WITH_STRUCT(DiagV2Arch35TilingData, tilingData, tiling);

    LaunchKernel<IS_1D_INPUT>(x, y, &tilingData);
}
