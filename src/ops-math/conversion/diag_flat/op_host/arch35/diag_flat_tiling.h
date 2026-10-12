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
 * \file diag_flat_tiling.h
 * \brief DiagFlat arch35 tiling function export (called by diag_v2 for 1D→2D delegation)
 *
 * One-way dependency: this header is self-contained (no diag_v2 includes).
 * diag_v2 includes this header and calls TilingDiagFlatArch35() for rank==1 input.
 */

#ifndef __DIAG_FLAT_ARCH35_TILING_H__
#define __DIAG_FLAT_ARCH35_TILING_H__

#include <cstdint>
#include "register/op_impl_registry.h"

namespace optiling {

// Output struct returned by TilingDiagFlatArch35.
// Caller fills its own TilingData from these values.
struct DiagFlatTilingOutput {
    int64_t numInput;
    int64_t diagonal;
    int64_t outWidth;
    int64_t outTotal;
    int64_t outPerCore;
    int64_t tileLength;
    int64_t realCoreNum;
    uint32_t localMemSize;  // ubSize - SIMT_DCACHE_SIZE, for SetLocalMemorySize
};

// Core 1D→2D tiling computation. Does NOT call context->GetTilingData<>()
// — the caller is responsible for filling its own TilingData struct.
// Also sets BlockDim, LocalMemorySize, and Workspace on the context.
// TilingKey selection is the caller's responsibility.
ge::graphStatus TilingDiagFlatArch35(gert::TilingContext* context, DiagFlatTilingOutput* out);

} // namespace optiling

#endif // __DIAG_FLAT_ARCH35_TILING_H__
