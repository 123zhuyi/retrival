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
 * \file trilu_tiling.cpp
 * \brief Tiling implementation for trilu operator (v2.0).
 *
 * Dispatches to Tiling4Trilu<upper, true> (diagonal_as_input=true, k read via
 * GetOptionalInputTensor). After Tiling4Trilu returns, the tiling key is
 * rewritten to upper*100 + baseKey so the kernel can recover `upper` from
 * the hundreds digit. Requires dimNum>=2 (aligned with tril/triu).
 */

#include "conversion/triu/op_host/arch35/triu_tiling.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

// ── Tiling data class registration (40 classes) ─────────────────────────────
// upper=0 (tril path)
REGISTER_TILING_DATA_CLASS(Trilu_21, TriangulatorNormalTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_22, TriangulatorNormalTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_24, TriangulatorNormalTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_28, TriangulatorNormalTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_31, TriangulatorTinyTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_32, TriangulatorTinyTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_34, TriangulatorTinyTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_38, TriangulatorTinyTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_41, TriangulatorMediumTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_42, TriangulatorMediumTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_44, TriangulatorMediumTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_48, TriangulatorMediumTilingData)
// upper=1 (triu path)
REGISTER_TILING_DATA_CLASS(Trilu_121, TriangulatorNormalTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_122, TriangulatorNormalTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_124, TriangulatorNormalTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_128, TriangulatorNormalTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_131, TriangulatorTinyTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_132, TriangulatorTinyTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_134, TriangulatorTinyTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_138, TriangulatorTinyTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_141, TriangulatorMediumTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_142, TriangulatorMediumTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_144, TriangulatorMediumTilingData)
REGISTER_TILING_DATA_CLASS(Trilu_148, TriangulatorMediumTilingData)

REGISTER_TILING_DATA_CLASS(Trilu, TriangulatorTilingData)
// ── Main tiling entry ───────────────────────────────────────────────────────

constexpr int64_t TRILU_UPPER_VALUE = 100;

static ge::graphStatus TriluTilingFunc(gert::TilingContext* context)
{
    context->GetWorkspaceSizes(1);

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* upperPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, upperPtr);
    bool isUpper = (*upperPtr != 0);

    ge::graphStatus ret;
    if (isUpper) {
        ret = Tiling4Trilu<true, true>(context);
    } else {
        ret = Tiling4Trilu<false, true>(context);
    }
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    uint64_t baseKey = context->GetTilingKey();
    uint64_t fullKey = static_cast<uint64_t>(isUpper ? 1 : 0) * TRILU_UPPER_VALUE + baseKey;
    context->SetTilingKey(fullKey);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Trilu)
    .Tiling(TriluTilingFunc)
    .TilingInputsDataDependency({INDEX_K})
    .TilingParse<TriluCompileInfo>(TilingPrepare4Trilu);

} // namespace optiling
