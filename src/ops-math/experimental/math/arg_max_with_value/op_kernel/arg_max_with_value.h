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
 * \file arg_max_with_value.h
 * \brief ArgWithValue kernel with modular per-pattern classes.
 *
 * The host flattens the reduce to firstDim x axisSize x lastDim and picks a pattern (see the tiling
 * data header). Each pattern is its own small class (Copy / Last / NLast) built on a shared ArgBase, and
 * uses fixed UB addresses, raw MTE transfers, and explicit point-to-point events. RunArgWithValue is the
 * compile-time MODE switch the entry dispatches into; the
 * The IS_MIN template selects the reduction direction so both operators share the same implementation.
 */
#ifndef ARG_MAX_WITH_VALUE_H
#define ARG_MAX_WITH_VALUE_H

#include "arg_max_with_value_tiling_data.h"
#include "arg_max_with_value_tiling_key.h" // visible to kernel codegen so the per-schMode binaries are emitted
#include "arg_max_with_value_base.h"
#include "arg_max_with_value_copy.h"
#include "arg_max_with_value_last.h"
#include "arg_max_with_value_nlast.h"
#include "arg_max_with_value_small_last.h"

namespace ArgWithValueNs {
using namespace AscendC;

// Compile-time MODE dispatch (host SetTilingKey -> entry if constexpr -> here). Namespace-agnostic name so
// both operators reuse it unchanged. ws feeds the LAST axis-split (splitAxis 1/2) and NLAST axis-split
// (splitAxis 3) cross-core combines (2D paths store one float value plane plus one int32 index plane).
template <typename T, bool IS_MIN, uint32_t SCHEDULE, bool GATHER>
__aicore__ inline void RunArgWithValue(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR ws,
                                       __tiling_data_ptr__ ArgMaxWithValueTilingData* t)
{
    if constexpr (SCHEDULE == ARG_SCH_COPY) {
        ArgCopy<T, IS_MIN> op;
        op.Init(x, indice, values, t);
        op.Process();
    } else if constexpr (SCHEDULE == ARG_SCH_LAST_DIRECT) {
        ArgLastDirect<T, IS_MIN> op;
        op.Init(x, indice, values, t);
        op.Process();
    } else if constexpr (SCHEDULE == ARG_SCH_LAST_LONG) {
        ArgLastDirect<T, IS_MIN, true> op;
        op.Init(x, indice, values, t);
        op.Process();
    } else if constexpr (SCHEDULE == ARG_SCH_LAST_LONG_PACKED) {
        ArgLastDirect<T, IS_MIN, false, true> op;
        op.Init(x, indice, values, t);
        op.Process();
    } else if constexpr (SCHEDULE <= ARG_SCH_LAST_SPLIT2) {
        ArgLast<T, IS_MIN, SCHEDULE, GATHER> op;
        op.Init(x, indice, values, ws, t);
        op.Process();
    } else {
        ArgNLast<T, IS_MIN, SCHEDULE> op;
        op.Init(x, indice, values, ws, t);
        op.Process();
    }
}
} // namespace ArgWithValueNs
#endif // ARG_MAX_WITH_VALUE_H
