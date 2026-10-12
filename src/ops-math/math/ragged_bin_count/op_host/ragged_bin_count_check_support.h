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
 * \file ragged_bin_count_check_support.h
 * \brief RaggedBinCount native dtype support policy for Ascend 950.
 */

#ifndef RAGGED_BIN_COUNT_CHECK_SUPPORT_H
#define RAGGED_BIN_COUNT_CHECK_SUPPORT_H

#include <cstdint>

#include "graph/operator.h"
#include "register/op_def_registry.h"

namespace ops {
inline bool IsRaggedBinCountNativeDtypeCombination(ge::DataType splitsDtype, ge::DataType valuesDtype,
                                                   ge::DataType sizeDtype, ge::DataType weightsDtype,
                                                   ge::DataType outputDtype)
{
    return splitsDtype == ge::DT_INT64 && (valuesDtype == ge::DT_INT32 || valuesDtype == ge::DT_INT64) &&
           sizeDtype == valuesDtype && weightsDtype == ge::DT_FLOAT && outputDtype == ge::DT_FLOAT;
}

inline ge::graphStatus CheckSupport4RaggedBinCount(const ge::Operator& op, ge::AscendString& result)
{
    constexpr uint32_t INPUT_SPLITS = 0U;
    constexpr uint32_t INPUT_VALUES = 1U;
    constexpr uint32_t INPUT_SIZE = 2U;
    constexpr uint32_t INPUT_WEIGHTS = 3U;
    constexpr uint32_t OUTPUT_RESULT = 0U;

    const ge::DataType splitsDtype = op.GetInputDesc(INPUT_SPLITS).GetDataType();
    const ge::DataType valuesDtype = op.GetInputDesc(INPUT_VALUES).GetDataType();
    const ge::DataType sizeDtype = op.GetInputDesc(INPUT_SIZE).GetDataType();
    const ge::DataType weightsDtype = op.GetInputDesc(INPUT_WEIGHTS).GetDataType();
    const ge::DataType outputDtype = op.GetOutputDesc(OUTPUT_RESULT).GetDataType();

    if (!IsRaggedBinCountNativeDtypeCombination(splitsDtype, valuesDtype, sizeDtype, weightsDtype, outputDtype)) {
        result = ge::AscendString(
            R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": "RaggedBinCount Ascend 950 supports only splits=int64, values/size=int32 or int64 with matching dtype, and weights/output=float32."})");
        return ge::GRAPH_FAILED;
    }

    result = ge::AscendString(
        R"({"isSupported": "True", "dynamicCompileStatic": "True", "reason": "RaggedBinCount native dtype contract passed."})");
    return ge::GRAPH_SUCCESS;
}
} // namespace ops

#endif // RAGGED_BIN_COUNT_CHECK_SUPPORT_H
