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
 * \file shape_util.cpp
 * \brief
 */
#include "op_common/log/log.h"
#include "op_common/op_host/util/shape_util.h"

namespace Ops {
namespace Base {
constexpr int64_t UNKNOWN_RANK_DIM_VALUE = -2LL;
constexpr int64_t UNKNOWN_DIM_VALUE = -1LL;

void SetUnknownRank(gert::Shape& shape)
{
    shape.SetDimNum(0);
    shape.AppendDim(UNKNOWN_RANK_DIM_VALUE);
}

/*
 * @brief: check whether the output shape is unknown rank
 * @param [out] shape: the output shape ptr
 * @return ge::graphStatus
 */
bool IsUnknownRank(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == UNKNOWN_RANK_DIM_VALUE;
}

void SetUnknownShape(int64_t rank, gert::Shape& shape)
{
    OP_CHECK_IF(rank < 0, OP_LOGE("SetUnknownShape", "the rank value is invalid, return unsuccessful"), return);
    const size_t dimNum = static_cast<size_t>(rank);
    shape.SetDimNum(dimNum);
    for (size_t i = 0; i < dimNum; i++) {
        shape.SetDim(i, UNKNOWN_DIM_VALUE);
    }
}

bool IsUnknownShape(const gert::Shape& shape)
{
    size_t dimNum = shape.GetDimNum();
    for (size_t i = 0; i < dimNum; i++) {
        if (shape.GetDim(i) == UNKNOWN_DIM_VALUE) {
            return true;
        }
    }
    return false;
}

} // namespace Base
} // namespace Ops
