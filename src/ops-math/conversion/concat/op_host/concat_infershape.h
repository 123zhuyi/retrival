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
 * \file concat_infershape.h
 * \brief
 */
#ifndef CONCAT_INFERSHAPE_H_
#define CONCAT_INFERSHAPE_H_
#include "log/log.h"
#include "infershape_broadcast_util.h"
#include "op_common/op_host/util/shape_util.h"
#include "register/op_impl_registry.h"
namespace ops {
constexpr size_t INDEX_CONCAT_DIM = 0;
constexpr size_t INDEX_N = 1;
constexpr size_t INPUT_IDX = 1;
constexpr size_t INPUT_IDX_FOR_CONCAT_V2 = 0;

ge::graphStatus ConcatInferShapeCommon(
    gert::InferShapeContext* context, const int64_t dynamic_input_idx, int64_t num_concat, int64_t axis);

template <typename T>
inline static int64_t GetConcatDim(gert::InferShapeContext* context, int64_t dimIdx);

ge::graphStatus InferShapeForConcatAndConcatV2(gert::InferShapeContext* context, int64_t inputIdx, int64_t dimIdx);
}  // namespace ops
#endif // CONCAT_INFERSHAPE_H_