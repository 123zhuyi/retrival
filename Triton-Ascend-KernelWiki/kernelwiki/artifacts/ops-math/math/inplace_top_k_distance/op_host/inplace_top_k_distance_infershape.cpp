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
 * \file inplace_top_k_distance_infershape.cpp
 * \brief InferShape for InplaceTopKDistance.
 *        The operator declares no output: it rewrites topk_pq_distance / topk_pq_index / topk_pq_ivf in place.
 *        There is no shape to infer, so this registration is intentionally empty.
 */

#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
ge::graphStatus InferShape4InplaceTopKDistance([[maybe_unused]] gert::InferShapeContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(InplaceTopKDistance).InferShape(InferShape4InplaceTopKDistance);
} // namespace ops
