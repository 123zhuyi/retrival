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
 * \file tensor_redirect_infershape.cpp
 * \brief
 */
#include "infershape_elewise_util.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
// output_x 与 x 同 shape（same-as-input）
// InferDataType 仅图场景使用，交付在 op_graph/tensor_redirect_graph_infer.cpp
IMPL_OP_INFERSHAPE(TensorRedirect).InferShape(Ops::Base::InferShape4Elewise);
} // namespace ops
