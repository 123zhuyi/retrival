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
 * \file non_zero_with_value_tf_plugin.cpp
 * \brief TensorFlow plugin for NonZeroWithValue. Mirrors canndev's nonzero_with_value_plugin.cc:
 *        AutoMappingFn + output_type -> dtype attribute remap.
 */
#include "register/register.h"

namespace domi {
static Status AutoMappingFnNonZeroWithValue(const google::protobuf::Message* op_src, ge::Operator& op)
{
    Status ret = AutoMappingFn(op_src, op);
    if (ret != SUCCESS) {
        return FAILED;
    }
    ge::DataType outputDataType;
    op.GetAttr("output_type", outputDataType);
    (void)op.SetAttr("dtype", outputDataType);
    return ret;
}

REGISTER_CUSTOM_OP("NonZeroWithValue")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("NonZeroWithValue")
    .ParseParamsFn(AutoMappingFnNonZeroWithValue)
    .ImplyType(ImplyType::TVM);
} // namespace domi
