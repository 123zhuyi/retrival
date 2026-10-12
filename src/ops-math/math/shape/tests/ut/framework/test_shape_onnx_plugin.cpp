/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "../../../framework/shape_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name, "TestOp"); }
} // namespace

TEST(OnnxShapePluginTest, SetsInt64Dtype)
{
    ge::Operator op_src = CreateOperator("src");
    ge::Operator op_dest = CreateOperator("shape");
    uint32_t dtype = ge::DT_UNDEFINED;

    EXPECT_EQ(domi::ParseParamsShape(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("dtype", dtype), ge::GRAPH_SUCCESS);
    EXPECT_EQ(dtype, static_cast<uint32_t>(ge::DT_INT64));
}
