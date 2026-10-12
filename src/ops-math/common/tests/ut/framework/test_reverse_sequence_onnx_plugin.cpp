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

#include "../../../src/framework/reverse_sequence_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name, "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

// 缺失 attribute 字段：应使用默认值 (batch_dim=1, seq_dim=0) 并返回 SUCCESS
TEST(OnnxReverseSequencePluginTest, NoAttributeParseReturnsSuccessWithDefaults)
{
    ge::Operator op_src = CreateOperator("src");
    ge::Operator op_dest = CreateOperator("reverse_sequence");

    EXPECT_EQ(domi::ParseParamsReverseSequence(op_src, op_dest), domi::SUCCESS);

    int batch_dim = -1;
    int seq_dim = -1;
    EXPECT_EQ(op_dest.GetAttr("batch_dim", batch_dim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("seq_dim", seq_dim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(batch_dim, 1);
    EXPECT_EQ(seq_dim, 0);
}

// 合法 JSON 但无 attribute 数组：保持默认值并返回 SUCCESS
TEST(OnnxReverseSequencePluginTest, EmptyAttributeArrayKeepsDefaults)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("reverse_sequence");

    EXPECT_EQ(domi::ParseParamsReverseSequence(op_src, op_dest), domi::SUCCESS);

    int batch_dim = -1;
    int seq_dim = -1;
    EXPECT_EQ(op_dest.GetAttr("batch_dim", batch_dim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("seq_dim", seq_dim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(batch_dim, 1);
    EXPECT_EQ(seq_dim, 0);
}

// 仅设置 batch_axis=0：batch_dim 应被覆盖为 0，seq_dim 保持默认 0
TEST(OnnxReverseSequencePluginTest, BatchAxisZeroSetsBatchDimZero)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"batch_axis","type":2,"i":0}]})");
    ge::Operator op_dest = CreateOperator("reverse_sequence");

    EXPECT_EQ(domi::ParseParamsReverseSequence(op_src, op_dest), domi::SUCCESS);

    int batch_dim = -1;
    EXPECT_EQ(op_dest.GetAttr("batch_dim", batch_dim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(batch_dim, 0);
}

// batch_axis=1：batch_dim 显式为 1（与默认相同）
TEST(OnnxReverseSequencePluginTest, BatchAxisOneKeepsBatchDimOne)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"batch_axis","type":2,"i":1}]})");
    ge::Operator op_dest = CreateOperator("reverse_sequence");

    EXPECT_EQ(domi::ParseParamsReverseSequence(op_src, op_dest), domi::SUCCESS);

    int batch_dim = -1;
    EXPECT_EQ(op_dest.GetAttr("batch_dim", batch_dim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(batch_dim, 1);
}

// time_axis=5：seq_dim 应被覆盖为 5
TEST(OnnxReverseSequencePluginTest, TimeAxisValueOverridesSeqDim)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"time_axis","type":2,"i":5}]})");
    ge::Operator op_dest = CreateOperator("reverse_sequence");

    EXPECT_EQ(domi::ParseParamsReverseSequence(op_src, op_dest), domi::SUCCESS);

    int seq_dim = -1;
    EXPECT_EQ(op_dest.GetAttr("seq_dim", seq_dim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(seq_dim, 5);
}

// 同时设置两个属性
TEST(OnnxReverseSequencePluginTest, BothAxesParsedTogether)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"batch_axis","type":2,"i":0},{"name":"time_axis","type":2,"i":3}]})");
    ge::Operator op_dest = CreateOperator("reverse_sequence");

    EXPECT_EQ(domi::ParseParamsReverseSequence(op_src, op_dest), domi::SUCCESS);

    int batch_dim = -1;
    int seq_dim = -1;
    EXPECT_EQ(op_dest.GetAttr("batch_dim", batch_dim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("seq_dim", seq_dim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(batch_dim, 0);
    EXPECT_EQ(seq_dim, 3);
}

// 非法 JSON 字符串：应返回 FAILED
TEST(OnnxReverseSequencePluginTest, MalformedJsonReturnsFailed)
{
    ge::Operator op_src = CreateSourceOperator("not-a-json");
    ge::Operator op_dest = CreateOperator("reverse_sequence");

    EXPECT_EQ(domi::ParseParamsReverseSequence(op_src, op_dest), domi::FAILED);
}

// 字段类型不匹配（i 不是整数）：应返回 FAILED
TEST(OnnxReverseSequencePluginTest, WrongFieldTypeReturnsFailed)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"batch_axis","type":2,"i":"not-int"}]})");
    ge::Operator op_dest = CreateOperator("reverse_sequence");

    EXPECT_EQ(domi::ParseParamsReverseSequence(op_src, op_dest), domi::FAILED);
}
