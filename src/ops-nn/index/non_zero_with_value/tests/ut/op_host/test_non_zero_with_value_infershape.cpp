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
#include <iostream>
#include "log/log.h"
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "platform/platform_info.h"

class NonZeroWithValueProtoTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "NonZeroWithValueProtoTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "NonZeroWithValueProtoTest TearDown" << std::endl; }
};

// 静态 max-size:value=[numel], index=[2*numel], count=[1]
TEST_F(NonZeroWithValueProtoTest, infershape_static_2d)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NonZeroWithValue"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NonZeroWithValue")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape x_shape = {4, 8}; // numel = 32
    gert::Shape value_shape = {};
    gert::Shape index_shape = {};
    gert::Shape count_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 3)
                      .IrInputNum({1})
                      .InputShapes({&x_shape})
                      .OutputShapes({&value_shape, &index_shape, &count_shape})
                      .NodeAttrs({{"transpose", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto ctx = holder.GetContext<gert::InferShapeContext>();
    gert::Shape expected_value = {32};
    gert::Shape expected_index = {64};
    gert::Shape expected_count = {1};
    ASSERT_EQ(Ops::Base::ToString(*ctx->GetOutputShape(0)), Ops::Base::ToString(expected_value));
    ASSERT_EQ(Ops::Base::ToString(*ctx->GetOutputShape(1)), Ops::Base::ToString(expected_index));
    ASSERT_EQ(Ops::Base::ToString(*ctx->GetOutputShape(2)), Ops::Base::ToString(expected_count));
}

// unknown rank → 三个输出都设为 unknown rank
TEST_F(NonZeroWithValueProtoTest, infershape_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NonZeroWithValue")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape x_shape = {-2};
    gert::Shape value_shape = {};
    gert::Shape index_shape = {};
    gert::Shape count_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 3)
                      .IrInputNum({1})
                      .InputShapes({&x_shape})
                      .OutputShapes({&value_shape, &index_shape, &count_shape})
                      .NodeAttrs({{"transpose", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

// 非 2D 输入 → 失败
TEST_F(NonZeroWithValueProtoTest, infershape_rank3_fail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NonZeroWithValue")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape x_shape = {2, 3, 4};
    gert::Shape value_shape = {};
    gert::Shape index_shape = {};
    gert::Shape count_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 3)
                      .IrInputNum({1})
                      .InputShapes({&x_shape})
                      .OutputShapes({&value_shape, &index_shape, &count_shape})
                      .NodeAttrs({{"transpose", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// InferDataType:value=float, index=int32(由 attr dtype), count=int32
TEST_F(NonZeroWithValueProtoTest, inferdatatype_basic)
{
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NonZeroWithValue")->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);

    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(1)
                      .NodeIoNum(1, 3)
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"transpose", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)}})
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDataTypeFunc(context), ge::GRAPH_SUCCESS);
}

TEST_F(NonZeroWithValueProtoTest, inferdatatype_nullptr)
{
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NonZeroWithValue")->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);
    ASSERT_EQ(inferDataTypeFunc(nullptr), ge::GRAPH_FAILED);
}
