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
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

// InferShapeBitwiseNot: out.shape = self.shape（逐元素，无 broadcast / 无降升维）。
// InferDataTypeBitwiseNot: out.dtype = self.dtype（同 dtype 直传）—— 该公共 faker 只校验输出 shape，
// dtype 透传由 InferDataTypeBitwiseNot 结构性保证（直接 SetOutputDataType(GetInputDataType)），
// 并由各 dtype 的 op_host tiling UT（GetDataTypeLength 分支）与 ST（NPU）联合佐证。

class BitwiseNotInferShape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "BitwiseNotInferShape SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "BitwiseNotInferShape TearDown" << std::endl;
    }
};

// 从运行时 vector 构造 gert::StorageShape（StorageShape 仅 initializer_list 构造，故用 AppendDim）。
static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& shape)
{
    gert::StorageShape s;
    for (int64_t d : shape) {
        s.MutableShape().AppendDim(d);
        s.MutableStorageShape().AppendDim(d);
    }
    return s;
}

static void RunInferShapeCase(const std::vector<int64_t>& shape, ge::DataType dtype)
{
    gert::StorageShape inShape = MakeStorageShape(shape);
    gert::InfershapeContextPara infershapeContextPara(
        "BitwiseNot",
        {
            {inShape, dtype, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, dtype, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {shape};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ---- out.shape == self.shape，6 dtype（同一 rank=3 shape） ----
TEST_F(BitwiseNotInferShape, bitwise_not_infershape_int8)
{
    RunInferShapeCase({4, 3, 4}, ge::DT_INT8);
}

TEST_F(BitwiseNotInferShape, bitwise_not_infershape_int16)
{
    RunInferShapeCase({4, 3, 4}, ge::DT_INT16);
}

TEST_F(BitwiseNotInferShape, bitwise_not_infershape_int32)
{
    RunInferShapeCase({4, 3, 4}, ge::DT_INT32);
}

TEST_F(BitwiseNotInferShape, bitwise_not_infershape_int64)
{
    RunInferShapeCase({4, 3, 4}, ge::DT_INT64);
}

TEST_F(BitwiseNotInferShape, bitwise_not_infershape_uint8)
{
    RunInferShapeCase({4, 3, 4}, ge::DT_UINT8);
}

TEST_F(BitwiseNotInferShape, bitwise_not_infershape_bool)
{
    RunInferShapeCase({4, 3, 4}, ge::DT_BOOL);
}

// ---- 不同 rank / 形状（确认逐元素 shape 透传，无降升维） ----
TEST_F(BitwiseNotInferShape, bitwise_not_infershape_rank2)
{
    RunInferShapeCase({2, 3}, ge::DT_INT16);
}

TEST_F(BitwiseNotInferShape, bitwise_not_infershape_rank0_scalar)
{
    RunInferShapeCase({}, ge::DT_INT32);
}

TEST_F(BitwiseNotInferShape, bitwise_not_infershape_empty)
{
    RunInferShapeCase({0}, ge::DT_INT8);
}

TEST_F(BitwiseNotInferShape, bitwise_not_infershape_rank4)
{
    RunInferShapeCase({2, 5, 7, 3}, ge::DT_INT64);
}

// ---- 动态 shape（-1 维透传） ----
TEST_F(BitwiseNotInferShape, bitwise_not_infershape_dynamic)
{
    RunInferShapeCase({3, -1, 4}, ge::DT_INT16);
}

// ============================================================================
// InferDataType: out.dtype == self.dtype（同 dtype 直传，无类型提升）
// 公共 infershape faker 不调用 infer_datatype，故直接构造 InferDataTypeContext 调用，
// 校验 6 dtype 输出 dtype == 输入 dtype。
// ============================================================================
static void RunInferDataTypeCase(ge::DataType dtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    auto holder = builder.OpType("BitwiseNot")
                      .OpName("BitwiseNot")
                      .IONum(1, 1)
                      .InputTensorDesc(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto* opImpl = spaceRegistry->GetOpImpl("BitwiseNot");
    ASSERT_NE(opImpl, nullptr);
    auto inferDataTypeFunc = opImpl->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);

    EXPECT_EQ(inferDataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), dtype);
}

TEST_F(BitwiseNotInferShape, bitwise_not_inferdatatype_int8)
{
    RunInferDataTypeCase(ge::DT_INT8);
}

TEST_F(BitwiseNotInferShape, bitwise_not_inferdatatype_int16)
{
    RunInferDataTypeCase(ge::DT_INT16);
}

TEST_F(BitwiseNotInferShape, bitwise_not_inferdatatype_int32)
{
    RunInferDataTypeCase(ge::DT_INT32);
}

TEST_F(BitwiseNotInferShape, bitwise_not_inferdatatype_int64)
{
    RunInferDataTypeCase(ge::DT_INT64);
}

TEST_F(BitwiseNotInferShape, bitwise_not_inferdatatype_uint8)
{
    RunInferDataTypeCase(ge::DT_UINT8);
}

TEST_F(BitwiseNotInferShape, bitwise_not_inferdatatype_bool)
{
    RunInferDataTypeCase(ge::DT_BOOL);
}
