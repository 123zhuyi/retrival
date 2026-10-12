/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_graph/sparse_segment_mean_grad_proto.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ge;

// 说明：InferShapeContextFaker 的 GetInputTensor() 返回的并不是 InputShapes() 里传入的那个 Tensor
// （实测 GetInputShape() 读到的是传入对象，GetInputTensor() 读到的是 faker 自建的空 tensor，GetSize()==0），
// 因此这里 GetConstInt(output_dim0) 必然失败、首维回退 -1。用例据此断言 -1，覆盖「output_dim0 非编译期
// 常量时首维回退」这条真实分支；output_dim0 为真常量时 y.dim(0)==output_dim0 的路径由图模式 example 在
// 真实硬件上端到端覆盖（AI CPU kernel 的 CheckShapePara 会校验 y.dim(0)==output_dim0）。
namespace {
constexpr int32_t kOutputDim0Value = 3;
constexpr int64_t kUnknownDim = -1;
constexpr int64_t kUnknownRank = -2;

// output_dim0 is a data-dependent input, so it must be fed as a rank 0 const tensor.
std::unique_ptr<uint8_t[]> BuildOutputDim0Tensor(const std::vector<int64_t>& shapeDims, int32_t value,
                                                 gert::Tensor*& tensor)
{
    size_t totalSize = 0;
    auto holder = gert::Tensor::CreateFollowing(1, ge::DT_INT32, totalSize);
    tensor = reinterpret_cast<gert::Tensor*>(holder.get());
    for (const int64_t dim : shapeDims) {
        tensor->MutableStorageShape().AppendDim(dim);
        tensor->MutableOriginShape().AppendDim(dim);
    }
    tensor->SetOriginFormat(ge::FORMAT_ND);
    tensor->SetStorageFormat(ge::FORMAT_ND);
    *(tensor->GetData<int32_t>()) = value;
    return holder;
}

} // namespace

class SparseSegmentMeanGradInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SparseSegmentMeanGradInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SparseSegmentMeanGradInfershapeTest TearDown" << std::endl; }
};

TEST_F(SparseSegmentMeanGradInfershapeTest, InferShape2dSuccess)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentMeanGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape indicesShape = {{3}, {3}};
    gert::StorageShape segmentIdsShape = {{3}, {3}};
    gert::Shape yShape;
    gert::Tensor* outputDim0Tensor = nullptr;
    auto holderData = BuildOutputDim0Tensor({}, kOutputDim0Value, outputDim0Tensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&xShape, &indicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const auto* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(CreateShape({kUnknownDim, 3})));
}

TEST_F(SparseSegmentMeanGradInfershapeTest, InferShape3dSuccess)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentMeanGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape indicesShape = {{3}, {3}};
    gert::StorageShape segmentIdsShape = {{3}, {3}};
    gert::Shape yShape;
    gert::Tensor* outputDim0Tensor = nullptr;
    auto holderData = BuildOutputDim0Tensor({}, kOutputDim0Value, outputDim0Tensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&xShape, &indicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const auto* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(CreateShape({kUnknownDim, 3, 4})));
}

TEST_F(SparseSegmentMeanGradInfershapeTest, InferShapeXUnknownRankSuccess)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentMeanGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{kUnknownRank}, {kUnknownRank}};
    gert::StorageShape indicesShape = {{3}, {3}};
    gert::StorageShape segmentIdsShape = {{3}, {3}};
    gert::Shape yShape;
    gert::Tensor* outputDim0Tensor = nullptr;
    auto holderData = BuildOutputDim0Tensor({}, kOutputDim0Value, outputDim0Tensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&xShape, &indicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const auto* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(CreateShape({kUnknownRank})));
}

// x must be at least 1-D. Without the rank guard, gert::Shape::SetDim(0, v) on a rank 0 shape keeps
// dim_num_ at 0, so the operator would silently report a rank 0 output and return GRAPH_SUCCESS.
TEST_F(SparseSegmentMeanGradInfershapeTest, InferShapeScalarXFailed)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentMeanGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{}, {}};
    gert::StorageShape indicesShape = {{3}, {3}};
    gert::StorageShape segmentIdsShape = {{3}, {3}};
    gert::Shape yShape;
    gert::Tensor* outputDim0Tensor = nullptr;
    auto holderData = BuildOutputDim0Tensor({}, kOutputDim0Value, outputDim0Tensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&xShape, &indicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// An unknown dim (-1) merges with any known dim, matching the Merge semantics of the source proto.
TEST_F(SparseSegmentMeanGradInfershapeTest, InferShapeIndicesUnknownDimSuccess)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentMeanGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape indicesShape = {{kUnknownDim}, {kUnknownDim}};
    gert::StorageShape segmentIdsShape = {{5}, {5}};
    gert::Shape yShape;
    gert::Tensor* outputDim0Tensor = nullptr;
    auto holderData = BuildOutputDim0Tensor({}, kOutputDim0Value, outputDim0Tensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&xShape, &indicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const auto* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(CreateShape({kUnknownDim, 3})));
}

// An unknown rank (-2) indices input must not be rejected either.
TEST_F(SparseSegmentMeanGradInfershapeTest, InferShapeIndicesUnknownRankSuccess)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentMeanGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape indicesShape = {{kUnknownRank}, {kUnknownRank}};
    gert::StorageShape segmentIdsShape = {{5}, {5}};
    gert::Shape yShape;
    gert::Tensor* outputDim0Tensor = nullptr;
    auto holderData = BuildOutputDim0Tensor({}, kOutputDim0Value, outputDim0Tensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&xShape, &indicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const auto* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(CreateShape({kUnknownDim, 3})));
}

// Two known but different dims still conflict.
TEST_F(SparseSegmentMeanGradInfershapeTest, InferShapeIndicesDimMismatchFailed)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentMeanGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape indicesShape = {{3}, {3}};
    gert::StorageShape segmentIdsShape = {{4}, {4}};
    gert::Shape yShape;
    gert::Tensor* outputDim0Tensor = nullptr;
    auto holderData = BuildOutputDim0Tensor({}, kOutputDim0Value, outputDim0Tensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&xShape, &indicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(SparseSegmentMeanGradInfershapeTest, InferShapeIndicesNot1dFailed)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentMeanGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape indicesShape = {{3, 1}, {3, 1}};
    gert::StorageShape segmentIdsShape = {{3}, {3}};
    gert::Shape yShape;
    gert::Tensor* outputDim0Tensor = nullptr;
    auto holderData = BuildOutputDim0Tensor({}, kOutputDim0Value, outputDim0Tensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&xShape, &indicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(SparseSegmentMeanGradInfershapeTest, InferShapeOutputDim0NotScalarFailed)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentMeanGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape indicesShape = {{3}, {3}};
    gert::StorageShape segmentIdsShape = {{3}, {3}};
    gert::Shape yShape;
    gert::Tensor* outputDim0Tensor = nullptr;
    auto holderData = BuildOutputDim0Tensor({1}, kOutputDim0Value, outputDim0Tensor);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&xShape, &indicesShape, &segmentIdsShape, outputDim0Tensor})
                      .OutputShapes({&yShape})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(SparseSegmentMeanGradInfershapeTest, InferDataTypeFollowsX)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentMeanGrad"), nullptr);
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentMeanGrad")->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);

    ge::DataType xDtype = ge::DT_DOUBLE;
    ge::DataType indicesDtype = ge::DT_INT32;
    ge::DataType segmentIdsDtype = ge::DT_INT64;
    ge::DataType outputDim0Dtype = ge::DT_INT32;
    ge::DataType yDtype = ge::DT_DOUBLE;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(4, 1)
                             .NodeInputTd(0, ge::DT_DOUBLE, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::DT_DOUBLE, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&xDtype, &indicesDtype, &segmentIdsDtype, &outputDim0Dtype})
                             .OutputDataTypes({&yDtype})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferDataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_DOUBLE);
}
