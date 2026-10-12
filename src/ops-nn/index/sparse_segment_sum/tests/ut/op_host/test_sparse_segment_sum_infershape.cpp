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
#include <memory>
#include <vector>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "ut_op_common.h"
#include "../../../op_graph/sparse_segment_sum_proto.h"

namespace {
constexpr size_t kInputNum = 3;
constexpr size_t kOutputNum = 1;
constexpr size_t kOutputIndex = 0;

class SparseSegmentSumInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SparseSegmentSumInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SparseSegmentSumInfershapeTest TearDown" << std::endl; }
};

template <typename T>
gert::Tensor* CreateConstTensor(const std::vector<T>& data, ge::DataType dataType,
                                std::unique_ptr<uint8_t[]>& tensorHolder)
{
    size_t totalSize = 0;
    tensorHolder = gert::Tensor::CreateFollowing(static_cast<int64_t>(data.size()), dataType, totalSize);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensorHolder.get());
    tensor->MutableStorageShape().AppendDim(static_cast<int64_t>(data.size()));
    tensor->MutableOriginShape().AppendDim(static_cast<int64_t>(data.size()));
    tensor->SetOriginFormat(ge::FORMAT_ND);
    tensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(tensor->GetData<uint8_t>(), totalSize - sizeof(gert::Tensor), data.data(), data.size() * sizeof(T));
    return tensor;
}
} // namespace

TEST_F(SparseSegmentSumInfershapeTest, sparse_segment_sum_infershape_int32_segment_ids_1d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentSum")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{8}, {8}};
    gert::StorageShape indicesShape = {{4}, {4}};
    gert::Shape yShape = {};
    constexpr int64_t expectedRank = 1;
    constexpr int64_t expectedDim0 = 4;

    std::unique_ptr<uint8_t[]> segmentIdsTensorHolder;
    auto segmentIdsTensor = CreateConstTensor<int32_t>({0, 0, 2, 3}, ge::DT_INT32, segmentIdsTensorHolder);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(kInputNum, kOutputNum)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &indicesShape, segmentIdsTensor})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    auto outputShape = context->GetOutputShape(kOutputIndex);
    ASSERT_NE(outputShape, nullptr);
    ASSERT_EQ(outputShape->GetDimNum(), expectedRank);
    EXPECT_EQ(outputShape->GetDim(0), expectedDim0);
}

TEST_F(SparseSegmentSumInfershapeTest, sparse_segment_sum_infershape_int64_segment_ids_3d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentSum")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{8, 16, 32}, {8, 16, 32}};
    gert::StorageShape indicesShape = {{4}, {4}};
    gert::Shape yShape = {};
    constexpr int64_t expectedRank = 3;
    constexpr int64_t expectedDim0 = 6;
    constexpr int64_t expectedDim1 = 16;
    constexpr int64_t expectedDim2 = 32;
    constexpr size_t expectedDim2Index = 2;

    std::unique_ptr<uint8_t[]> segmentIdsTensorHolder;
    auto segmentIdsTensor = CreateConstTensor<int64_t>({0, 1, 1, 5}, ge::DT_INT64, segmentIdsTensorHolder);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(kInputNum, kOutputNum)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &indicesShape, segmentIdsTensor})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    auto outputShape = context->GetOutputShape(kOutputIndex);
    ASSERT_NE(outputShape, nullptr);
    ASSERT_EQ(outputShape->GetDimNum(), expectedRank);
    EXPECT_EQ(outputShape->GetDim(0), expectedDim0);
    EXPECT_EQ(outputShape->GetDim(1), expectedDim1);
    EXPECT_EQ(outputShape->GetDim(expectedDim2Index), expectedDim2);
}

TEST_F(SparseSegmentSumInfershapeTest, sparse_segment_sum_infershape_non_const_segment_ids)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentSum")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8, 16};
    gert::Shape indicesShape = {4};
    gert::Shape segmentIdsShape = {4};
    gert::Shape yShape = {};
    constexpr int64_t expectedRank = 2;
    constexpr int64_t expectedDim0 = -1;
    constexpr int64_t expectedDim1 = 16;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(kInputNum, kOutputNum)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &indicesShape, &segmentIdsShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_DOUBLE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_DOUBLE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    auto outputShape = context->GetOutputShape(kOutputIndex);
    ASSERT_NE(outputShape, nullptr);
    ASSERT_EQ(outputShape->GetDimNum(), expectedRank);
    EXPECT_EQ(outputShape->GetDim(0), expectedDim0);
    EXPECT_EQ(outputShape->GetDim(1), expectedDim1);
}

TEST_F(SparseSegmentSumInfershapeTest, sparse_segment_sum_infershape_indices_not_1d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentSum")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8, 16};
    gert::Shape indicesShape = {2, 2};
    gert::Shape segmentIdsShape = {4};
    gert::Shape yShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(kInputNum, kOutputNum)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &indicesShape, &segmentIdsShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(SparseSegmentSumInfershapeTest, sparse_segment_sum_infershape_indices_segment_ids_shape_mismatch)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentSum")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {8, 16};
    gert::Shape indicesShape = {3};
    gert::Shape segmentIdsShape = {4};
    gert::Shape yShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(kInputNum, kOutputNum)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &indicesShape, &segmentIdsShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(SparseSegmentSumInfershapeTest, sparse_segment_sum_inferdtype_pass_through)
{
    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseSegmentSum")->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);

    ge::DataType xDtype = ge::DT_UINT64;
    ge::DataType indicesDtype = ge::DT_INT32;
    ge::DataType segmentIdsDtype = ge::DT_INT32;
    ge::DataType outputDtype = ge::DT_FLOAT;

    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(kInputNum, kOutputNum)
                      .IrInstanceNum({1, 1, 1})
                      .InputDataTypes({&xDtype, &indicesDtype, &segmentIdsDtype})
                      .OutputDataTypes({&outputDtype})
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(kOutputIndex), ge::DT_UINT64);
}
