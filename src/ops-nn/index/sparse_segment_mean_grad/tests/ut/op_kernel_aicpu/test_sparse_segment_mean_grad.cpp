/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected

using namespace std;
using namespace aicpu;

namespace {
// x is viewed as [kNumSegments, kColumn], y as [kOutputDim0, kColumn].
constexpr int64_t kNumSegments = 2;
constexpr int64_t kColumn = 3;
constexpr int64_t kIndicesNum = 3;
constexpr int64_t kOutputDim0 = 3;
constexpr uint64_t kXElementNum = static_cast<uint64_t>(kNumSegments * kColumn);
constexpr uint64_t kYElementNum = static_cast<uint64_t>(kOutputDim0 * kColumn);

std::shared_ptr<NodeDef> CreateNodeDef(const vector<vector<int64_t>>& shapes, const vector<DataType>& dataTypes,
                                       const vector<void*>& datas)
{
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "SparseSegmentMeanGrad", "SparseSegmentMeanGrad")
        .Input({"x", dataTypes[0], shapes[0], datas[0]})
        .Input({"indices", dataTypes[1], shapes[1], datas[1]})
        .Input({"segment_ids", dataTypes[2], shapes[2], datas[2]})
        .Input({"output_dim0", dataTypes[3], shapes[3], datas[3]})
        .Output({"y", dataTypes[4], shapes[4], datas[4]});
    return nodeDef;
}
} // namespace

class TEST_SPARSE_SEGMENT_MEAN_GRAD_UT : public testing::Test {};

// segment_ids = {0, 0, 1}: segment 0 is referenced twice (scale 0.5), segment 1 once (scale 1.0).
// indices = {0, 2, 1} scatters x[0]*0.5 to y[0], x[0]*0.5 to y[2] and x[1]*1.0 to y[1].
TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, FloatInt32Int32Success)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_INT32, DT_INT32, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {
        {kNumSegments, kColumn}, {kIndicesNum}, {kIndicesNum}, {}, {kOutputDim0, kColumn}};
    float x[kXElementNum] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    int32_t indices[kIndicesNum] = {0, 2, 1};
    int32_t segmentIds[kIndicesNum] = {0, 0, 1};
    int32_t outputDim0 = kOutputDim0;
    float y[kYElementNum] = {0.0F};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    float yExp[kYElementNum] = {0.5F, 1.0F, 1.5F, 4.0F, 5.0F, 6.0F, 0.5F, 1.0F, 1.5F};
    EXPECT_EQ(CompareResult(y, yExp, kYElementNum), true);
}

// indices = {0, 0, 1} writes y[0] twice, so the second write must accumulate instead of overwrite.
// y[2] is never referenced and must stay zero.
TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, DoubleAccumulateAndZeroFillSuccess)
{
    vector<DataType> dataTypes = {DT_DOUBLE, DT_INT32, DT_INT32, DT_INT32, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {
        {kNumSegments, kColumn}, {kIndicesNum}, {kIndicesNum}, {}, {kOutputDim0, kColumn}};
    double x[kXElementNum] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int32_t indices[kIndicesNum] = {0, 0, 1};
    int32_t segmentIds[kIndicesNum] = {0, 1, 1};
    int32_t outputDim0 = kOutputDim0;
    double y[kYElementNum] = {0.0};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    double yExp[kYElementNum] = {3.0, 4.5, 6.0, 2.0, 2.5, 3.0, 0.0, 0.0, 0.0};
    EXPECT_EQ(CompareResult(y, yExp, kYElementNum), true);
}

TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, Float16Int32Int32Success)
{
    vector<DataType> dataTypes = {DT_FLOAT16, DT_INT32, DT_INT32, DT_INT32, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {
        {kNumSegments, kColumn}, {kIndicesNum}, {kIndicesNum}, {}, {kOutputDim0, kColumn}};
    Eigen::half x[kXElementNum] = {Eigen::half(1.0F), Eigen::half(2.0F), Eigen::half(3.0F),
                                   Eigen::half(4.0F), Eigen::half(5.0F), Eigen::half(6.0F)};
    int32_t indices[kIndicesNum] = {0, 2, 1};
    int32_t segmentIds[kIndicesNum] = {0, 0, 1};
    int32_t outputDim0 = kOutputDim0;
    Eigen::half y[kYElementNum] = {Eigen::half(0.0F)};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    Eigen::half yExp[kYElementNum] = {Eigen::half(0.5F), Eigen::half(1.0F), Eigen::half(1.5F),
                                      Eigen::half(4.0F), Eigen::half(5.0F), Eigen::half(6.0F),
                                      Eigen::half(0.5F), Eigen::half(1.0F), Eigen::half(1.5F)};
    EXPECT_EQ(CompareResult(y, yExp, kYElementNum), true);
}

TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, Int32IndicesInt64SegmentIdsSuccess)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_INT32, DT_INT64, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {
        {kNumSegments, kColumn}, {kIndicesNum}, {kIndicesNum}, {}, {kOutputDim0, kColumn}};
    float x[kXElementNum] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    int32_t indices[kIndicesNum] = {0, 2, 1};
    int64_t segmentIds[kIndicesNum] = {0, 0, 1};
    int32_t outputDim0 = kOutputDim0;
    float y[kYElementNum] = {0.0F};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    float yExp[kYElementNum] = {0.5F, 1.0F, 1.5F, 4.0F, 5.0F, 6.0F, 0.5F, 1.0F, 1.5F};
    EXPECT_EQ(CompareResult(y, yExp, kYElementNum), true);
}

TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, Int64IndicesInt32SegmentIdsSuccess)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_INT64, DT_INT32, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {
        {kNumSegments, kColumn}, {kIndicesNum}, {kIndicesNum}, {}, {kOutputDim0, kColumn}};
    float x[kXElementNum] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    int64_t indices[kIndicesNum] = {0, 2, 1};
    int32_t segmentIds[kIndicesNum] = {0, 0, 1};
    int32_t outputDim0 = kOutputDim0;
    float y[kYElementNum] = {0.0F};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    float yExp[kYElementNum] = {0.5F, 1.0F, 1.5F, 4.0F, 5.0F, 6.0F, 0.5F, 1.0F, 1.5F};
    EXPECT_EQ(CompareResult(y, yExp, kYElementNum), true);
}

TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, Int64IndicesInt64SegmentIdsSuccess)
{
    vector<DataType> dataTypes = {DT_DOUBLE, DT_INT64, DT_INT64, DT_INT32, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {
        {kNumSegments, kColumn}, {kIndicesNum}, {kIndicesNum}, {}, {kOutputDim0, kColumn}};
    double x[kXElementNum] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int64_t indices[kIndicesNum] = {0, 2, 1};
    int64_t segmentIds[kIndicesNum] = {0, 0, 1};
    int32_t outputDim0 = kOutputDim0;
    double y[kYElementNum] = {0.0};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    double yExp[kYElementNum] = {0.5, 1.0, 1.5, 4.0, 5.0, 6.0, 0.5, 1.0, 1.5};
    EXPECT_EQ(CompareResult(y, yExp, kYElementNum), true);
}

TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, NullInputDataFail)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_INT32, DT_INT32, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {
        {kNumSegments, kColumn}, {kIndicesNum}, {kIndicesNum}, {}, {kOutputDim0, kColumn}};
    int32_t indices[kIndicesNum] = {0, 2, 1};
    int32_t segmentIds[kIndicesNum] = {0, 0, 1};
    int32_t outputDim0 = kOutputDim0;
    float y[kYElementNum] = {0.0F};
    vector<void*> datas = {nullptr, static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, XDtypeNotSupportFail)
{
    vector<DataType> dataTypes = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {
        {kNumSegments, kColumn}, {kIndicesNum}, {kIndicesNum}, {}, {kOutputDim0, kColumn}};
    int32_t x[kXElementNum] = {1, 2, 3, 4, 5, 6};
    int32_t indices[kIndicesNum] = {0, 2, 1};
    int32_t segmentIds[kIndicesNum] = {0, 0, 1};
    int32_t outputDim0 = kOutputDim0;
    int32_t y[kYElementNum] = {0};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, XAndYDtypeMismatchFail)
{
    vector<DataType> dataTypes = {DT_DOUBLE, DT_INT32, DT_INT32, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {
        {kNumSegments, kColumn}, {kIndicesNum}, {kIndicesNum}, {}, {kOutputDim0, kColumn}};
    double x[kXElementNum] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int32_t indices[kIndicesNum] = {0, 2, 1};
    int32_t segmentIds[kIndicesNum] = {0, 0, 1};
    int32_t outputDim0 = kOutputDim0;
    float y[kYElementNum] = {0.0F};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, OutputDim0DtypeNotInt32Fail)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_INT32, DT_INT32, DT_INT64, DT_FLOAT};
    vector<vector<int64_t>> shapes = {
        {kNumSegments, kColumn}, {kIndicesNum}, {kIndicesNum}, {}, {kOutputDim0, kColumn}};
    float x[kXElementNum] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    int32_t indices[kIndicesNum] = {0, 2, 1};
    int32_t segmentIds[kIndicesNum] = {0, 0, 1};
    int64_t outputDim0 = kOutputDim0;
    float y[kYElementNum] = {0.0F};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, XRankLessThanOneFail)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_INT32, DT_INT32, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{}, {kIndicesNum}, {kIndicesNum}, {}, {kOutputDim0, kColumn}};
    float x[kXElementNum] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    int32_t indices[kIndicesNum] = {0, 2, 1};
    int32_t segmentIds[kIndicesNum] = {0, 0, 1};
    int32_t outputDim0 = kOutputDim0;
    float y[kYElementNum] = {0.0F};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, OutputDim0NotScalarFail)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_INT32, DT_INT32, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {
        {kNumSegments, kColumn}, {kIndicesNum}, {kIndicesNum}, {1}, {kOutputDim0, kColumn}};
    float x[kXElementNum] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    int32_t indices[kIndicesNum] = {0, 2, 1};
    int32_t segmentIds[kIndicesNum] = {0, 0, 1};
    int32_t outputDim0 = kOutputDim0;
    float y[kYElementNum] = {0.0F};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, YRankLessThanOneFail)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_INT32, DT_INT32, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{kNumSegments, kColumn}, {kIndicesNum}, {kIndicesNum}, {}, {}};
    float x[kXElementNum] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    int32_t indices[kIndicesNum] = {0, 2, 1};
    int32_t segmentIds[kIndicesNum] = {0, 0, 1};
    int32_t outputDim0 = kOutputDim0;
    float y[kYElementNum] = {0.0F};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, IndicesAndSegmentIdsSizeMismatchFail)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_INT32, DT_INT32, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {
        {kNumSegments, kColumn}, {kIndicesNum}, {kIndicesNum - 1}, {}, {kOutputDim0, kColumn}};
    float x[kXElementNum] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    int32_t indices[kIndicesNum] = {0, 2, 1};
    int32_t segmentIds[kIndicesNum] = {0, 0, 1};
    int32_t outputDim0 = kOutputDim0;
    float y[kYElementNum] = {0.0F};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, XAndYRowSizeMismatchFail)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_INT32, DT_INT32, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {
        {kNumSegments, kColumn}, {kIndicesNum}, {kIndicesNum}, {}, {kOutputDim0, kColumn + 1}};
    float x[kXElementNum] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    int32_t indices[kIndicesNum] = {0, 2, 1};
    int32_t segmentIds[kIndicesNum] = {0, 0, 1};
    int32_t outputDim0 = kOutputDim0;
    float y[kOutputDim0 * (kColumn + 1)] = {0.0F};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, YDim0MismatchOutputDim0Fail)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_INT32, DT_INT32, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {
        {kNumSegments, kColumn}, {kIndicesNum}, {kIndicesNum}, {}, {kOutputDim0, kColumn}};
    float x[kXElementNum] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    int32_t indices[kIndicesNum] = {0, 2, 1};
    int32_t segmentIds[kIndicesNum] = {0, 0, 1};
    int32_t outputDim0 = kOutputDim0 - 1;
    float y[kYElementNum] = {0.0F};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, IndicesDtypeNotSupportFail)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_FLOAT, DT_INT32, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {
        {kNumSegments, kColumn}, {kIndicesNum}, {kIndicesNum}, {}, {kOutputDim0, kColumn}};
    float x[kXElementNum] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    float indices[kIndicesNum] = {0.0F, 2.0F, 1.0F};
    int32_t segmentIds[kIndicesNum] = {0, 0, 1};
    int32_t outputDim0 = kOutputDim0;
    float y[kYElementNum] = {0.0F};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, SegmentIdsDtypeNotSupportFail)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {
        {kNumSegments, kColumn}, {kIndicesNum}, {kIndicesNum}, {}, {kOutputDim0, kColumn}};
    float x[kXElementNum] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    int32_t indices[kIndicesNum] = {0, 2, 1};
    float segmentIds[kIndicesNum] = {0.0F, 0.0F, 1.0F};
    int32_t outputDim0 = kOutputDim0;
    float y[kYElementNum] = {0.0F};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, SegmentIdOutOfRangeFail)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_INT32, DT_INT32, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {
        {kNumSegments, kColumn}, {kIndicesNum}, {kIndicesNum}, {}, {kOutputDim0, kColumn}};
    float x[kXElementNum] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    int32_t indices[kIndicesNum] = {0, 2, 1};
    int32_t segmentIds[kIndicesNum] = {0, 0, static_cast<int32_t>(kNumSegments)};
    int32_t outputDim0 = kOutputDim0;
    float y[kYElementNum] = {0.0F};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, IndexOutOfRangeFail)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_INT32, DT_INT32, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {
        {kNumSegments, kColumn}, {kIndicesNum}, {kIndicesNum}, {}, {kOutputDim0, kColumn}};
    float x[kXElementNum] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    int32_t indices[kIndicesNum] = {0, 2, static_cast<int32_t>(kOutputDim0)};
    int32_t segmentIds[kIndicesNum] = {0, 0, 1};
    int32_t outputDim0 = kOutputDim0;
    float y[kYElementNum] = {0.0F};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_SEGMENT_MEAN_GRAD_UT, NegativeIndexFail)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_INT32, DT_INT32, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {
        {kNumSegments, kColumn}, {kIndicesNum}, {kIndicesNum}, {}, {kOutputDim0, kColumn}};
    float x[kXElementNum] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    int32_t indices[kIndicesNum] = {0, 2, -1};
    int32_t segmentIds[kIndicesNum] = {0, 0, 1};
    int32_t outputDim0 = kOutputDim0;
    float y[kYElementNum] = {0.0F};
    vector<void*> datas = {static_cast<void*>(x), static_cast<void*>(indices), static_cast<void*>(segmentIds),
                           static_cast<void*>(&outputDim0), static_cast<void*>(y)};
    auto nodeDef = CreateNodeDef(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}
