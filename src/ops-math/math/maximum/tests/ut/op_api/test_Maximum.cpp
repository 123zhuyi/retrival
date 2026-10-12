/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_maximum.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class l2_maximum_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "maximum_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "maximum_test TearDown" << endl; }
};

TEST_F(l2_maximum_test, aclnnMaximum_001_aclnnMaximum_1_4_3_2_1_6_5_1_float_nd_2_1_3_2_1_1_5_5_float_nd)
{
    // left input
    const vector<int64_t>& selfShape = {1, 4, 3, 2, 1, 6, 5, 1};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {2, 1, 3, 2, 1, 1, 5, 5};
    aclDataType otherDtype = ACL_FLOAT;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {2, 4, 3, 2, 1, 6, 5, 5};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_maximum_test, aclnnMaximum_002_aclnnMaximum_223_32_17_17_float16_nchw_223_1_1_17_float16_nchw)
{
    // left input
    const vector<int64_t>& selfShape = {223, 32, 17, 17};
    aclDataType selfDtype = ACL_FLOAT16;
    aclFormat selfFormat = ACL_FORMAT_NCHW;
    // right input
    const vector<int64_t>& otherShape = {223, 1, 1, 17};
    aclDataType otherDtype = ACL_FLOAT16;
    aclFormat otherFormat = ACL_FORMAT_NCHW;
    // output
    const vector<int64_t>& outShape = {223, 32, 17, 17};
    aclDataType outDtype = ACL_FLOAT16;
    aclFormat outFormat = ACL_FORMAT_NCHW;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_maximum_test, aclnnMaximum_003_aclnnMaximum_64_31_31_16_uint8_nhwc_64_31_1_1_uint8_nhwc_float16)
{
    // left input
    const vector<int64_t>& selfShape = {64, 31, 31, 16};
    aclDataType selfDtype = ACL_UINT8;
    aclFormat selfFormat = ACL_FORMAT_NHWC;
    // right input
    const vector<int64_t>& otherShape = {64, 31, 1, 1};
    aclDataType otherDtype = ACL_UINT8;
    aclFormat otherFormat = ACL_FORMAT_NHWC;
    // output
    const vector<int64_t>& outShape = {64, 31, 31, 16};
    aclDataType outDtype = ACL_FLOAT16;
    aclFormat outFormat = ACL_FORMAT_NHWC;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_maximum_test, aclnnMaximum_004_aclnnMaximum_1_1_16_129_int8_hwcn_32_32_16_1_int8_hwcn_int32)
{
    // left input
    const vector<int64_t>& selfShape = {1, 1, 16, 129};
    aclDataType selfDtype = ACL_INT8;
    aclFormat selfFormat = ACL_FORMAT_HWCN;
    // right input
    const vector<int64_t>& otherShape = {32, 32, 16, 1};
    aclDataType otherDtype = ACL_INT8;
    aclFormat otherFormat = ACL_FORMAT_HWCN;
    // output
    const vector<int64_t>& outShape = {32, 32, 16, 129};
    aclDataType outDtype = ACL_INT32;
    aclFormat outFormat = ACL_FORMAT_HWCN;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_maximum_test, aclnnMaximum_005_aclnnMaximum_1_5_1_16_129_int16_ndhwc_32_5_32_16_1_int16_ndhwc_float32)
{
    // left input
    const vector<int64_t>& selfShape = {1, 5, 1, 16, 129};
    aclDataType selfDtype = ACL_INT16;
    aclFormat selfFormat = ACL_FORMAT_NDHWC;
    // right input
    const vector<int64_t>& otherShape = {32, 5, 32, 16, 1};
    aclDataType otherDtype = ACL_INT16;
    aclFormat otherFormat = ACL_FORMAT_NDHWC;
    // output
    const vector<int64_t>& outShape = {32, 5, 32, 16, 129};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_NDHWC;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_maximum_test, aclnnMaximum_005_1_aclnnMaximum_1_5_1_16_129_int32_ndhwc_32_5_32_16_1_int32_ndhwc)
{
    // left input
    const vector<int64_t>& selfShape = {1, 5, 1, 16, 129};
    aclDataType selfDtype = ACL_INT32;
    aclFormat selfFormat = ACL_FORMAT_NDHWC;
    // right input
    const vector<int64_t>& otherShape = {32, 5, 32, 16, 1};
    aclDataType otherDtype = ACL_INT32;
    aclFormat otherFormat = ACL_FORMAT_NDHWC;
    // output
    const vector<int64_t>& outShape = {32, 5, 32, 16, 129};
    aclDataType outDtype = ACL_INT32;
    aclFormat outFormat = ACL_FORMAT_NDHWC;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();  // comment bcz of timeout in model tests (222507 ms)
}

TEST_F(l2_maximum_test, aclnnMaximum_006_aclnnMaximum_23_71_16_5_1_int64_ncdhw_1_1_16_5_23_int64_ndhwc)
{
    // left input
    const vector<int64_t>& selfShape = {23, 71, 16, 5, 1};
    aclDataType selfDtype = ACL_INT64;
    aclFormat selfFormat = ACL_FORMAT_NCDHW;
    // right input
    const vector<int64_t>& otherShape = {1, 1, 16, 5, 23};
    aclDataType otherDtype = ACL_INT64;
    aclFormat otherFormat = ACL_FORMAT_NCDHW;
    // output
    const vector<int64_t>& outShape = {23, 71, 16, 5, 23};
    aclDataType outDtype = ACL_INT64;
    aclFormat outFormat = ACL_FORMAT_NCDHW;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();  // comment bcz of timeout in model tests (294524 ms)
}

TEST_F(l2_maximum_test, aclnnMaximum_006_1_aclnnMaximum_1_11_21_6_float32_NHWC_12_1_1_1_float16_NHWC)
{
    // left input
    const vector<int64_t>& selfShape = {1, 11, 21, 16};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_NHWC;
    // right input
    const vector<int64_t>& otherShape = {12, 1, 1, 1};
    aclDataType otherDtype = ACL_FLOAT16;
    aclFormat otherFormat = ACL_FORMAT_NHWC;
    // output
    const vector<int64_t>& outShape = {12, 11, 21, 16};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_NHWC;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_maximum_test, aclnnMaximum_007_aclnnMaximum_92_111_1_7_3_1_6_nd_92_111_1_7_3_1_6_bool_nd)
{
    // left input
    const vector<int64_t>& selfShape = {92, 111, 1, 7, 3, 1, 6};
    aclDataType selfDtype = ACL_BOOL;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {92, 111, 1, 7, 3, 1, 6};
    aclDataType otherDtype = ACL_BOOL;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {92, 111, 1, 7, 3, 1, 6};
    aclDataType outDtype = ACL_BOOL;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_maximum_test, aclnnMaximum_008_aclnnMaximum_92_111_1_7_3_1_6_bool_nd_92_111_1_7_3_1_6_int32_nd_int64)
{
    // left input
    const vector<int64_t>& selfShape = {92, 111, 1, 7, 3, 1, 6};
    aclDataType selfDtype = ACL_BOOL;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {92, 111, 1, 7, 3, 1, 6};
    aclDataType otherDtype = ACL_INT32;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {92, 111, 1, 7, 3, 1, 6};
    aclDataType outDtype = ACL_INT64;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision(); // comment bcz of timeout in model tests (73849 ms)
}

TEST_F(l2_maximum_test, aclnnMaximum_009_aclnnMaximum_1_0_1_2_float32_nd_2_1_0_1_float32_nd_empty_tensor)
{
    // left input
    const vector<int64_t>& selfShape = {1, 0, 1, 2};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {2, 1, 0, 1};
    aclDataType otherDtype = ACL_FLOAT;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {2, 0, 0, 2};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_maximum_test, aclnnMaximum_010_aclnnMaximum_100_int8_100_int8_in_overflow)
{
    // left input
    const vector<int64_t>& selfShape = {100};
    aclDataType selfDtype = ACL_INT8;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {100};
    aclDataType otherDtype = ACL_INT8;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {100};
    aclDataType outDtype = ACL_INT8;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(257, 300);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(1, 1);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_maximum_test, aclnnMaximum_011_aclnnMaximum_1_2_float16_2_1_float16_boundary_value)
{
    // left input
    const vector<int64_t>& selfShape = {1, 2};
    aclDataType selfDtype = ACL_FLOAT16;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {2, 1};
    aclDataType otherDtype = ACL_FLOAT16;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {2, 2};
    aclDataType outDtype = ACL_FLOAT16;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<float>{65504.0, -65504.0});
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).Value(vector<float>{1, -1});
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_maximum_test, aclnnMaximum_011_1_aclnnMaximum_float16_range)
{
    // left input
    const vector<int64_t>& selfShape = {5, 2, 3, 6};
    aclDataType selfDtype = ACL_FLOAT16;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {5, 2, 1, 6};
    aclDataType otherDtype = ACL_FLOAT16;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {5, 2, 3, 6};
    aclDataType outDtype = ACL_FLOAT16;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(-1, 1);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(-1, 1);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_maximum_test, aclnnMaximum_011_2_aclnnMaximum_float32_range)
{
    // left input
    const vector<int64_t>& selfShape = {8, 2, 3, 6};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {1, 2, 1, 6};
    aclDataType otherDtype = ACL_FLOAT;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {8, 2, 3, 6};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(-1, 1);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat).ValueRange(-1, 1);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// TODO: special data value
// TEST_F(l2_maximum_test, aclnnMaximum_012_aclnnMaximum_6_1_float32_1_6_float32_special_value) {
//   // left input
//   const vector<int64_t>& selfShape = {6,1};
//   aclDataType selfDtype = ACL_FLOAT;
//   aclFormat selfFormat = ACL_FORMAT_ND;
//   // right input
//   const vector<int64_t>& otherShape = {1,6};
//   aclDataType otherDtype = ACL_FLOAT;
//   aclFormat otherFormat = ACL_FORMAT_ND;
//   // output
//   const vector<int64_t>& outShape = {6,6};
//   aclDataType outDtype = ACL_FLOAT;
//   aclFormat outFormat = ACL_FORMAT_ND;

//   auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat)
//                               .Value(vector<float>{float("inf"), float("-inf"), float("nan"), 0, 1, -1});
//   auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat)
//                               .Value(vector<float>{float("inf"), float("-inf"), float("nan"), 0, 1, -1});
//   auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

//   auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   // SAMPLE: precision simulate
//   ut.TestPrecision();
// }

TEST_F(l2_maximum_test, aclnnMaximum_013_1_aclnnMaximum_5_4_float32_input_not_contiguous)
{
    auto selfTensorDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto otherTensorDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto outTensorDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_maximum_test, aclnnMaximum_013_aclnnMaximum_2_3_4_float64)
{
    // left input
    const vector<int64_t>& selfShape = {2, 3, 4};
    aclDataType selfDtype = ACL_DOUBLE;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {2, 3, 4};
    aclDataType otherDtype = ACL_DOUBLE;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {2, 3, 4};
    aclDataType outDtype = ACL_DOUBLE;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_maximum_test, aclnnMaximum_014_1_aclnnMaximum_5_4_float32_out_not_contiguous)
{
    auto selfTensorDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto otherTensorDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto outTensorDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_maximum_test, ascend910B2_aclnnMaximum_014_aclnnMaximum_2_3_4_bfloat16)
{
    // left input
    const vector<int64_t>& selfShape = {2, 3, 4};
    aclDataType selfDtype = ACL_BF16;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {2, 3, 4};
    aclDataType otherDtype = ACL_BF16;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {2, 3, 4};
    aclDataType outDtype = ACL_BF16;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_maximum_test, ascend910B2_case_float_bf16)
{
    auto selfTensorDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto otherTensorDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outTensorDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_maximum_test, case_bf16_910)
{
    auto selfTensorDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto otherTensorDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outTensorDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_maximum_test, case_flaot_bf16_910)
{
    auto selfTensorDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto otherTensorDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outTensorDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_maximum_test, aclnnMaximum_015_aclnnMaximum_input_out_nullptr)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut_l = OP_API_UT(aclnnMaximum, INPUT(nullptr, tensor_desc), OUTPUT(tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut_l.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_r = OP_API_UT(aclnnMaximum, INPUT(tensor_desc, nullptr), OUTPUT(tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_r.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_o = OP_API_UT(aclnnMaximum, INPUT(tensor_desc, tensor_desc), OUTPUT(nullptr));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_o.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_maximum_test, aclnnMaximum_016_aclnnMaximum_input_error_shape)
{
    // left input
    const vector<int64_t>& selfShape = {123, 11, 2};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {123, 8, 2};
    aclDataType otherDtype = ACL_FLOAT;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {123, 11, 2};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_maximum_test, aclnnMaximum_017_aclnnMaximum_input_error_shape_and_empty_tensor)
{
    // left input
    const vector<int64_t>& selfShape = {123, 0, 2};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {1, 8, 2};
    aclDataType otherDtype = ACL_FLOAT;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {123, 0, 2};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_maximum_test, aclnnMaximum_018_aclnnMaximum_input_out_diff_shape)
{
    // left input
    const vector<int64_t>& selfShape = {123, 11, 2};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {123, 11, 2};
    aclDataType otherDtype = ACL_FLOAT;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {123, 8, 2};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_maximum_test, aclnnMaximum_022_aclnnMaximum_out_error_cast_dtype)
{
    // left input
    const vector<int64_t>& selfShape = {6, 12, 11, 1};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_NHWC;
    // right input
    const vector<int64_t>& otherShape = {6, 12, 11, 1};
    aclDataType otherDtype = ACL_FLOAT;
    aclFormat otherFormat = ACL_FORMAT_NHWC;
    // output
    const vector<int64_t>& outShape = {6, 12, 11, 1};
    aclDataType outDtype = ACL_INT32;
    aclFormat outFormat = ACL_FORMAT_NHWC;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_maximum_test, aclnnMaximum_022_1_aclnnMaximum_ascend950_out_dtype_compute)
{
    auto curSocVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    if (curSocVersion != op::SocVersion::ASCEND950) {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }

    const vector<int64_t>& selfShape = {2, 3};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    const vector<int64_t>& otherShape = {2, 3};
    aclDataType otherDtype = ACL_FLOAT;
    aclFormat otherFormat = ACL_FORMAT_ND;
    const vector<int64_t>& outShape = {2, 3};
    aclDataType outDtype = ACL_INT32;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc =
        TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<float>{1.2, -3.4, 5.6, 7.8, -9.1, 2.3});
    auto otherTensorDesc =
        TensorDesc(otherShape, otherDtype, otherFormat).Value(vector<float>{0.5, -4.5, 6.1, 7.2, -8.8, 3.9});
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    if (curSocVersion != op::SocVersion::ASCEND950) {
        op::SetPlatformSocVersion(curSocVersion);
    }
}

TEST_F(l2_maximum_test, aclnnMaximum_023_aclnnMaximum_error_input_dtype)
{
    // left input
    const vector<int64_t>& selfShape = {6, 12, 11, 16};
    aclDataType selfDtype = ACL_UINT64;
    aclFormat selfFormat = ACL_FORMAT_NHWC;
    // right input
    const vector<int64_t>& otherShape = {6, 12, 11, 1};
    aclDataType otherDtype = ACL_UINT64;
    aclFormat otherFormat = ACL_FORMAT_NHWC;
    // output
    const vector<int64_t>& outShape = {6, 12, 11, 16};
    aclDataType outDtype = ACL_UINT64;
    aclFormat outFormat = ACL_FORMAT_NHWC;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_maximum_test, aclnnMaximum_024_aclnnMaximum_input_error_shape_len)
{
    // left input
    const vector<int64_t>& selfShape = {2, 3, 4, 5, 6, 7, 8, 9, 10};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& otherShape = {2, 3, 4, 5, 6, 7, 8, 9, 1};
    aclDataType otherDtype = ACL_FLOAT;
    aclFormat otherFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {2, 3, 4, 5, 6, 7, 8, 9, 10};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// TODO: COMPLEX64 NOT SUPPORT NOW
//  TEST_F(l2_maximum_test, aclnnMaximum_025_aclnnMaximum_23_71_complex64_nd_23_1_complex64_nd) {
//    // left input
//    const vector<int64_t>& selfShape = {23,71};
//    aclDataType selfDtype = ACL_COMPLEX64;
//    aclFormat selfFormat = ACL_FORMAT_ND;
//    // right input
//    const vector<int64_t>& otherShape = {23,1};
//    aclDataType otherDtype = ACL_COMPLEX64;
//    aclFormat otherFormat = ACL_FORMAT_ND;
//    // output
//    const vector<int64_t>& outShape = {23,71};
//    aclDataType outDtype = ACL_COMPLEX64;
//    aclFormat outFormat = ACL_FORMAT_ND;

//   auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
//   auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
//   auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

//   auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspaceSize = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   // SAMPLE: precision simulate
//   ut.TestPrecision();
// }

// TODO: COMPLEX128 NOT SUPPORT NOW
//  TEST_F(l2_maximum_test, aclnnMaximum_026_aclnnMaximum_1_65_23_complex128_nd_55_1_1_complex128_nd) {
//    // left input
//    const vector<int64_t>& selfShape = {1,65,23};
//    aclDataType selfDtype = ACL_COMPLEX128;
//    aclFormat selfFormat = ACL_FORMAT_ND;
//    // right input
//    const vector<int64_t>& otherShape = {55,1,1};
//    aclDataType otherDtype = ACL_COMPLEX128;
//    aclFormat otherFormat = ACL_FORMAT_ND;
//    // output
//    const vector<int64_t>& outShape = {55,65,23};
//    aclDataType outDtype = ACL_COMPLEX128;
//    aclFormat outFormat = ACL_FORMAT_ND;

//   auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
//   auto otherTensorDesc = TensorDesc(otherShape, otherDtype, otherFormat);
//   auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

//   auto ut = OP_API_UT(aclnnMaximum, INPUT(selfTensorDesc, otherTensorDesc), OUTPUT(outTensorDesc));
//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspaceSize = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   // SAMPLE: precision simulate
//   ut.TestPrecision();
// }
