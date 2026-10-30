/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "platform/platform_info.h"
#include "../../../op_api/aclnn_roll.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/array_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_roll_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "roll_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "roll_test TearDown" << std::endl;
    }
};

TEST_F(l2_roll_test, test_dtype_float)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_dtype_float16)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_dtype_int32)
{
    auto input =
        TensorDesc({4, 3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_dtype_int64)
{
    auto input =
        TensorDesc({4, 3}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_dtype_int8)
{
    auto input =
        TensorDesc({4, 3}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_dtype_uint8)
{
    auto input =
        TensorDesc({4, 3}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_dtype_uint32)
{
    auto input =
        TensorDesc({4, 3}, ACL_UINT32, ACL_FORMAT_ND).Value(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_UINT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_dtype_bool)
{
    auto input = TensorDesc({4, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<int32_t>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_format_nd)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_format_nchw)
{
    auto input = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_format_nhwc)
{
    auto input = TensorDesc({2, 4, 5, 3}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({2, 4, 5, 3}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_shift_dim_0)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_shift_dim_1)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{1});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    getWorkspaceResult = 0;
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_shift_negative_dim)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{-1});
    auto dims = IntArrayDesc(vector<int64_t>{-1});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    getWorkspaceResult = 0;
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_multiple_shifts_dims)
{
    auto input = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto shifts = IntArrayDesc(vector<int64_t>{1, 2});
    auto dims = IntArrayDesc(vector<int64_t>{0, 2});
    auto output = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    getWorkspaceResult = 0;
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_shift_larger_than_dim)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{5});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_empty_dims_single_shift)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{2});
    auto dims = IntArrayDesc(vector<int64_t>{});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_1d_tensor)
{
    auto input = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto shifts = IntArrayDesc(vector<int64_t>{2});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_3d_tensor)
{
    auto input = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{1});
    auto output = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    getWorkspaceResult = 0;
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_4d_tensor)
{
    auto input = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{2});
    auto output = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    getWorkspaceResult = 0;
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_5d_tensor)
{
    auto input = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_8d_tensor)
{
    auto input = TensorDesc({1, 2, 1, 2, 1, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({1, 2, 1, 2, 1, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_0d_tensor)
{
    auto input = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{50});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{});
    auto output = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_empty_tensor)
{
    auto input = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_single_element_tensor)
{
    auto input = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1.0});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_shift_zero)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{0});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_special_2d_1xN)
{
    auto input = TensorDesc({1, 6}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto shifts = IntArrayDesc(vector<int64_t>{2});
    auto dims = IntArrayDesc(vector<int64_t>{1});
    auto output = TensorDesc({1, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_special_2d_Nx1)
{
    auto input = TensorDesc({6, 1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto shifts = IntArrayDesc(vector<int64_t>{2});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({6, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_non_contiguous)
{
    auto input = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {5, 5});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_invalid_dims_size_not_match_shifts)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{1, 2});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roll_test, test_invalid_dims_out_of_range)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{5});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roll_test, test_invalid_dims_out_of_range_negative)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{-5});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roll_test, test_invalid_dims_size_not_zero_with_empty_shifts)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roll_test, test_invalid_dims_size_not_1_with_empty_shifts)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{1, 2});
    auto dims = IntArrayDesc(vector<int64_t>{});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roll_test, test_invalid_shape_mismatch)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roll_test, test_invalid_dtype_mismatch)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roll_test, test_invalid_dim_gt_8)
{
    auto input = TensorDesc({1, 2, 3, 1, 2, 3, 1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({1, 2, 3, 1, 2, 3, 1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roll_test, test_nullptr_x)
{
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(nullptr, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_roll_test, test_nullptr_shifts)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, nullptr, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_roll_test, test_nullptr_dims)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, nullptr), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_roll_test, test_nullptr_out)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_roll_test, test_0d_tensor_invalid_dims)
{
    auto input = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{5.0});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roll_test, test_value_range_negative_to_positive)
{
    auto input = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_value_range_0_to_1)
{
    auto input = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_large_tensor)
{
    auto input = TensorDesc({100, 100}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto shifts = IntArrayDesc(vector<int64_t>{50});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({100, 100}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_dim_1_wrap)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{3});
    auto dims = IntArrayDesc(vector<int64_t>{1});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    getWorkspaceResult = 0;
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_wrap_dim_1)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{1});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    getWorkspaceResult = 0;
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_2d_squeeze_case)
{
    auto input = TensorDesc({1, 6}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto shifts = IntArrayDesc(vector<int64_t>{2});
    auto dims = IntArrayDesc(vector<int64_t>{1});
    auto output = TensorDesc({1, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_axis_0_direct_roll)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{2});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_2d_1x1_tensor)
{
    auto input = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1.0});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{1});
    auto output = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_negative_shift_large)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{-5});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_wrapdim_negative_dim)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{-2});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_0d_tensor_invalid_shifts)
{
    auto input = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{5.0});
    auto shifts = IntArrayDesc(vector<int64_t>{1, 2});
    auto dims = IntArrayDesc(vector<int64_t>{});
    auto output = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_roll_test, test_multi_dim_all_negative)
{
    auto input = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto shifts = IntArrayDesc(vector<int64_t>{-1, -2});
    auto dims = IntArrayDesc(vector<int64_t>{-1, -2});
    auto output = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    getWorkspaceResult = 0;
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_mixed_positive_negative_shifts)
{
    auto input = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto shifts = IntArrayDesc(vector<int64_t>{1, -2});
    auto dims = IntArrayDesc(vector<int64_t>{0, 2});
    auto output = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    getWorkspaceResult = 0;
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_2d_squeeze_multiple_shift)
{
    auto input = TensorDesc({1, 6}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto shifts = IntArrayDesc(vector<int64_t>{3});
    auto dims = IntArrayDesc(vector<int64_t>{1});
    auto output = TensorDesc({1, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_large_2d_tensor)
{
    auto input = TensorDesc({1000, 100}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto shifts = IntArrayDesc(vector<int64_t>{50});
    auto dims = IntArrayDesc(vector<int64_t>{1});
    auto output = TensorDesc({1000, 100}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    getWorkspaceResult = 0;
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_6d_tensor)
{
    auto input = TensorDesc({1, 2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{3});
    auto output = TensorDesc({1, 2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    getWorkspaceResult = 0;
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_7d_tensor)
{
    auto input = TensorDesc({1, 1, 2, 2, 3, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto shifts = IntArrayDesc(vector<int64_t>{2});
    auto dims = IntArrayDesc(vector<int64_t>{5});
    auto output = TensorDesc({1, 1, 2, 2, 3, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    getWorkspaceResult = 0;
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_contiguous_false)
{
    auto input = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND, {1, 6}, 0, {6, 6});
    auto shifts = IntArrayDesc(vector<int64_t>{2});
    auto dims = IntArrayDesc(vector<int64_t>{1});
    auto output = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    getWorkspaceResult = 0;
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_shift_equal_to_dim)
{
    auto input =
        TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto shifts = IntArrayDesc(vector<int64_t>{4});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_roll_test, test_invalid_format_mismatch)
{
    auto input = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto shifts = IntArrayDesc(vector<int64_t>{1});
    auto dims = IntArrayDesc(vector<int64_t>{0});
    auto output = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto ut = OP_API_UT(aclnnRoll, INPUT(input, shifts, dims), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, 0);
}