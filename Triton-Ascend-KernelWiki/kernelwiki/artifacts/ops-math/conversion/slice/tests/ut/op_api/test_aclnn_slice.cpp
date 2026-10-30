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
#include "../../../op_api/aclnn_slice.h"
#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_slice_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_slice_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_slice_test TearDown" << endl;
    }
};

TEST_F(l2_slice_test, test0)
{
    auto self = TensorDesc({1, 4, 2, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = 1;
    auto start = 0;
    auto end = 4;
    auto step = 1;
    auto out = TensorDesc({1, 4, 2, 2}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_float32)
{
    auto self = TensorDesc({6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = 0;
    auto start = 0;
    auto end = 3;
    auto step = 1;
    auto out = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_float16)
{
    auto self = TensorDesc({6, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = 0;
    auto start = 0;
    auto end = 3;
    auto step = 1;
    auto out = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_int64)
{
    auto self = TensorDesc({6, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = 0;
    auto start = 0;
    auto end = 3;
    auto step = 1;
    auto out = TensorDesc({3, 4}, ACL_INT64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_int32)
{
    auto self = TensorDesc({6, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = 0;
    auto start = 0;
    auto end = 3;
    auto step = 1;
    auto out = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_int8)
{
    auto self = TensorDesc({6, 4}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = 0;
    auto start = 0;
    auto end = 3;
    auto step = 1;
    auto out = TensorDesc({3, 4}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_uint8)
{
    auto self = TensorDesc({6, 4}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 10);
    auto dim = 0;
    auto start = 0;
    auto end = 3;
    auto step = 1;
    auto out = TensorDesc({3, 4}, ACL_UINT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_bool)
{
    auto self = TensorDesc({6, 4}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 5);
    auto dim = 0;
    auto start = 0;
    auto end = 3;
    auto step = 1;
    auto out = TensorDesc({3, 4}, ACL_BOOL, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_negative_start)
{
    auto self = TensorDesc({6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = 0;
    auto start = -3;
    auto end = 3;
    auto step = 1;
    auto out = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slice_test, aclnnSlice_negative_end)
{
    auto self = TensorDesc({6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = 0;
    auto start = 0;
    auto end = -1;
    auto step = 1;
    auto out = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_negative_dim)
{
    auto self = TensorDesc({6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = -1;
    auto start = 0;
    auto end = 3;
    auto step = 1;
    auto out = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slice_test, aclnnSlice_step_greater_than_1)
{
    auto self = TensorDesc({6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = 0;
    auto start = 0;
    auto end = 6;
    auto step = 2;
    auto out = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_start_exceeds_dim)
{
    auto self = TensorDesc({6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = 0;
    auto start = 10;
    auto end = 3;
    auto step = 1;
    auto out = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_end_exceeds_dim)
{
    auto self = TensorDesc({6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = 0;
    auto start = 0;
    auto end = 100;
    auto step = 1;
    auto out = TensorDesc({6, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_dim_out_of_range)
{
    auto self = TensorDesc({6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = 5;
    auto start = 0;
    auto end = 3;
    auto step = 1;
    auto out = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slice_test, aclnnSlice_step_zero_invalid)
{
    auto self = TensorDesc({6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = 0;
    auto start = 0;
    auto end = 3;
    auto step = 0;
    auto out = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slice_test, aclnnSlice_negative_step_invalid)
{
    auto self = TensorDesc({6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = 0;
    auto start = 0;
    auto end = 3;
    auto step = -1;
    auto out = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slice_test, aclnnSlice_shape_mismatch)
{
    auto self = TensorDesc({6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = 0;
    auto start = 0;
    auto end = 3;
    auto step = 1;
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slice_test, aclnnSlice_dtype_mismatch)
{
    auto self = TensorDesc({6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = 0;
    auto start = 0;
    auto end = 3;
    auto step = 1;
    auto out = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slice_test, aclnnSlice_empty_input)
{
    auto self = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = 0;
    auto start = 0;
    auto end = 0;
    auto step = 1;
    auto out = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_empty_output)
{
    auto self = TensorDesc({6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = 0;
    auto start = 0;
    auto end = 0;
    auto step = 1;
    auto out = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_single_element)
{
    auto self = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = 0;
    auto start = 0;
    auto end = 1;
    auto step = 1;
    auto out = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slice_test, aclnnSlice_multi_dim)
{
    auto self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dim = 2;
    auto start = 1;
    auto end = 3;
    auto step = 1;
    auto out = TensorDesc({2, 3, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSlice, INPUT(self, dim, start, end, step), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
