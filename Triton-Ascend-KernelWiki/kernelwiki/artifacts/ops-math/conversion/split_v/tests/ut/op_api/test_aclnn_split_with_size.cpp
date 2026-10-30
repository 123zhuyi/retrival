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

#include "conversion/split_v/op_api/aclnn_split_with_size.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_split_with_size_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "split_with_size_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "split_with_size_test TearDown" << endl;
    }
};

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_20_4_float32_dim_0)
{
    const vector<int64_t>& self_shape = {20, 4};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{15, 5});
    int64_t dim = 0;

    auto out1 = TensorDesc({15, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out2 = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1, out2});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_negative_dim)
{
    const vector<int64_t>& self_shape = {8, 16, 4};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{4, 4, 4, 4});
    int64_t dim = -2;

    auto out1 = TensorDesc({8, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out2 = TensorDesc({8, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out3 = TensorDesc({8, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out4 = TensorDesc({8, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1, out2, out3, out4});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_single_split)
{
    const vector<int64_t>& self_shape = {8, 4};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{8});
    int64_t dim = 0;

    auto out1 = TensorDesc({8, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_empty_tensor)
{
    const vector<int64_t>& self_shape = {0, 4};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto split_size = IntArrayDesc(vector<int64_t>{0, 0});
    int64_t dim = 0;

    auto out1 = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out2 = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1, out2});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_float16)
{
    const vector<int64_t>& self_shape = {12, 8};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{3, 3, 3, 3});
    int64_t dim = 0;

    auto out1 = TensorDesc({3, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto out2 = TensorDesc({3, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto out3 = TensorDesc({3, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto out4 = TensorDesc({3, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto out_list = TensorListDesc({out1, out2, out3, out4});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_int32)
{
    const vector<int64_t>& self_shape = {10, 5};
    auto self_desc = TensorDesc(self_shape, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto split_size = IntArrayDesc(vector<int64_t>{2, 3, 5});
    int64_t dim = 0;

    auto out1 = TensorDesc({2, 5}, ACL_INT32, ACL_FORMAT_ND);
    auto out2 = TensorDesc({3, 5}, ACL_INT32, ACL_FORMAT_ND);
    auto out3 = TensorDesc({5, 5}, ACL_INT32, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1, out2, out3});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_int64)
{
    const vector<int64_t>& self_shape = {6, 6};
    auto self_desc = TensorDesc(self_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto split_size = IntArrayDesc(vector<int64_t>{2, 2, 2});
    int64_t dim = 1;

    auto out1 = TensorDesc({6, 2}, ACL_INT64, ACL_FORMAT_ND);
    auto out2 = TensorDesc({6, 2}, ACL_INT64, ACL_FORMAT_ND);
    auto out3 = TensorDesc({6, 2}, ACL_INT64, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1, out2, out3});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_3d_tensor)
{
    const vector<int64_t>& self_shape = {4, 12, 8};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{3, 3, 3, 3});
    int64_t dim = 1;

    auto out1 = TensorDesc({4, 3, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out2 = TensorDesc({4, 3, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out3 = TensorDesc({4, 3, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out4 = TensorDesc({4, 3, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1, out2, out3, out4});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_4d_tensor)
{
    const vector<int64_t>& self_shape = {2, 4, 8, 16};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{4, 4});
    int64_t dim = 2;

    auto out1 = TensorDesc({2, 4, 4, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out2 = TensorDesc({2, 4, 4, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1, out2});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_invalid_dim_exceeds)
{
    const vector<int64_t>& self_shape = {4, 4};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{2, 2});
    int64_t dim = 2;

    auto out1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out2 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1, out2});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_invalid_dim_negative_exceeds)
{
    const vector<int64_t>& self_shape = {4, 4};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{2, 2});
    int64_t dim = -3;

    auto out1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out2 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1, out2});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_invalid_splitSize_sum_mismatch)
{
    const vector<int64_t>& self_shape = {8, 4};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{3, 3});
    int64_t dim = 0;

    auto out1 = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out2 = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1, out2});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_invalid_splitSize_outNum_mismatch)
{
    const vector<int64_t>& self_shape = {8, 4};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{4, 4});
    int64_t dim = 0;

    auto out1 = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_list = TensorListDesc({out1});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_bf16)
{
    const vector<int64_t>& self_shape = {12, 6};
    auto self_desc = TensorDesc(self_shape, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto split_size = IntArrayDesc(vector<int64_t>{4, 4, 4});
    int64_t dim = 0;

    auto out1 = TensorDesc({4, 6}, ACL_BF16, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto out2 = TensorDesc({4, 6}, ACL_BF16, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto out3 = TensorDesc({4, 6}, ACL_BF16, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto out_list = TensorListDesc({out1, out2, out3});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_split_with_size_test, aclnnSplitWithSize_bool)
{
    const vector<int64_t>& self_shape = {4, 4};
    auto self_desc = TensorDesc(self_shape, ACL_BOOL, ACL_FORMAT_ND);
    auto split_size = IntArrayDesc(vector<int64_t>{2, 2});
    int64_t dim = 0;

    auto out1 = TensorDesc({2, 4}, ACL_BOOL, ACL_FORMAT_ND);
    auto out2 = TensorDesc({2, 4}, ACL_BOOL, ACL_FORMAT_ND);
    auto out_list = TensorListDesc({out1, out2});

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(self_desc, split_size, dim), OUTPUT(out_list));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}