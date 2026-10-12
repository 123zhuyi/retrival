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
#include "../../../op_api/aclnn_max_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_max_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_max_v2_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_max_v2_test TearDown" << endl; }
};

// self为空指针
TEST_F(l2_max_v2_test, l2_max_v2_test_nullptr_self)
{
    auto outDesc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{0});
    bool keepDims = true;
    bool noopWithEmptyDims = false;

    auto ut = OP_API_UT(aclnnMaxV2, INPUT(nullptr, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// dims为空指针
TEST_F(l2_max_v2_test, l2_max_v2_test_nullptr_dims)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    bool keepDims = true;
    bool noopWithEmptyDims = false;

    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, nullptr, keepDims, noopWithEmptyDims), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// out为空指针
TEST_F(l2_max_v2_test, l2_max_v2_test_nullptr_out)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{0});
    bool keepDims = true;
    bool noopWithEmptyDims = false;

    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 正常路径，float32
TEST_F(l2_max_v2_test, l2_max_v2_test_dtype_float32)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto outDesc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto dims = IntArrayDesc(vector<int64_t>{0});
    bool keepDims = true;
    bool noopWithEmptyDims = false;

    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    // ut.TestPrecision();
}

// 正常路径，float16
TEST_F(l2_max_v2_test, l2_max_v2_test_dtype_float16)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto outDesc = TensorDesc({1, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto dims = IntArrayDesc(vector<int64_t>{0});
    bool keepDims = true;
    bool noopWithEmptyDims = false;

    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    // ut.TestPrecision();
}

// 正常路径，int64
TEST_F(l2_max_v2_test, l2_max_v2_test_dtype_int64)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto outDesc = TensorDesc({1, 4}, ACL_INT64, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{0});
    bool keepDims = true;
    bool noopWithEmptyDims = false;

    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    // ut.TestPrecision();
}

// 正常路径，dim为1
TEST_F(l2_max_v2_test, l2_max_v2_test_dim_1)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto outDesc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto dims = IntArrayDesc(vector<int64_t>{1});
    bool keepDims = true;
    bool noopWithEmptyDims = false;

    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    // ut.TestPrecision();
}

// 正常路径，keepDims为false
TEST_F(l2_max_v2_test, l2_max_v2_test_keepdim_false)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto outDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto dims = IntArrayDesc(vector<int64_t>{0});
    bool keepDims = false;
    bool noopWithEmptyDims = false;

    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    // ut.TestPrecision();
}

// 正常路径，dims为空且noopWithEmptyDims为true
TEST_F(l2_max_v2_test, l2_max_v2_test_empty_dims_noop)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto dims = IntArrayDesc(vector<int64_t>{});
    bool keepDims = false;
    bool noopWithEmptyDims = true;

    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    // ut.TestPrecision();
}

// self为空tensor
TEST_F(l2_max_v2_test, l2_max_v2_test_empty_self)
{
    auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{0});
    bool keepDims = true;
    bool noopWithEmptyDims = false;

    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// 用例不支持，最大维度超过8
TEST_F(l2_max_v2_test, l2_max_v2_test_dimension)
{
    auto selfDesc = TensorDesc({2, 2, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 2, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{0});
    bool keepDims = true;
    bool noopWithEmptyDims = false;

    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 异常路径，dim值不在self范围内
TEST_F(l2_max_v2_test, l2_max_v2_dim_invalid)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dims = IntArrayDesc(vector<int64_t>{2});
    bool keepDims = true;
    bool noopWithEmptyDims = false;

    auto ut = OP_API_UT(aclnnMaxV2, INPUT(selfDesc, dims, keepDims, noopWithEmptyDims), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}
