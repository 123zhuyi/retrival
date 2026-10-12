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
#include "../../../op_api/aclnn_max.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_max_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_max_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_max_test TearDown" << endl; }
};

// self为空指针
TEST_F(l2_max_test, l2_max_test_nullptr_self)
{
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMax, INPUT(nullptr), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// out为空指针
TEST_F(l2_max_test, l2_max_test_nullptr_out)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMax, INPUT(selfDesc), OUTPUT(nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 正常路径，float32
TEST_F(l2_max_test, l2_max_test_dtype_float32)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMax, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    // ut.TestPrecision();
}

// 正常路径，float16
TEST_F(l2_max_test, l2_max_test_dtype_float16)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto outDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnMax, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    // ut.TestPrecision();
}

// 正常路径，int64
TEST_F(l2_max_test, l2_max_test_dtype_int64)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto outDesc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMax, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    // ut.TestPrecision();
}

// 正常路径，int32
TEST_F(l2_max_test, l2_max_test_dtype_int32)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto outDesc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMax, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    // ut.TestPrecision();
}

// 正常路径，3维输入
TEST_F(l2_max_test, l2_max_test_3d)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMax, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    // ut.TestPrecision();
}

// self为空tensor
TEST_F(l2_max_test, l2_max_test_empty_self)
{
    auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMax, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// 用例不支持，最大维度超过8
TEST_F(l2_max_test, l2_max_test_dimension)
{
    auto selfDesc = TensorDesc({2, 2, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMax, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，bool类型
TEST_F(l2_max_test, l2_max_test_dtype_bool)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_BOOL, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMax, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    // ut.TestPrecision();
}
