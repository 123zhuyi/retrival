/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "math/square/op_api/aclnn_square.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/inner/types.h"
#include "opdev/platform.h"

using namespace std;

class l2_square_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
        cout << "l2_square_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_square_test TearDown" << endl;
    }
};

// test nullptr
TEST_F(l2_square_test, ascend950_case_nullptr)
{
    auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(nullptr), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut1 = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(nullptr));
    aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 非法dype：不支持的dtype
TEST_F(l2_square_test, ascend950_case_dtype_invalid_0)
{
    auto self = TensorDesc({2, 2}, ACL_INT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法dtype：dtype不一致
TEST_F(l2_square_test, ascend950_case_dtype_invalid_1)
{
    auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法format：format超8维
TEST_F(l2_square_test, ascend950_case_format_invalid_0)
{
    auto self = TensorDesc({1, 2, 1, 2, 1, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 2, 1, 2, 1, 2, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法format：format不一致
TEST_F(l2_square_test, ascend950_case_format_invalid_1)
{
    auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape：shape不一致
TEST_F(l2_square_test, ascend950_case_shape_invalid)
{
    auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常场景：float类型
TEST_F(l2_square_test, ascend950_case_dtype_float)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：float16类型
TEST_F(l2_square_test, ascend950_case_dtype_float16)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：int32类型
TEST_F(l2_square_test, ascend950_case_dtype_int32)
{
    auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：int64类型
TEST_F(l2_square_test, ascend950_case_dtype_int64)
{
    auto self = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：bfloat16类型
TEST_F(l2_square_test, ascend950_case_dtype_bfloat16)
{
    auto self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空tensor场景
TEST_F(l2_square_test, ascend950_case_empty_tensor)
{
    auto self = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 非连续tensor场景
TEST_F(l2_square_test, ascend950_case_not_contiguous)
{
    auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {2, 2}).ValueRange(-2, 2);
    auto out = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 1D tensor
TEST_F(l2_square_test, ascend950_case_shape_1D)
{
    auto self = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 3D tensor
TEST_F(l2_square_test, ascend950_case_shape_3D)
{
    auto self = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 4D tensor
TEST_F(l2_square_test, ascend950_case_shape_4D)
{
    auto self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 5D tensor
TEST_F(l2_square_test, ascend950_case_shape_5D)
{
    auto self = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 8D tensor
TEST_F(l2_square_test, ascend950_case_shape_8D)
{
    auto self = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquare, INPUT(self), OUTPUT(out));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
