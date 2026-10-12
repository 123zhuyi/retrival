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

#include "aclnn_cast_v3.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class CastV3Test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "CastV3Test SetUp" << endl; }

    static void TearDownTestCase() { cout << "CastV3Test TearDown" << endl; }
};

TEST_F(CastV3Test, case_nullptr_input)
{
    auto out_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t dstType = ACL_FLOAT16;

    auto ut = OP_API_UT(aclnnCastV3, INPUT((aclTensor*)nullptr, dstType), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(CastV3Test, case_nullptr_output)
{
    auto x_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dstType = ACL_FLOAT16;

    auto ut = OP_API_UT(aclnnCastV3, INPUT(x_desc, dstType), OUTPUT((aclTensor*)nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(CastV3Test, case_float_to_float16)
{
    auto x_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t dstType = ACL_FLOAT16;

    auto ut = OP_API_UT(aclnnCastV3, INPUT(x_desc, dstType), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(CastV3Test, case_float16_to_float)
{
    auto x_desc = TensorDesc({1, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_desc = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = ACL_FLOAT;

    auto ut = OP_API_UT(aclnnCastV3, INPUT(x_desc, dstType), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(CastV3Test, case_float_to_int32)
{
    auto x_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto out_desc = TensorDesc({4, 5}, ACL_INT32, ACL_FORMAT_ND);
    int64_t dstType = ACL_INT32;

    auto ut = OP_API_UT(aclnnCastV3, INPUT(x_desc, dstType), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(CastV3Test, case_int8_to_float)
{
    auto x_desc = TensorDesc({32}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-128, 127);
    auto out_desc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = ACL_FLOAT;

    auto ut = OP_API_UT(aclnnCastV3, INPUT(x_desc, dstType), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(CastV3Test, case_uint8_to_bool)
{
    auto x_desc = TensorDesc({2, 2}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 255);
    auto out_desc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);
    int64_t dstType = ACL_BOOL;

    auto ut = OP_API_UT(aclnnCastV3, INPUT(x_desc, dstType), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(CastV3Test, case_bf16_to_float)
{
    auto x_desc = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = ACL_FLOAT;

    auto ut = OP_API_UT(aclnnCastV3, INPUT(x_desc, dstType), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(CastV3Test, case_shape_mismatch)
{
    auto x_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_desc = TensorDesc({3, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t dstType = ACL_FLOAT16;

    auto ut = OP_API_UT(aclnnCastV3, INPUT(x_desc, dstType), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
