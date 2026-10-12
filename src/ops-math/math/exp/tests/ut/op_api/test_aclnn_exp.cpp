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

#include "../../../op_api/aclnn_exp.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_exp_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "exp_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "exp_test TearDown" << endl;
    }
};

// 正常场景_FLOAT_ND
TEST_F(l2_exp_test, l2_exp_normal_dtype_FLOAT_ND)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常场景_FLOAT16_NCHW
TEST_F(l2_exp_test, l2_exp_normal_dtype_FLOAT16_NCHW)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常场景_BOOL_NCDHW
TEST_F(l2_exp_test, l2_exp_normal_dtype_BOOL_NCDHW)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_BOOL, ACL_FORMAT_NCDHW);
    auto outDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常场景_INT64_NCHW_NHWC
TEST_F(l2_exp_test, l2_exp_normal_dtype_INT64_NCHW_NHWC)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常场景_BFLOAT16_ND
TEST_F(l2_exp_test, l2_exp_normal_dtype_BFLOAT16_ND)
{
    auto selfDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    } else {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

// 空tensor场景
TEST_F(l2_exp_test, l2_exp_normal_empty_tensor)
{
    auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckNotNull_self_nullptr
TEST_F(l2_exp_test, l2_exp_abnormal_self_nullptr)
{
    auto selfDesc = nullptr;
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_out_nullptr
TEST_F(l2_exp_test, l2_exp_abnormal_out_nullptr)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = nullptr;

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid_INT32
TEST_F(l2_exp_test, l2_exp_abnormal_dtype_INT32)
{
    auto selfDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_INT64
TEST_F(l2_exp_test, l2_exp_abnormal_dtype_INT64)
{
    auto selfDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_INT16
TEST_F(l2_exp_test, l2_exp_abnormal_dtype_INT16)
{
    auto selfDesc = TensorDesc({2}, ACL_INT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_INT8
TEST_F(l2_exp_test, l2_exp_abnormal_dtype_INT8)
{
    auto selfDesc = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_UINT8
TEST_F(l2_exp_test, l2_exp_abnormal_dtype_UINT8)
{
    auto selfDesc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_BOOL
TEST_F(l2_exp_test, l2_exp_abnormal_dtype_BOOL)
{
    auto selfDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_UNDEFINED
TEST_F(l2_exp_test, l2_exp_abnormal_dtype_UNDEFINED)
{
    auto selfDesc = TensorDesc({2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_dtype_unequal
TEST_F(l2_exp_test, l2_exp_abnormal_dtype_unequal)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckPromoteType_can_cast
TEST_F(l2_exp_test, l2_exp_normal_dtype_can_cast)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckPromoteType_cannot_cast
TEST_F(l2_exp_test, l2_exp_abnormal_dtype_cannot_cast)
{
    auto selfDesc = TensorDesc({2}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckPromoteType_cannot_cast2
TEST_F(l2_exp_test, l2_exp_abnormal_dtype_cannot_cast2)
{
    auto selfDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckPromoteType_cannot_cast3
TEST_F(l2_exp_test, l2_exp_abnormal_dtype_cannot_cast3)
{
    auto selfDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape
TEST_F(l2_exp_test, l2_exp_abnormal_shape_unequal)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_10D
TEST_F(l2_exp_test, l2_exp_abnormal_shape_10D)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 数据范围[-1，1]
TEST_F(l2_exp_test, l2_exp_normal_valuerange)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 非连续
TEST_F(l2_exp_test, l2_exp_normal_uncontiguous)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});

    auto ut = OP_API_UT(aclnnExp, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_exp_test, l2_inplace_exp_not_supported_dtype)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT64, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto ut = OP_API_UT(aclnnInplaceExp, INPUT(selfDesc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}