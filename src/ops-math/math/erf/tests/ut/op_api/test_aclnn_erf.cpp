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
#include "math/erf/op_api/aclnn_erf.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2ErfTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2ErfTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2ErfTest TearDown" << std::endl;
    }
};

// self的数据类型不在支持范围内
TEST_F(l2ErfTest, l2_erf_test_001)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// out的数据类型不在支持范围内
TEST_F(l2ErfTest, l2_erf_test_002)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self和out的shape不一致
TEST_F(l2ErfTest, l2_erf_test_004)
{
    auto selfDesc = TensorDesc({2, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self和out的shape不一致
TEST_F(l2ErfTest, l2_erf_test_005)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 空tensor
TEST_F(l2ErfTest, l2_erf_test_006)
{
    auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float32,万分之一精度不合格，用千分之一精度。
TEST_F(l2ErfTest, l2_erf_test_007)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float16  由于NUMPY目前不支持ERF函数，无法进行精度测试
TEST_F(l2ErfTest, l2_erf_test_008)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// self为空
TEST_F(l2ErfTest, l2_erf_test_009)
{
    auto outDesc = TensorDesc({2, 4, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErf, INPUT((aclTensor*)nullptr), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// out为空
TEST_F(l2ErfTest, l2_erf_test_010)
{
    auto selfDesc = TensorDesc({2, 4, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT((aclTensor*)nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// dtype int32
TEST_F(l2ErfTest, l2_erf_test_013)
{
    auto selfDesc = TensorDesc({2, 4, 6, 7}, ACL_INT32, ACL_FORMAT_NDHWC);
    auto outDesc = TensorDesc({2, 4, 6, 7}, ACL_INT32, ACL_FORMAT_NDHWC);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// dtype int64
TEST_F(l2ErfTest, l2_erf_test_014)
{
    auto selfDesc = TensorDesc({2, 4, 6, 7}, ACL_INT64, ACL_FORMAT_NDHWC);
    auto outDesc = TensorDesc({2, 4, 6, 7}, ACL_INT64, ACL_FORMAT_NDHWC);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// dtype int16
TEST_F(l2ErfTest, l2_erf_test_015)
{
    auto selfDesc = TensorDesc({2, 4, 6, 7}, ACL_INT16, ACL_FORMAT_NDHWC);
    auto outDesc = TensorDesc({2, 4, 6, 7}, ACL_INT16, ACL_FORMAT_NDHWC);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// dtype int8
TEST_F(l2ErfTest, l2_erf_test_016)
{
    auto selfDesc = TensorDesc({2, 4, 6, 7}, ACL_INT8, ACL_FORMAT_NDHWC);
    auto outDesc = TensorDesc({2, 4, 6, 7}, ACL_INT8, ACL_FORMAT_NDHWC);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// dtype uint8
TEST_F(l2ErfTest, l2_erf_test_017)
{
    auto selfDesc = TensorDesc({2, 4, 6, 7}, ACL_UINT8, ACL_FORMAT_NDHWC);
    auto outDesc = TensorDesc({2, 4, 6, 7}, ACL_UINT8, ACL_FORMAT_NDHWC);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// dtype bool
TEST_F(l2ErfTest, l2_erf_test_018)
{
    auto selfDesc = TensorDesc({2, 4, 6, 7}, ACL_BOOL, ACL_FORMAT_NDHWC);
    auto outDesc = TensorDesc({2, 4, 6, 7}, ACL_FLOAT, ACL_FORMAT_NDHWC);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// dtype complex64
TEST_F(l2ErfTest, l2_erf_test_019)
{
    auto selfDesc = TensorDesc({2, 4, 6, 7}, ACL_COMPLEX64, ACL_FORMAT_NDHWC);
    auto outDesc = TensorDesc({2, 4, 6, 7}, ACL_COMPLEX64, ACL_FORMAT_NDHWC);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// dtype complex128
TEST_F(l2ErfTest, l2_erf_test_020)
{
    auto selfDesc = TensorDesc({2, 4, 6, 7}, ACL_COMPLEX128, ACL_FORMAT_NDHWC);
    auto outDesc = TensorDesc({2, 4, 6, 7}, ACL_COMPLEX128, ACL_FORMAT_NDHWC);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// dtype undefine
TEST_F(l2ErfTest, l2_erf_test_021)
{
    auto selfDesc = TensorDesc({2, 4, 6, 7}, ACL_DT_UNDEFINED, ACL_FORMAT_NDHWC);
    auto outDesc = TensorDesc({2, 4, 6, 7}, ACL_DT_UNDEFINED, ACL_FORMAT_NDHWC);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// format self是私有格式
TEST_F(l2ErfTest, l2_erf_test_025)
{
    auto selfDesc = TensorDesc({2, 4, 6, 8, 5}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
    auto outDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_HWCN);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// format out是私有格式
TEST_F(l2ErfTest, l2_erf_test_026)
{
    auto selfDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto outDesc = TensorDesc({2, 4, 6, 8, 8}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，float32，inplace
TEST_F(l2ErfTest, l2_erf_test_027)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnInplaceErf, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 非连续
TEST_F(l2ErfTest, l2_erf_test_029)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// CheckShape_10D
TEST_F(l2ErfTest, l2_erf_test_030)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Check dtype
TEST_F(l2ErfTest, l2_erf_test_031)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_BOOL, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Check dtype
TEST_F(l2ErfTest, l2_erf_test_032)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErf, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
