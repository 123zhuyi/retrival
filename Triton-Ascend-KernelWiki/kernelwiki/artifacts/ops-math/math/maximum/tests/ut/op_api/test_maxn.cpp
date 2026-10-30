/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../op_api/aclnn_maxn.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/inner/types.h"

class l2_maxn_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_maxn_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_maxn_test TearDown" << std::endl;
    }
};

// 输入为空指针
TEST_F(l2_maxn_test, l2_maxn_test_nullptr_input)
{
    auto outTensorDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMaxN, INPUT(nullptr), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 输出为空指针
TEST_F(l2_maxn_test, l2_maxn_test_nullptr_output)
{
    auto tensor1Desc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor2Desc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorListDesc = TensorListDesc({tensor1Desc, tensor2Desc});
    auto ut = OP_API_UT(aclnnMaxN, INPUT(tensorListDesc), OUTPUT(nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 空tensors
TEST_F(l2_maxn_test, l2_maxn_test_empty_tensors)
{
    auto tensor1Desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor2Desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorListDesc = TensorListDesc({tensor1Desc, tensor2Desc});
    auto outTensorDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMaxN, INPUT(tensorListDesc), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 不能广播
TEST_F(l2_maxn_test, l2_maxn_test_broadcast_failed1)
{
    auto tensor1Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor2Desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorListDesc = TensorListDesc({tensor1Desc, tensor2Desc});
    auto outTensorDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMaxN, INPUT(tensorListDesc), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_maxn_test, l2_maxn_test_broadcast_failed2)
{
    auto tensor1Desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor2Desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorListDesc = TensorListDesc({tensor1Desc, tensor2Desc});
    auto outTensorDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMaxN, INPUT(tensorListDesc), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输入维度超过8
TEST_F(l2_maxn_test, l2_maxn_test_dim_over_8)
{
    auto tensor1Desc = TensorDesc({2, 1, 1, 1, 1, 1, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor2Desc = TensorDesc({2, 1, 1, 1, 1, 1, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorListDesc = TensorListDesc({tensor1Desc, tensor2Desc});
    auto outTensorDesc = TensorDesc({2, 1, 1, 1, 1, 1, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMaxN, INPUT(tensorListDesc), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径
TEST_F(l2_maxn_test, l2_maxn_test_dypte_float32)
{
    auto tensor1Desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2});
    auto tensor2Desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{3, 4, 5, 6, 7, 8});
    auto tensorListDesc = TensorListDesc({tensor1Desc, tensor2Desc});
    auto outTensorDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMaxN, INPUT(tensorListDesc), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_maxn_test, l2_maxn_test_dypte_float16)
{
    auto tensor1Desc = TensorDesc({2, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor2Desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensorListDesc = TensorListDesc({tensor1Desc, tensor2Desc});
    auto outTensorDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMaxN, INPUT(tensorListDesc), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_maxn_test, ascend910B2_maxn_dypte_bfloat16)
{
    auto tensor1Desc = TensorDesc({2, 1}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor2Desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto tensorListDesc = TensorListDesc({tensor1Desc, tensor2Desc});
    auto outTensorDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMaxN, INPUT(tensorListDesc), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_maxn_test, l2_maxn_test_dypte_int64)
{
    auto tensor1Desc = TensorDesc({2, 1}, ACL_INT64, ACL_FORMAT_ND);
    auto tensor2Desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto tensorListDesc = TensorListDesc({tensor1Desc, tensor2Desc});
    auto outTensorDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMaxN, INPUT(tensorListDesc), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_maxn_test, l2_maxn_test_dypte_int32)
{
    auto tensor1Desc = TensorDesc({2, 1}, ACL_INT32, ACL_FORMAT_ND);
    auto tensor2Desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto tensorListDesc = TensorListDesc({tensor1Desc, tensor2Desc});
    auto outTensorDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMaxN, INPUT(tensorListDesc), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_maxn_test, l2_maxn_test_dypte_int8)
{
    auto tensor1Desc = TensorDesc({2, 1}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor2Desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto tensorListDesc = TensorListDesc({tensor1Desc, tensor2Desc});
    auto outTensorDesc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnMaxN, INPUT(tensorListDesc), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_maxn_test, l2_maxn_test_dypte_tensor_num_1)
{
    auto tensor1Desc = TensorDesc({2, 1}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1, 2});
    auto tensorListDesc = TensorListDesc({tensor1Desc});
    auto outTensorDesc = TensorDesc({2, 1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMaxN, INPUT(tensorListDesc), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_maxn_test, l2_maxn_test_dypte_tensor_num_20)
{
    auto tensor1Desc = TensorDesc({2, 1}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1, 2});
    auto tensorListDesc = TensorListDesc(20, tensor1Desc);
    auto outTensorDesc = TensorDesc({2, 1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnMaxN, INPUT(tensorListDesc), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_maxn_test, l2_maxn_test_empty_tensor_21)
{
    auto tensor1Desc = TensorDesc({0, 1}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor2Desc = TensorDesc({0, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto tensorListDesc = TensorListDesc({tensor1Desc, tensor2Desc});
    auto outTensorDesc = TensorDesc({0, 3}, ACL_INT8, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnMaxN, INPUT(tensorListDesc), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}