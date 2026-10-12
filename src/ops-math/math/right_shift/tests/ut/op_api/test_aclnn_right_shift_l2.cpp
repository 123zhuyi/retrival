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
#include "math/right_shift/op_api/aclnn_right_shift.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;
using namespace op;

class l2_right_shift_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "tensor_right_shift_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "tensor_right_shift_test TearDown" << endl;
    }
};

// 正常调用流程
TEST_F(l2_right_shift_test, normal_success)
{
    auto inputDesc = TensorDesc({1, 4}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{10, 20, 30, 40});
    auto shiftBitsDesc = TensorDesc({1, 4}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1, 2, 3, 4});
    auto outDesc = TensorDesc(inputDesc);
    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(aclnnRightShift, INPUT(inputDesc, shiftBitsDesc), OUTPUT(outDesc));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 输入input为空指针
TEST_F(l2_right_shift_test, nullptr_input)
{
    auto inputDesc = TensorDesc({1, 4}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{10, 20, 30, 40});
    auto shiftBitsDesc = TensorDesc({1, 4}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1, 2, 3, 4});
    auto outDesc = TensorDesc(inputDesc);
    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(aclnnRightShift, INPUT(nullptr, shiftBitsDesc), OUTPUT(outDesc));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 输入shiftBits为空指针
TEST_F(l2_right_shift_test, nullptr_shiftBits)
{
    auto inputDesc = TensorDesc({1, 4}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{10, 20, 30, 40});
    auto shiftBitsDesc = TensorDesc({1, 4}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1, 2, 3, 4});
    auto outDesc = TensorDesc(inputDesc);
    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(aclnnRightShift, INPUT(inputDesc, nullptr), OUTPUT(outDesc));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 输入out为空指针
TEST_F(l2_right_shift_test, nullptr_out)
{
    auto inputDesc = TensorDesc({1, 4}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{10, 20, 30, 40});
    auto shiftBitsDesc = TensorDesc({1, 4}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1, 2, 3, 4});
    auto outDesc = TensorDesc(inputDesc);
    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(aclnnRightShift, INPUT(inputDesc, shiftBitsDesc), OUTPUT(nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 输入类型不支持
TEST_F(l2_right_shift_test, type_unsupport)
{
    auto inputDesc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{10, 20, 30, 40});
    auto shiftBitsDesc = TensorDesc({1, 4}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1, 2, 3, 4});
    auto outDesc = TensorDesc(inputDesc);
    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(aclnnRightShift, INPUT(inputDesc, shiftBitsDesc), OUTPUT(outDesc));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输入类型可以正常类型推导
TEST_F(l2_right_shift_test, type_promote_normal)
{
    auto inputDesc = TensorDesc({1, 4}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{10, 20, 30, 40});
    auto shiftBitsDesc = TensorDesc({1, 4}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1, 2, 3, 4});
    auto outDesc = TensorDesc(inputDesc);
    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(aclnnRightShift, INPUT(inputDesc, shiftBitsDesc), OUTPUT(outDesc));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 输入类型不支持类型推导
TEST_F(l2_right_shift_test, type_promote_unsupport)
{
    auto inputDesc = TensorDesc({1, 4}, ACL_UINT16, ACL_FORMAT_ND).Value(vector<uint16_t>{10, 20, 30, 40});
    auto shiftBitsDesc = TensorDesc({1, 4}, ACL_UINT32, ACL_FORMAT_ND).Value(vector<uint32_t>{1, 2, 3, 4});
    auto outDesc = TensorDesc(inputDesc);
    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(aclnnRightShift, INPUT(inputDesc, shiftBitsDesc), OUTPUT(outDesc));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输入类型推导结果无法转换为out类型-不存在，整型可以向任何类型转换，结果应该为true
TEST_F(l2_right_shift_test, type_promote_can_not_cast_to_out)
{
    auto inputDesc = TensorDesc({1, 4}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{10, 20, 30, 40});
    auto shiftBitsDesc = TensorDesc({1, 4}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{1, 2, 3, 4});
    auto outDesc = TensorDesc({1, 4}, ACL_UINT16, ACL_FORMAT_ND).Value(vector<int64_t>{0, 0, 0, 0});
    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(aclnnRightShift, INPUT(inputDesc, shiftBitsDesc), OUTPUT(outDesc));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 输入shape为空
TEST_F(l2_right_shift_test, shape_empty_input)
{
    auto inputDesc = TensorDesc({1, 4}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{10, 20, 30, 40});
    auto shiftBitsDesc = TensorDesc({}, ACL_INT8, ACL_FORMAT_ND);
    auto outDesc = TensorDesc(inputDesc);
    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(aclnnRightShift, INPUT(inputDesc, shiftBitsDesc), OUTPUT(outDesc));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 输入input shape维度超过8
TEST_F(l2_right_shift_test, input_shape_dim_over_8)
{
    auto inputDesc =
        TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 4}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{10, 20, 30, 40});
    auto shiftBitsDesc = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1});
    auto outDesc = TensorDesc(inputDesc);
    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(aclnnRightShift, INPUT(inputDesc, shiftBitsDesc), OUTPUT(outDesc));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输入shiftbits shape维度超过8
TEST_F(l2_right_shift_test, shiftBits_shape_dim_over_8)
{
    auto inputDesc = TensorDesc({1, 4}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{10, 20, 30, 40});
    auto shiftBitsDesc =
        TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 4}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1, 1, 1, 1});
    auto outDesc = TensorDesc(inputDesc);
    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(aclnnRightShift, INPUT(inputDesc, shiftBitsDesc), OUTPUT(outDesc));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输入shape与输出shape不一致
TEST_F(l2_right_shift_test, shape_input_noeq_out)
{
    auto inputDesc = TensorDesc({1, 4}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{10, 20, 30, 40});
    auto shiftBitsDesc = TensorDesc({1, 4}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1, 1, 1, 1});
    auto outDesc = TensorDesc({4}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{10, 20, 30, 40});
    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(aclnnRightShift, INPUT(inputDesc, shiftBitsDesc), OUTPUT(outDesc));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输入shape支持广播
TEST_F(l2_right_shift_test, shape_broadcast_normal)
{
    auto inputDesc = TensorDesc({1, 4}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{10, 20, 30, 40});
    auto shiftBitsDesc = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1});
    auto outDesc = TensorDesc(inputDesc);
    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(aclnnRightShift, INPUT(inputDesc, shiftBitsDesc), OUTPUT(outDesc));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 输入shape不支持广播-维度数一致，大小不一致，且没有等于1的维度
TEST_F(l2_right_shift_test, shape_broadcast_unsupport)
{
    auto inputDesc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{10, 20, 30, 40, 50, 60});
    auto shiftBitsDesc = TensorDesc({3, 2}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1, 2, 3, 4, 1, 2});
    auto outDesc = TensorDesc(inputDesc);
    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(aclnnRightShift, INPUT(inputDesc, shiftBitsDesc), OUTPUT(outDesc));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输入shape广播后结果与out的shape不一致
TEST_F(l2_right_shift_test, shape_broadcast_not_equal_with_out)
{
    auto inputDesc = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{10});
    auto shiftBitsDesc = TensorDesc({1, 4}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{1, 1, 1, 1});
    auto outDesc = TensorDesc(inputDesc);
    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(aclnnRightShift, INPUT(inputDesc, shiftBitsDesc), OUTPUT(outDesc));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}