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

#include "conversion/pack/op_api/aclnn_stack.h"
#include "gtest/gtest.h"
#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class l2_stack_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_stack_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_stack_test TearDown" << endl;
    }

    void TearDown() override
    {
        op::SetPlatformNpuArch(NpuArch::DAV_2201);
    }
};

// 输入为空指针
TEST_F(l2_stack_test, l2_stack_test_nullptr_input)
{
    int64_t dim = 0;
    auto out_tensor_desc = TensorDesc({2, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnStack, INPUT(nullptr, dim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// out为空指针
TEST_F(l2_stack_test, l2_stack_test_nullptr_out)
{
    auto tensor_1_desc = TensorDesc({2, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_2_desc = TensorDesc({2, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = 0;
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// 空tensors
TEST_F(l2_stack_test, l2_stack_test_empty_tensors)
{
    auto tensor_1_desc = TensorDesc({2, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_2_desc = TensorDesc({2, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = 0;
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    auto out_tensor_desc = TensorDesc({2, 2, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，float16
TEST_F(l2_stack_test, l2_stack_test_dtype_float16)
{
    auto tensor_1_desc =
        TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc =
        TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，float32
TEST_F(l2_stack_test, l2_stack_test_dtype_float32)
{
    auto tensor_1_desc =
        TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc =
        TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，double
TEST_F(l2_stack_test, l2_stack_test_dtype_double)
{
    auto tensor_1_desc =
        TensorDesc({2, 5}, ACL_DOUBLE, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc =
        TensorDesc({2, 5}, ACL_DOUBLE, ACL_FORMAT_ND).Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，int8
TEST_F(l2_stack_test, l2_stack_test_dtype_int8)
{
    auto tensor_1_desc =
        TensorDesc({2, 5}, ACL_INT8, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc =
        TensorDesc({2, 5}, ACL_INT8, ACL_FORMAT_ND).Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，int16
TEST_F(l2_stack_test, l2_stack_test_dtype_int16)
{
    auto tensor_1_desc =
        TensorDesc({2, 5}, ACL_INT16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc =
        TensorDesc({2, 5}, ACL_INT16, ACL_FORMAT_ND).Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_INT16, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，int32
TEST_F(l2_stack_test, l2_stack_test_dtype_int32)
{
    auto tensor_1_desc =
        TensorDesc({2, 5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc =
        TensorDesc({2, 5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_INT32, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，int64
TEST_F(l2_stack_test, l2_stack_test_dtype_int64)
{
    auto tensor_1_desc =
        TensorDesc({2, 5}, ACL_INT64, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc =
        TensorDesc({2, 5}, ACL_INT64, ACL_FORMAT_ND).Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_INT64, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，uint8
TEST_F(l2_stack_test, l2_stack_test_dtype_uint8)
{
    auto tensor_1_desc =
        TensorDesc({2, 5}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc =
        TensorDesc({2, 5}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_UINT8, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，uint16
TEST_F(l2_stack_test, l2_stack_test_dtype_uint16)
{
    auto tensor_1_desc =
        TensorDesc({2, 5}, ACL_UINT16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc =
        TensorDesc({2, 5}, ACL_UINT16, ACL_FORMAT_ND).Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_UINT16, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，uint32
TEST_F(l2_stack_test, l2_stack_test_dtype_uint32)
{
    auto tensor_1_desc =
        TensorDesc({2, 5}, ACL_UINT32, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc =
        TensorDesc({2, 5}, ACL_UINT32, ACL_FORMAT_ND).Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_UINT32, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，uint64
TEST_F(l2_stack_test, l2_stack_test_dtype_uint64)
{
    auto tensor_1_desc =
        TensorDesc({2, 5}, ACL_UINT64, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc =
        TensorDesc({2, 5}, ACL_UINT64, ACL_FORMAT_ND).Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_UINT64, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，bool
TEST_F(l2_stack_test, l2_stack_test_dtype_bool)
{
    auto tensor_1_desc =
        TensorDesc({2, 5}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc =
        TensorDesc({2, 5}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_BOOL, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，complex64
TEST_F(l2_stack_test, l2_stack_test_dtype_complex64)
{
    auto tensor_1_desc =
        TensorDesc({2, 5}, ACL_COMPLEX64, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc =
        TensorDesc({2, 5}, ACL_COMPLEX64, ACL_FORMAT_ND).Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，bfloat16
TEST_F(l2_stack_test, ascend910B2_l2_stack_test_dtype_bfloat16)
{
    auto tensor_1_desc =
        TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc =
        TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 用例不支持，float32转int8
TEST_F(l2_stack_test, l2_stack_test_dtype_cast)
{
    auto tensor_1_desc =
        TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc =
        TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输入tensors维度数量不同
TEST_F(l2_stack_test, l2_stack_test_different_dim_num)
{
    auto tensor_1_desc =
        TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc =
        TensorDesc({2, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输入tensors维度不同
TEST_F(l2_stack_test, l2_stack_test_different_shape)
{
    auto tensor_1_desc =
        TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{11, 12, 13, 14, 15, 16});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，输入1个tensor
TEST_F(l2_stack_test, l2_stack_test_tensors_1)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto tensor_list_desc = TensorListDesc(1, tensor_1_desc);
    auto out_tensor_desc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，输入200个tensors
TEST_F(l2_stack_test, l2_stack_test_tensors_200)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto tensor_list_desc = TensorListDesc(200, tensor_1_desc);
    auto out_tensor_desc = TensorDesc({200, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 用例不支持，输入tensor维度超过8
TEST_F(l2_stack_test, l2_stack_test_9dim)
{
    auto tensor_1_desc = TensorDesc({2, 3, 2, 3, 2, 3, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_2_desc = TensorDesc({2, 3, 2, 3, 2, 3, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 2, 3, 2, 3, 2, 3, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dim测试
TEST_F(l2_stack_test, l2_stack_test_different_dim)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_2_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});

    int64_t dim = 0;
    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    dim = -3;
    ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    workspace_size = 0;
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    dim = 3;
    ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    workspace_size = 0;
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    dim = -4;
    ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    workspace_size = 0;
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输入1个tensor且为空
TEST_F(l2_stack_test, l2_stack_test_one_empty_tensor)
{
    auto tensor_1_desc = TensorDesc({5, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc});
    auto out_tensor_desc = TensorDesc({1, 5, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，bfloat16
TEST_F(l2_stack_test, ascend310P_l2_stack_test_dtype_bfloat16)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2002);
    auto tensor_1_desc =
        TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto tensor_2_desc =
        TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto out_tensor_desc = TensorDesc({2, 2, 5}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnStack, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
