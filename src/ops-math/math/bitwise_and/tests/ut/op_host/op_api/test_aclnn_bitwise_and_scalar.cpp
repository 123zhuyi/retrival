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

#include "aclnn_bitwise_and_scalar.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_bitwise_and_scalar_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "bitwise_and_scalar_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "bitwise_and_scalar_test TearDown" << endl;
    }
};

TEST_F(l2_bitwise_and_scalar_test, case_int16)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_ND);
    int16_t value = 1;
    auto other_tensor_desc = ScalarDesc(value);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnBitwiseAndScalar, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_bitwise_and_scalar_test, case_int32)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_ND);
    int32_t value = 1;
    auto other_tensor_desc = ScalarDesc(value);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnBitwiseAndScalar, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_bitwise_and_scalar_test, case_int64)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND);
    int64_t value = 1;
    auto other_tensor_desc = ScalarDesc(value);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnBitwiseAndScalar, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_bitwise_and_scalar_test, case_int8)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT8, ACL_FORMAT_ND);
    int8_t value = 1;
    auto other_tensor_desc = ScalarDesc(value);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnBitwiseAndScalar, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_bitwise_and_scalar_test, case_uint8)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_UINT8, ACL_FORMAT_ND);
    uint8_t value = 1;
    auto other_tensor_desc = ScalarDesc(value);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnBitwiseAndScalar, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_bitwise_and_scalar_test, case_bool)
{
    auto self_tensor_desc = TensorDesc({true, false, false, true}, ACL_BOOL, ACL_FORMAT_ND);
    bool value = true;
    auto other_tensor_desc = ScalarDesc(value);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnBitwiseAndScalar, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_bitwise_and_scalar_test, case_suppport_format)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_NCHW);
    int32_t value = 1;
    auto other_tensor_desc = ScalarDesc(value);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnBitwiseAndScalar, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_NHWC);
    out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut1 = OP_API_UT(aclnnBitwiseAndScalar, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    workspace_size = 0;
    aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_HWCN);
    out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut2 = OP_API_UT(aclnnBitwiseAndScalar, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    workspace_size = 0;
    aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_NDHWC);
    out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut3 = OP_API_UT(aclnnBitwiseAndScalar, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    workspace_size = 0;
    aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_NCDHW);
    out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut4 = OP_API_UT(aclnnBitwiseAndScalar, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    workspace_size = 0;
    aclRet = ut4.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_bitwise_and_scalar_test, case_nullptr)
{
    auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBitwiseAndScalar, INPUT((aclTensor*)nullptr, (aclScalar*)nullptr), OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 空tensor
TEST_F(l2_bitwise_and_scalar_test, case_empty)
{
    auto self_tensor_desc = TensorDesc({7, 0, 6}, ACL_INT32, ACL_FORMAT_ND);
    int32_t value = 1;
    auto other_tensor_desc = ScalarDesc(value);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnBitwiseAndScalar, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_bitwise_and_scalar_test, case_errtype)
{
    auto self_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
    int32_t value = 1;
    auto other_tensor_desc = ScalarDesc(value);
    auto out_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBitwiseAndScalar, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckNotNull
TEST_F(l2_bitwise_and_scalar_test, case_nullptr2)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_INT32, ACL_FORMAT_ND);
    int32_t value = 1;
    auto other_tensor_desc = ScalarDesc(value);

    auto ut_2 = OP_API_UT(aclnnBitwiseAndScalar, INPUT(nullptr, other_tensor_desc), OUTPUT(tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_3 = OP_API_UT(aclnnBitwiseAndScalar, INPUT(tensor_desc, nullptr), OUTPUT(tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_4 = OP_API_UT(aclnnBitwiseAndScalar, INPUT(tensor_desc, other_tensor_desc), OUTPUT(nullptr));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_4.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid
TEST_F(l2_bitwise_and_scalar_test, case_dtype_uint32)
{
    // uint32
    auto tensor_desc = TensorDesc({10, 5}, ACL_UINT32, ACL_FORMAT_ND);
    int32_t value = 1;
    auto other_tensor_desc = ScalarDesc(value);

    auto ut = OP_API_UT(aclnnBitwiseAndScalar, INPUT(tensor_desc, other_tensor_desc), OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // uint16
    tensor_desc = TensorDesc({10, 5}, ACL_UINT16, ACL_FORMAT_ND);

    auto ut_2 = OP_API_UT(aclnnBitwiseAndScalar, INPUT(tensor_desc, other_tensor_desc), OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    workspace_size = 0;
    aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // uint64
    tensor_desc = TensorDesc({10, 5}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut_3 = OP_API_UT(aclnnBitwiseAndScalar, INPUT(tensor_desc, other_tensor_desc), OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    workspace_size = 0;
    aclRet = ut_3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // float
    tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut_4 = OP_API_UT(aclnnBitwiseAndScalar, INPUT(tensor_desc, other_tensor_desc), OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    workspace_size = 0;
    aclRet = ut_4.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape
TEST_F(l2_bitwise_and_scalar_test, case_shape)
{
    auto self_tensor_desc = TensorDesc({10, 5, 2, 10}, ACL_INT32, ACL_FORMAT_ND);
    int32_t value = 1;
    auto other_tensor_desc = ScalarDesc(value);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnBitwiseAndScalar, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    self_tensor_desc = TensorDesc({10, 5, 5, 10}, ACL_INT32, ACL_FORMAT_ND);
    out_tensor_desc = TensorDesc({10, 5, 10, 5}, ACL_INT32, ACL_FORMAT_ND);

    auto ut_2 = OP_API_UT(aclnnBitwiseAndScalar, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckFormat
TEST_F(l2_bitwise_and_scalar_test, case_format)
{
    auto self_tensor_desc = TensorDesc({10, 5, 2, 10}, ACL_INT32, ACL_FORMAT_NHWC);
    int32_t value = 1;
    auto other_tensor_desc = ScalarDesc(value);
    auto out_tensor_desc = TensorDesc({10, 5, 2, 10}, ACL_INT32, ACL_FORMAT_NC1HWC0);

    auto ut_1 = OP_API_UT(aclnnBitwiseAndScalar, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut_1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    auto tensor_desc = TensorDesc({10, 5, 2, 10, 1}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);

    auto ut_2 = OP_API_UT(aclnnBitwiseAndScalar, INPUT(tensor_desc, other_tensor_desc), OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// uncontiguous
TEST_F(l2_bitwise_and_scalar_test, case_uncontiguous)
{
    auto self_tensor_desc = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 5}, 0, {4, 5});
    int32_t value = 1;
    auto other_tensor_desc = ScalarDesc(value);
    auto out_tensor_desc = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBitwiseAndScalar, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_bitwise_and_scalar_test, case_inplace_teset)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_INT32, ACL_FORMAT_ND);
    int32_t value = 1;
    auto other_desc = ScalarDesc(value);

    int64_t dim = 0;
    auto ut = OP_API_UT(aclnnInplaceBitwiseAndScalar, INPUT(tensor_1_desc, other_desc), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
