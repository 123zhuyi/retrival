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
#include "opdev/op_log.h"
#include "aclnn_tan.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_tan_test : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "Tan Test Setup" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "Tan Test TearDown" << std::endl;
    }
};

TEST_F(l2_tan_test, case_fp32)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND)
        .ValueRange(-2, 2)
        .Value(vector<float>{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6});
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnTan, INPUT(tensor_desc), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_tan_test, case_empty_tensor)
{
    auto self_tensor_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_tan_test, case_nullptr_input)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTan, INPUT(nullptr), OUTPUT(tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_tan_test, case_nullptr_output)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTan, INPUT(tensor_desc), OUTPUT(nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_tan_test, case_data_cannot_cast_1)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_INT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_tan_test, case_data_cannot_cast_2)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_INT64, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_tan_test, case_invalid_dtype)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_UINT64, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_tan_test, case_float16)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_tan_test, case_float16_inplace)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceTan, INPUT(self_desc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_tan_test, case_data_cannot_cast_inplace)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceTan, INPUT(self_desc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_tan_test, case_float)
{
    auto self_desc = TensorDesc({32, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({32, 128}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_tan_test, ascend910B2_case_bfloat16)
{
    auto self_desc = TensorDesc({32, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({32, 128}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_tan_test, case_invalid_shape)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_tan_test, case_double)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_tan_test, case_float_nhwc)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_tan_test, case_float_hwcn)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_HWCN);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_tan_test, case_float_nchw)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_tan_test, case_int32)
{
    auto self_desc = TensorDesc({1, 16}, ACL_INT32, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_tan_test, case_int8)
{
    auto self_desc = TensorDesc({1, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_tan_test, case_uint8)
{
    auto self_desc = TensorDesc({1, 16}, ACL_UINT8, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_tan_test, case_int16)
{
    auto self_desc = TensorDesc({1, 16}, ACL_INT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_tan_test, case_int64)
{
    auto self_desc = TensorDesc({1, 16}, ACL_INT64, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_tan_test, case_bool)
{
    auto self_desc = TensorDesc({1, 16}, ACL_BOOL, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_tan_test, case_complex64)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_tan_test, case_complex128)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX128, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_tan_test, case_invalid_dim)
{
    auto self_desc = TensorDesc({1, 16, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_tan_test, case_bool_transpose)
{
    auto self_desc = TensorDesc({2, 16, 4}, ACL_BOOL, ACL_FORMAT_ND, {64, 1, 4}, 0, {2, 16, 4});
    auto out_desc = TensorDesc({2, 16, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}