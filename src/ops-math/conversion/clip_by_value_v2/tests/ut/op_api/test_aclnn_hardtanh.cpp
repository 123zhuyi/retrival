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
#include "conversion/clip_by_value_v2/op_api/aclnn_hardtanh.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_hardtanh_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "hardtanh_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "hardtanh_test TearDown" << std::endl;
    }
};

TEST_F(l2_hardtanh_test, case_ACL_COMPLEX64)
{
    auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto min_scalar_desc = ScalarDesc(-1);
    auto max_scalar_desc = ScalarDesc(2);
    auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardtanh, INPUT(tensor_desc, min_scalar_desc, max_scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardtanh_test, case_ACL_COMPLEX128)
{
    auto tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto min_scalar_desc = ScalarDesc(-1);
    auto max_scalar_desc = ScalarDesc(2);
    auto out_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardtanh, INPUT(tensor_desc, min_scalar_desc, max_scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardtanh_test, case_empty)
{
    auto tensor_desc = TensorDesc({2, 0}, ACL_INT64, ACL_FORMAT_ND);
    auto min_scalar_desc = ScalarDesc(-1);
    auto max_scalar_desc = ScalarDesc(2);
    auto out_tensor_desc = TensorDesc({2, 0}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardtanh, INPUT(tensor_desc, min_scalar_desc, max_scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_hardtanh_test, ascend910B2_case_ACL_FP16_min)
{
    auto min_scalar_desc = ScalarDesc(3);
    auto max_scalar_desc = ScalarDesc(2);
    uint64_t workspace_size = 0;
    auto tensor_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut_1 =
        OP_API_UT(aclnnHardtanh, INPUT(tensor_desc_1, min_scalar_desc, max_scalar_desc), OUTPUT(out_tensor_desc_1));

    aclnnStatus aclRet_1 = ut_1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet_1, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardtanh_test, ascend910B2_case_ACL_INT_min)
{
    auto min_scalar_desc = ScalarDesc(3);
    auto max_scalar_desc = ScalarDesc(2);
    uint64_t workspace_size = 0;
    auto tensor_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_ND);
    auto out_tensor_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut_1 =
        OP_API_UT(aclnnHardtanh, INPUT(tensor_desc_1, min_scalar_desc, max_scalar_desc), OUTPUT(out_tensor_desc_1));

    aclnnStatus aclRet_1 = ut_1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet_1, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardtanh_test, case_unsupport_dtype)
{
    auto min_scalar_desc = ScalarDesc(-1);
    auto max_scalar_desc = ScalarDesc(2);
    uint64_t workspace_size = 0;

    // different dtype
    auto tensor_desc_2 = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND);
    auto out_tensor_desc_2 = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut_2 =
        OP_API_UT(aclnnHardtanh, INPUT(tensor_desc_2, min_scalar_desc, max_scalar_desc), OUTPUT(out_tensor_desc_2));

    aclnnStatus aclRet_2 = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet_2, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardtanh_test, case_unsupport_shape)
{
    auto min_scalar_desc = ScalarDesc(-1);
    auto max_scalar_desc = ScalarDesc(2);
    uint64_t workspace_size = 0;

    auto tensor_desc_1 = TensorDesc({5, 4, 3, 2}, ACL_INT64, ACL_FORMAT_NCHW);
    auto out_tensor_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut_1 =
        OP_API_UT(aclnnHardtanh, INPUT(tensor_desc_1, min_scalar_desc, max_scalar_desc), OUTPUT(out_tensor_desc_1));

    aclnnStatus aclRet = ut_1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardtanh_test, case_null)
{
    auto min_scalar_desc = ScalarDesc(-1);
    auto max_scalar_desc = ScalarDesc(2);
    uint64_t workspace_size = 0;

    auto tensor_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_NCHW);
    auto out_tensor_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut_1 = OP_API_UT(aclnnHardtanh, INPUT(nullptr, min_scalar_desc, max_scalar_desc), OUTPUT(out_tensor_desc_1));

    aclnnStatus aclRet_1 = ut_1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet_1, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_2 = OP_API_UT(aclnnHardtanh, INPUT(tensor_desc_1, min_scalar_desc, max_scalar_desc), OUTPUT(nullptr));

    aclnnStatus aclRet_2 = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet_2, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_3 = OP_API_UT(aclnnHardtanh, INPUT(tensor_desc_1, nullptr, max_scalar_desc), OUTPUT(out_tensor_desc_1));

    aclnnStatus aclRet_3 = ut_3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet_3, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_4 = OP_API_UT(aclnnHardtanh, INPUT(tensor_desc_1, min_scalar_desc, nullptr), OUTPUT(out_tensor_desc_1));

    aclnnStatus aclRet_4 = ut_4.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet_4, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_hardtanh_test, case_9dim)
{
    auto tensor_desc = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_INT64, ACL_FORMAT_NCDHW);
    auto min_scalar_desc = ScalarDesc(-1);
    auto max_scalar_desc = ScalarDesc(2);
    auto out_tensor_desc =
        TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_INT64, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardtanh, INPUT(tensor_desc, min_scalar_desc, max_scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常路径，ACL_COMPLEX128，inplace
TEST_F(l2_hardtanh_test, case_inplace_complex128)
{
    auto tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto min_scalar_desc = ScalarDesc(-1.5f);
    auto max_scalar_desc = ScalarDesc(2.5f);

    auto ut = OP_API_UT(aclnnInplaceHardtanh, INPUT(tensor_desc, min_scalar_desc, max_scalar_desc), OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 异常路径，ACL_F32，inplace
TEST_F(l2_hardtanh_test, ascend910B2_case_ACL_f32_min)
{
    auto min_scalar_desc = ScalarDesc(3.5f);
    auto max_scalar_desc = ScalarDesc(2.5f);
    uint64_t workspace_size = 0;
    auto tensor_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut_1 =
        OP_API_UT(aclnnHardtanh, INPUT(tensor_desc_1, min_scalar_desc, max_scalar_desc), OUTPUT(out_tensor_desc_1));

    aclnnStatus aclRet_1 = ut_1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet_1, ACLNN_ERR_PARAM_INVALID);
}