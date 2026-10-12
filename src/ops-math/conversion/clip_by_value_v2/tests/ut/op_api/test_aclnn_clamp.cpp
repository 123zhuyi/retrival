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

#include "conversion/clip_by_value_v2/op_api/aclnn_clamp.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_clamp_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "clamp_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "clamp_test TearDown" << endl;
    }
};

TEST_F(l2_clamp_test, case_unsupport_shape)
{
    auto min_scalar_desc = ScalarDesc(-1);
    auto max_scalar_desc = ScalarDesc(2);
    uint64_t workspace_size = 0;

    auto tensor_desc_1 = TensorDesc({5, 4, 3, 2}, ACL_INT64, ACL_FORMAT_NCHW);
    auto out_tensor_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut_1 =
        OP_API_UT(aclnnClamp, INPUT(tensor_desc_1, min_scalar_desc, max_scalar_desc), OUTPUT(out_tensor_desc_1));

    aclnnStatus aclRet = ut_1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_clamp_test, case_null)
{
    auto min_scalar_desc = ScalarDesc(-1);
    auto max_scalar_desc = ScalarDesc(2);
    uint64_t workspace_size = 0;

    auto tensor_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_NCHW);
    auto out_tensor_desc_1 = TensorDesc({2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut_1 = OP_API_UT(aclnnClamp, INPUT(nullptr, min_scalar_desc, max_scalar_desc), OUTPUT(out_tensor_desc_1));

    aclnnStatus aclRet_1 = ut_1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet_1, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_2 = OP_API_UT(aclnnClamp, INPUT(tensor_desc_1, min_scalar_desc, max_scalar_desc), OUTPUT(nullptr));

    aclnnStatus aclRet_2 = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet_2, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_3 = OP_API_UT(aclnnClamp, INPUT(tensor_desc_1, nullptr, nullptr), OUTPUT(out_tensor_desc_1));

    aclnnStatus aclRet_3 = ut_3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet_3, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_clamp_test, case_9dim)
{
    auto tensor_desc = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_INT64, ACL_FORMAT_NCDHW);
    auto min_scalar_desc = ScalarDesc(-1);
    auto max_scalar_desc = ScalarDesc(2);
    auto out_tensor_desc =
        TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_INT64, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnClamp, INPUT(tensor_desc, min_scalar_desc, max_scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_clamp_test, case_self_uncast_out)
{
    auto tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto min_scalar_desc = ScalarDesc(-1.5f);
    auto max_scalar_desc = ScalarDesc(2.5f);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnClamp, INPUT(tensor_desc, min_scalar_desc, max_scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}