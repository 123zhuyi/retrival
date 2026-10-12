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

#include "math/add/op_api/aclnn_add_v3.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_add_v3_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "add_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "add_test TearDown" << endl;
    }
};

TEST_F(l2_add_v3_test, case_1)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto scalar_desc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddV3, INPUT(scalar_desc, tensor_desc, scalar_desc), OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_add_v3_test, case_nullptr)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut =
        OP_API_UT(aclnnAddV3, INPUT((aclScalar*)nullptr, (aclTensor*)nullptr, (aclScalar*)nullptr), OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 计算图三
TEST_F(l2_add_v3_test, case_001)
{
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(other_tensor_desc);
    auto scalar_desc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddV3, INPUT(scalar_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_add_v3_test, Ascend910B2_case_002)
{
    auto other_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(0, 100);
    auto out_tensor_desc = TensorDesc(other_tensor_desc).Precision(0.001, 0.001);
    auto scalar_desc = ScalarDesc(1.2f);

    auto ut = OP_API_UT(aclnnAddV3, INPUT(scalar_desc, other_tensor_desc, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}