/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "../../../op_api/aclnn_unfold_grad.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_unfold_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "unfold_grad_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "unfold_grad_test TearDown" << endl;
    }
};

TEST_F(l2_unfold_grad_test, case_last_dim_FLOAT)
{
    auto grad_out_desc = TensorDesc({1, 658, 320, 20}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> input_sizes = {1, 658, 658};
    auto input_sizes_desc = IntArrayDesc(input_sizes);
    auto out_desc = TensorDesc({1, 658, 658}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnUnfoldGrad,
        INPUT(grad_out_desc, input_sizes_desc, 2, 20, 2),
        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 1000000;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_unfold_grad_test, case_last_second_dim_FLOAT16)
{
    auto grad_out_desc = TensorDesc({1, 320, 658, 20}, ACL_FLOAT16, ACL_FORMAT_ND);
    vector<int64_t> input_sizes = {1, 658, 658};
    auto input_sizes_desc = IntArrayDesc(input_sizes);
    auto out_desc = TensorDesc({1, 658, 658}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnUnfoldGrad,
        INPUT(grad_out_desc, input_sizes_desc, 1, 20, 2),
        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 1000000;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_unfold_grad_test, case_last_third_dim_FLOAT16)
{
    auto grad_out_desc = TensorDesc({1, 320, 2, 658, 20}, ACL_FLOAT16, ACL_FORMAT_ND);
    vector<int64_t> input_sizes = {1, 658, 2, 658};
    auto input_sizes_desc = IntArrayDesc(input_sizes);
    auto out_desc = TensorDesc({1, 658, 2, 658}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnUnfoldGrad,
        INPUT(grad_out_desc, input_sizes_desc, 1, 20, 2),
        OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 1000000;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
