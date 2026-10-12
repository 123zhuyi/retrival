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

#include "../../../op_api/aclnn_logspace.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_logspace_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_logspace SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_logspace TearDown" << std::endl; }
};

// 输入float16
TEST_F(l2_logspace_test, aclnnLogSpace_input_float16)
{
    auto start = ScalarDesc(static_cast<float>(3.5));
    auto end = ScalarDesc(static_cast<float>(15.5));
    double base = 10.0;
    int64_t steps = 5;    

    auto outTensor = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSpace, INPUT(start, end, steps, base), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 输入float
TEST_F(l2_logspace_test, aclnnLogSpace_input_float)
{
    auto start = ScalarDesc(static_cast<float>(3.5));
    auto end = ScalarDesc(static_cast<float>(15.5));
    double base = 10.0;
    int64_t steps = 5;

    auto outTensor = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSpace, INPUT(start, end, steps, base), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// 输出bfloat16
TEST_F(l2_logspace_test, aclnnLogSpace_output_bfloat16)
{
    auto start = ScalarDesc(static_cast<float>(3.5));
    auto end = ScalarDesc(static_cast<float>(15.5));
    double base = 10.0;
    int64_t steps = 5;

    auto outTensor = TensorDesc({5}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSpace, INPUT(start, end, steps, base), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// 输入step = 0
TEST_F(l2_logspace_test, aclnnLogSpace_steps_0)
{
    auto start = ScalarDesc(static_cast<float>(3));
    auto end = ScalarDesc(static_cast<float>(15));
    double base = 10.0;
    int64_t steps = 0;

    auto outTensor = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSpace, INPUT(start, end, steps, base), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// 输入step = 1
TEST_F(l2_logspace_test, aclnnLogSpace_steps_1)
{
    auto start = ScalarDesc(static_cast<float>(3));
    auto end = ScalarDesc(static_cast<float>(15));
    double base = 10.0;
    int64_t steps = 1;

    auto outTensor = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSpace, INPUT(start, end, steps, base), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

}

// start 空指针
TEST_F(l2_logspace_test, aclnnLogSpace_start_nullptr)
{
    auto start = ScalarDesc(static_cast<float>(3));
    auto end = ScalarDesc(static_cast<float>(17));
    double base = 10.0;
    int64_t steps = 5;

    auto outTensor = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSpace, INPUT(nullptr, end, steps, base), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// end 空指针
TEST_F(l2_logspace_test, aclnnLogSpace_end_nullptr)
{
    auto start = ScalarDesc(static_cast<float>(3));
    auto end = ScalarDesc(static_cast<float>(17));
    double base = 10.0;
    int64_t steps = 5;

    auto outTensor = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSpace, INPUT(start, nullptr, steps,base), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// steps = 5, start > end
TEST_F(l2_logspace_test, aclnnLogSpace_start_greater_than_end)
{
    auto start = ScalarDesc(static_cast<float>(15));
    auto end = ScalarDesc(static_cast<float>(3));
    double base = 10.0;
    int64_t steps = 5;

    auto outTensor = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSpace, INPUT(start, end, steps, base), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// step = -2 
TEST_F(l2_logspace_test, aclnnLogSpace_steps_less_than_0)
{
    auto start = ScalarDesc(static_cast<float>(3));
    auto end = ScalarDesc(static_cast<float>(15));
    double base = 10.0;
    int64_t steps = -2;

    auto outTensor = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSpace, INPUT(start, end, steps, base), OUTPUT(outTensor));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}