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
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "../../../op_api/aclnn_grouped_bias_add_grad.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;


class l2_grouped_bias_add_grad_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GroupedBiasAddGrad Test Setup" << std::endl; }
    static void TearDownTestCase() { std::cout << "GroupedBiasAddGrad Test TearDown" << std::endl; }
};

TEST_F(l2_grouped_bias_add_grad_test, grouped_bias_add_grad_testcase_000_no_grp_idx_in_nullptr_check)
{
    auto gradYDesc = TensorDesc({2, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1,1);
    auto gradBiasDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int64_t groupIdxType = 0;
    auto ut = OP_API_UT(aclnnGroupedBiasAddGradV2, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr, groupIdxType), OUTPUT(gradBiasDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_grouped_bias_add_grad_test, grouped_bias_add_grad_testcase_001_no_grp_idx_out_nullptr_check)
{
    auto gradYDesc = TensorDesc({2, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1,1);
    auto gradBiasDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int64_t groupIdxType = 0;
    auto ut = OP_API_UT(aclnnGroupedBiasAddGradV2, INPUT(gradYDesc, (aclTensor*)nullptr, groupIdxType), OUTPUT((aclTensor*)nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}


TEST_F(l2_grouped_bias_add_grad_test, grouped_bias_add_grad_testcase_002_no_grp_idx)
{
    auto gradYDesc = TensorDesc({2, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1,1);
    auto gradBiasDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int64_t groupIdxType = 0;
    auto ut = OP_API_UT(aclnnGroupedBiasAddGradV2, INPUT(gradYDesc, (aclTensor*)nullptr, groupIdxType), OUTPUT(gradBiasDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST_F(l2_grouped_bias_add_grad_test, grouped_bias_add_grad_testcase_003_no_grp_idx_in_shape_invalid)
{
    auto gradYDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1,1);
    auto gradBiasDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int64_t groupIdxType = 0;
    auto ut = OP_API_UT(aclnnGroupedBiasAddGradV2, INPUT(gradYDesc, (aclTensor*)nullptr, groupIdxType), OUTPUT(gradBiasDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grouped_bias_add_grad_test, grouped_bias_add_grad_testcase_004_no_grp_idx_out_invalid_shape)
{
    auto gradYDesc = TensorDesc({2, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradBiasDesc = TensorDesc({2, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int64_t groupIdxType = 0;
    auto ut = OP_API_UT(aclnnGroupedBiasAddGradV2, INPUT(gradYDesc, (aclTensor*)nullptr, groupIdxType), OUTPUT(gradBiasDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grouped_bias_add_grad_test, grouped_bias_add_grad_testcase_005_no_grp_idx_shape_size_exceed)
{
    auto gradYDesc = TensorDesc({2049, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradBiasDesc = TensorDesc({2, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int64_t groupIdxType = 0;
    auto ut = OP_API_UT(aclnnGroupedBiasAddGradV2, INPUT(gradYDesc, (aclTensor*)nullptr, groupIdxType), OUTPUT(gradBiasDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grouped_bias_add_grad_test, grouped_bias_add_grad_testcase_006_no_grp_idx_invalid_dtype)
{
    auto gradYDesc = TensorDesc({2, 1, 5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradBiasDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int64_t groupIdxType = 0;
    auto ut = OP_API_UT(aclnnGroupedBiasAddGradV2, INPUT(gradYDesc, (aclTensor*)nullptr, groupIdxType), OUTPUT(gradBiasDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grouped_bias_add_grad_test, grouped_bias_add_grad_testcase_007_grp_idx_int32)
{
    auto gradYDesc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto groupIdxOptional = TensorDesc({2,}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 4});
    auto gradBiasDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int64_t groupIdxType = 0;
    auto ut = OP_API_UT(aclnnGroupedBiasAddGradV2, INPUT(gradYDesc, groupIdxOptional, groupIdxType), OUTPUT(gradBiasDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST_F(l2_grouped_bias_add_grad_test, grouped_bias_add_grad_testcase_008_grp_idx_int64)
{
    auto gradYDesc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto groupIdxOptional = TensorDesc({2,}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{2, 4});
    auto gradBiasDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int64_t groupIdxType = 0;
    auto ut = OP_API_UT(aclnnGroupedBiasAddGradV2, INPUT(gradYDesc, groupIdxOptional, groupIdxType), OUTPUT(gradBiasDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST_F(l2_grouped_bias_add_grad_test, grouped_bias_add_grad_testcase_009_grp_idx_int64_attr_invalid)
{
    auto gradYDesc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto groupIdxOptional = TensorDesc({2,}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{2, 4});
    auto gradBiasDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int64_t groupIdxType = -1;
    auto ut = OP_API_UT(aclnnGroupedBiasAddGradV2, INPUT(gradYDesc, groupIdxOptional, groupIdxType), OUTPUT(gradBiasDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
