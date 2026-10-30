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
#include "random/stateless_random_normal_v2/op_api/aclnn_normal_out.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_normal_float_float_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_normal_float_float SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "l2_normal_float_float TearDown" << std::endl;
    }
};

// float_ND 场景
TEST_F(l2_normal_float_float_test, case_float_ND_001)
{
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    float mean = 1.5f;
    float std = 2.5f;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatFloat, INPUT(mean, std, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// ======================== 正常场景新增 ========================

// 正常场景：float16 out类型，走StatelessRandomNormalV3路径
TEST_F(l2_normal_float_float_test, case_float16_ND_normal)
{
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    float mean = 1.5f;
    float std = 2.5f;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatFloat, INPUT(mean, std, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：double out类型，走StatelessRandomNormalV2 fallback路径
TEST_F(l2_normal_float_float_test, case_double_ND_normal)
{
    auto outDesc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    float mean = 1.5f;
    float std = 2.5f;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatFloat, INPUT(mean, std, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// ======================== 异常场景新增 ========================

// 异常场景：out为nullptr
TEST_F(l2_normal_float_float_test, case_out_nullptr)
{
    float mean = 1.5f;
    float std = 2.5f;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatFloat, INPUT(mean, std, seed, offset), OUTPUT((aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景：超过8维
TEST_F(l2_normal_float_float_test, case_9dim_invalid)
{
    auto outDesc = TensorDesc({2, 3, 2, 2, 3, 2, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float mean = 1.5f;
    float std = 2.5f;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatFloat, INPUT(mean, std, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：不支持的dtype（INT32不在DTYPE_SUPPORT_LIST中）
TEST_F(l2_normal_float_float_test, case_unsupported_dtype)
{
    auto outDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    float mean = 1.5f;
    float std = 2.5f;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnNormalFloatFloat, INPUT(mean, std, seed, offset), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}