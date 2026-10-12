/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "math/masked_scale/op_api/aclnn_masked_scale.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;

/**
 * @brief MaskedScale算子单元测试Fixture
 * @details 测试aclnnMaskedScale接口的各种场景，包括：
 *          1. 常规类型测试 (Float, Float16, 不同的mask类型)
 *          2. 空指针边界测试
 *          3. 非法参数测试 (shape不匹配、数据类型不支持)
 *          4. Ascend 910B2特定类型测试 (bfloat16)
 *          5. 不同scale值测试、格式测试
 */
class masked_scale_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
        cout << "masked_scale_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "masked_scale_test TearDown" << endl;
    }
};

/**
 * @brief case_1: Float类型masked scale运算测试 (scale=1.0)
 * @details 验证Float类型输入配合Float类型mask时接口正常工作
 */
TEST_F(masked_scale_test, case_1)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    float scale = 1.0f;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_2: Float16类型masked scale运算测试 (scale=2.0)
 * @details 验证Float16类型输入配合Float16类型mask时接口正常工作
 */
TEST_F(masked_scale_test, case_2)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    float scale = 2.0f;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_3: Float类型输入 Uint8类型mask masked scale运算测试 (scale=0.5)
 * @details 验证Float类型输入配合Uint8类型mask时接口正常工作
 */
TEST_F(masked_scale_test, case_3)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    float scale = 0.5f;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_4: Float16类型输入 Uint8类型mask masked scale运算测试 (scale=1.5)
 * @details 验证Float16类型输入配合Uint8类型mask时接口正常工作
 */
TEST_F(masked_scale_test, case_4)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    float scale = 1.5f;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_5: Float类型输入 Int8类型mask masked scale运算测试
 * @details 验证Float类型输入配合Int8类型mask时接口正常工作
 */
TEST_F(masked_scale_test, case_5)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    float scale = 1.0f;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_6: Float16类型输入 Int8类型mask masked scale运算测试
 * @details 验证Float16类型输入配合Int8类型mask时接口正常工作
 */
TEST_F(masked_scale_test, case_6)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    float scale = 1.0f;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_7: 空Tensor masked scale运算测试
 * @details 验证空tensor的masked scale运算接口正常工作
 */
TEST_F(masked_scale_test, case_7)
{
    auto self = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    float scale = 1.0f;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_8: 输入tensor为空指针测试
 * @details 验证输入tensor(self)为nullptr时返回ACLNN_ERR_PARAM_NULLPTR错误码
 */
TEST_F(masked_scale_test, case_8)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    float scale = 1.0f;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(nullptr, mask, scale), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

/**
 * @brief case_9: mask tensor为空指针测试
 * @details 验证mask tensor为nullptr时返回ACLNN_ERR_PARAM_NULLPTR错误码
 */
TEST_F(masked_scale_test, case_9)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    float scale = 1.0f;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, nullptr, scale), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

/**
 * @brief case_10: 输出tensor为空指针测试
 * @details 验证输出tensor为nullptr时返回ACLNN_ERR_PARAM_NULLPTR错误码
 */
TEST_F(masked_scale_test, case_10)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    float scale = 1.0f;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

/**
 * @brief case_11: 输入self与mask shape不匹配测试
 * @details 验证self和mask的shape不一致时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(masked_scale_test, case_11)
{
    auto self = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    float scale = 1.0f;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_12: mask与输出shape不匹配测试
 * @details 验证mask和out的shape不一致时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(masked_scale_test, case_12)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 4}, ACL_BOOL, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    float scale = 1.0f;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_13: self与out数据类型不匹配测试
 * @details 验证self和out的数据类型不一致时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(masked_scale_test, case_13)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    float scale = 1.0f;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_14: self数据类型不支持测试
 * @details 验证Int32类型的self输入时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(masked_scale_test, case_14)
{
    auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    float scale = 1.0f;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_15: mask数据类型不支持测试
 * @details 验证Int32类型的mask输入时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(masked_scale_test, case_15)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    float scale = 1.0f;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_16: 维度超限测试
 * @details 验证输入tensor维度超过支持的最大维度时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(masked_scale_test, case_16)
{
    auto self = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_BOOL, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    float scale = 1.0f;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_17: Float类型非空值范围masked scale运算测试
 * @details 验证Float类型在指定值范围内(-2到2)接口正常工作
 */
TEST_F(masked_scale_test, case_17)
{
    auto self = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto mask = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5});
    auto out = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    float scale = 1.0f;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend910B2_case_18: BFloat16类型masked scale运算测试 (Ascend 910B2)
 * @details 验证BFloat16类型输入配合Float16类型mask时接口正常工作
 */
TEST_F(masked_scale_test, ascend910B2_case_18)
{
    auto self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    float scale = 1.0f;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_19: 负数scale值masked scale运算测试
 * @details 验证scale为负数(-1.0)时接口正常工作
 */
TEST_F(masked_scale_test, case_19)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    float scale = -1.0f;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend950_case_20: FRACTAL_NZ格式测试
 * @details 验证输入为FRACTAL_NZ格式时接口正常工作（Ascend950平台支持）
 */
TEST_F(masked_scale_test, ascend950_case_20)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);
    auto mask = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);
    float scale = 1.0f;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
