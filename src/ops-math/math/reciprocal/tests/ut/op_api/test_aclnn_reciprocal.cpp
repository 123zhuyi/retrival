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
#include "math/reciprocal/op_api/aclnn_reciprocal.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;

/**
 * @brief Reciprocal算子单元测试Fixture
 * @details 测试aclnnReciprocal接口的各种数据类型场景，包括：
 *          1. 常规类型测试 (Float16, Float, Double, Int, Uint, Bool)
 *          2. 空指针边界测试
 *          3. 非法参数测试 (shape不匹配、格式不支持、维度超限)
 *          4. Ascend 910B2特定类型测试 (bfloat16)
 *          5. 不同输出类型测试
 */
class reciprocal_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "reciprocal_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "reciprocal_test TearDown" << endl;
    }
};

/**
 * @brief case_1: Float16类型求倒数运算测试
 * @details 验证Float16类型输入输出时接口正常工作
 */
TEST_F(reciprocal_test, case_1)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_2: Float类型求倒数运算测试
 * @details 验证Float类型输入输出时接口正常工作
 */
TEST_F(reciprocal_test, case_2)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_3: Double类型求倒数运算测试
 * @details 验证Double类型输入输出时接口正常工作
 */
TEST_F(reciprocal_test, case_3)
{
    auto self = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_4: Int8 -> Float类型转换求倒数运算测试
 * @details 验证Int8输入转Float输出时接口正常工作
 */
TEST_F(reciprocal_test, case_4)
{
    auto self = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_5: Int16 -> Float类型转换求倒数运算测试
 * @details 验证Int16输入转Float输出时接口正常工作
 */
TEST_F(reciprocal_test, case_5)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto self = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_6: Int32 -> Float类型转换求倒数运算测试
 * @details 验证Int32输入转Float输出时接口正常工作
 */
TEST_F(reciprocal_test, case_6)
{
    auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_7: Int64 -> Float类型转换求倒数运算测试
 * @details 验证Int64输入转Float输出时接口正常工作
 */
TEST_F(reciprocal_test, case_7)
{
    auto self = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_8: Uint8 -> Float类型转换求倒数运算测试
 * @details 验证Uint8输入转Float输出时接口正常工作
 */
TEST_F(reciprocal_test, case_8)
{
    auto self = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_9: Bool -> Float类型转换求倒数运算测试
 * @details 验证Bool输入转Float输出时接口正常工作
 */
TEST_F(reciprocal_test, case_9)
{
    auto self = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_10: 空Tensor shape (0维) 求倒数运算测试
 * @details 验证空tensor的求倒数运算接口正常工作
 */
TEST_F(reciprocal_test, case_10)
{
    auto self = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_11: 输入tensor为空指针测试
 * @details 验证输入tensor为nullptr时返回ACLNN_ERR_PARAM_NULLPTR错误码
 */
TEST_F(reciprocal_test, case_11)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(nullptr), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

/**
 * @brief case_12: 输出tensor为空指针测试
 * @details 验证输出tensor为nullptr时返回ACLNN_ERR_PARAM_NULLPTR错误码
 */
TEST_F(reciprocal_test, case_12)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

/**
 * @brief case_13: 输入输出shape不匹配测试
 * @details 验证输入输出shape不一致时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(reciprocal_test, case_13)
{
    auto self = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_14: 输入tensor格式不支持测试
 * @details 验证输入为FRACTAL_NZ格式时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(reciprocal_test, case_14)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_15: 输出tensor格式不支持测试
 * @details 验证输出为FRACTAL_NZ格式时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(reciprocal_test, case_15)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_16: Float类型非空值范围求倒数运算测试
 * @details 验证Float类型在指定值范围内(1-10)接口正常工作
 */
TEST_F(reciprocal_test, case_16)
{
    auto self = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(1, 10);
    auto out = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(1, 10);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_17: 维度超限测试
 * @details 验证输入tensor维度超过支持的最大维度时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(reciprocal_test, case_17)
{
    auto self = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief ascend910B2_case_18: BFloat16类型求倒数运算测试 (Ascend 910B2)
 * @details 验证BFloat16类型输入输出时接口正常工作
 */
TEST_F(reciprocal_test, ascend910B2_case_18)
{
    auto self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_19: Float16 -> Double类型转换求倒数运算测试
 * @details 验证Float16输入转Double输出时接口正常工作
 */
TEST_F(reciprocal_test, case_19)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_20: Uint16类型求倒数运算测试
 * @details 验证Uint16类型输入时返回ACLNN_ERR_PARAM_INVALID错误码(reciprocal不支持Uint16)
 */
TEST_F(reciprocal_test, case_20)
{
    auto self = TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_21: Uint32类型求倒数运算测试
 * @details 验证Uint32类型输入时返回ACLNN_ERR_PARAM_INVALID错误码(reciprocal不支持Uint32)
 */
TEST_F(reciprocal_test, case_21)
{
    auto self = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_22: Uint64类型求倒数运算测试
 * @details 验证Uint64类型输入时返回ACLNN_ERR_PARAM_INVALID错误码(reciprocal不支持Uint64)
 */
TEST_F(reciprocal_test, case_22)
{
    auto self = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_23: Float类型空Tensor求倒数运算测试
 * @details 验证Float类型空tensor的求倒数运算接口正常工作
 */
TEST_F(reciprocal_test, case_23)
{
    auto self = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnReciprocal, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
