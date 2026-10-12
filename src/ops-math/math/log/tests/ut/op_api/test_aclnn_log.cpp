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
#include "math/log/op_api/aclnn_log.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

/**
 * @brief Log算子单元测试Fixture
 * @details 测试aclnnLog接口的各种数据类型场景，包括：
 *          1. 常规类型测试 (Float16, Float, Double, Int, Uint, Bool)
 *          2. 空指针边界测试
 *          3. 非法参数测试 (shape不匹配、维度超限、不支持的数据类型)
 *          4. Ascend 910B2特定类型测试 (bfloat16)
 *          5. 不同输出类型测试
 */
class log_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "log_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "log_test TearDown" << endl;
    }
};

/**
 * @brief case_1: Float16类型对数运算测试
 * @details 验证Float16类型输入输出时接口正常工作
 */
TEST_F(log_test, case_1)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_2: Float类型对数运算测试
 * @details 验证Float类型输入输出时接口正常工作
 */
TEST_F(log_test, case_2)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_3: Double类型对数运算测试
 * @details 验证Double类型输入输出时接口正常工作
 */
TEST_F(log_test, case_3)
{
    auto self = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_4: Int32 -> Float类型转换对数运算测试
 * @details 验证Int32输入转Float输出时接口正常工作
 */
TEST_F(log_test, case_4)
{
    auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_5: Int64 -> Float类型转换对数运算测试
 * @details 验证Int64输入转Float输出时接口正常工作
 */
TEST_F(log_test, case_5)
{
    auto self = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_6: Int8 -> Float类型转换对数运算测试
 * @details 验证Int8输入转Float输出时接口正常工作
 */
TEST_F(log_test, case_6)
{
    auto self = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_7: Uint8 -> Float类型转换对数运算测试
 * @details 验证Uint8输入转Float输出时接口正常工作
 */
TEST_F(log_test, case_7)
{
    auto self = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_8: Bool -> Float类型转换对数运算测试
 * @details 验证Bool输入转Float输出时接口正常工作
 */
TEST_F(log_test, case_8)
{
    auto self = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_9: 空Tensor shape (0维) 对数运算测试
 * @details 验证空tensor的对数运算接口正常工作
 */
TEST_F(log_test, case_9)
{
    auto self = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_10: 输入tensor为空指针测试
 * @details 验证输入tensor为nullptr时返回ACLNN_ERR_PARAM_NULLPTR错误码
 */
TEST_F(log_test, case_10)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog, INPUT(nullptr), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

/**
 * @brief case_11: 输出tensor为空指针测试
 * @details 验证输出tensor为nullptr时返回ACLNN_ERR_PARAM_NULLPTR错误码
 */
TEST_F(log_test, case_11)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog, INPUT(self), OUTPUT(nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

/**
 * @brief case_12: 输入输出shape不匹配测试
 * @details 验证输入输出shape不一致时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(log_test, case_12)
{
    auto self = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_13: Int16 -> Float类型转换对数运算测试
 * @details 验证Int16输入转Float输出时接口正常工作
 */
TEST_F(log_test, case_13)
{
    auto self = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend910B2_case_14: BFloat16类型对数运算测试 (Ascend 910B2)
 * @details 验证BFloat16类型输入输出时接口正常工作
 */
TEST_F(log_test, ascend910B2_case_14)
{
    auto self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_15: Float16 -> Float类型转换对数运算测试
 * @details 验证Float16输入转Float输出时接口正常工作
 */
TEST_F(log_test, case_15)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_16: Float类型非空值范围对数运算测试
 * @details 验证Float类型在指定值范围内(1-10)接口正常工作
 */
TEST_F(log_test, case_16)
{
    auto self = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(1, 10);
    auto out = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(1, 10);

    auto ut = OP_API_UT(aclnnLog, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_17: Float16 -> Double类型转换对数运算测试
 * @details 验证Float16输入转Double输出时接口正常工作
 */
TEST_F(log_test, case_17)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_18: Uint16类型对数运算测试
 * @details 验证Uint16类型输入时返回ACLNN_ERR_PARAM_INVALID错误码(log不支持Uint16)
 */
TEST_F(log_test, case_18)
{
    auto self = TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_19: Uint32类型对数运算测试
 * @details 验证Uint32类型输入时返回ACLNN_ERR_PARAM_INVALID错误码(log不支持Uint32)
 */
TEST_F(log_test, case_19)
{
    auto self = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_20: 维度超限测试
 * @details 验证输入tensor维度超过支持的最大维度时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(log_test, case_20)
{
    auto self = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLog, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
