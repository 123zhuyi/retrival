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
#include "math/round/op_api/aclnn_round.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

/**
 * @brief Round算子单元测试Fixture
 * @details 测试aclnnRound/aclnnRoundDecimals接口的各种场景，包括：
 *          1. 常规类型测试 (Float16, Float, Double, Int32, Int64)
 *          2. 空指针边界测试
 *          3. 非法参数测试 (shape不匹配、数据类型不支持、格式不支持、维度超限)
 *          4. Ascend 910B2特定类型测试 (bfloat16)
 *          5. 不同decimals参数测试、原地操作测试
 */
class round_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "round_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "round_test TearDown" << endl;
    }
};

/**
 * @brief case_1: Float16类型四舍五入运算测试
 * @details 验证Float16类型输入输出时接口正常工作
 */
TEST_F(round_test, case_1)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRound, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_2: 空Tensor shape (0维) 四舍五入运算测试
 * @details 验证空tensor的四舍五入运算接口正常工作
 */
TEST_F(round_test, case_2)
{
    auto self = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRound, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_3: 输入tensor为空指针测试
 * @details 验证输入tensor为nullptr时返回ACLNN_ERR_PARAM_NULLPTR错误码
 */
TEST_F(round_test, case_3)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRound, INPUT(nullptr), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

/**
 * @brief case_4: 输出tensor为空指针测试
 * @details 验证输出tensor为nullptr时返回ACLNN_ERR_PARAM_NULLPTR错误码
 */
TEST_F(round_test, case_4)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRound, INPUT(self), OUTPUT(nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

/**
 * @brief case_5: 输入输出shape不匹配测试
 * @details 验证输入输出shape不一致时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(round_test, case_5)
{
    auto self = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRound, INPUT(self), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_7: Int16类型四舍五入运算测试
 * @details 验证Int16类型输入时返回ACLNN_ERR_PARAM_INVALID错误码(round不支持Int16)
 */
TEST_F(round_test, case_7)
{
    auto self = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRound, INPUT(self), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_8: Float16 -> Float类型转换四舍五入运算测试
 * @details 验证输入输出数据类型不一致时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(round_test, case_8)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRound, INPUT(self), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_9: 维度超限测试
 * @details 验证输入tensor维度超过支持的最大维度时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(round_test, case_9)
{
    auto self = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRound, INPUT(self), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_10: Float16类型指定小数位数四舍五入运算测试
 * @details 验证Float16输入Float输出时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(round_test, case_10)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t decimals = 1;

    auto ut = OP_API_UT(aclnnRoundDecimals, INPUT(self, decimals), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_11: Float类型指定小数位数四舍五入运算测试 (decimals=1)
 * @details 验证Float类型输入decimals=1时接口正常工作
 */
TEST_F(round_test, case_11)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t decimals = 1;

    auto ut = OP_API_UT(aclnnRoundDecimals, INPUT(self, decimals), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_12: Float类型指定小数位数四舍五入运算测试 (decimals=2)
 * @details 验证Float类型输入decimals=2时接口正常工作
 */
TEST_F(round_test, case_12)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t decimals = 2;

    auto ut = OP_API_UT(aclnnRoundDecimals, INPUT(self, decimals), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_13: Float类型原地指定小数位数四舍五入运算测试
 * @details 验证Float类型原地操作decimals=2时接口正常工作
 */
TEST_F(round_test, case_13)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t decimals = 2;
    auto ut = OP_API_UT(aclnnInplaceRoundDecimals, INPUT(self, decimals), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend910B2_case_15: BFloat16类型四舍五入运算测试 (Ascend 910B2)
 * @details 验证BFloat16类型输入输出时接口正常工作
 */
TEST_F(round_test, ascend910B2_case_15)
{
    auto self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRound, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend910B2_case_16: BFloat16类型指定小数位数四舍五入运算测试 (Ascend 910B2)
 * @details 验证BFloat16类型输入decimals=2时接口正常工作
 */
TEST_F(round_test, ascend910B2_case_16)
{
    auto self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    int64_t decimals = 2;

    auto ut = OP_API_UT(aclnnRoundDecimals, INPUT(self, decimals), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_17: 输入tensor格式不支持测试
 * @details 验证输入为FRACTAL_NZ格式时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(round_test, case_17)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRound, INPUT(self), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_18: 输出tensor格式不支持测试
 * @details 验证输出为FRACTAL_NZ格式时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(round_test, case_18)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);

    auto ut = OP_API_UT(aclnnRound, INPUT(self), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_19: Float类型原地四舍五入运算测试
 * @details 验证Float类型原地操作时接口正常工作
 */
TEST_F(round_test, case_19)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceRound, INPUT(self), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_20: Float类型非空tensor指定小数位数四舍五入运算测试
 * @details 验证Float类型decimals=1时接口正常工作
 */
TEST_F(round_test, case_20)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t decimals = 1;

    auto ut = OP_API_UT(aclnnRoundDecimals, INPUT(self, decimals), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_21: Float类型非空tensor四舍五入运算测试
 * @details 验证Float类型四舍五入时接口正常工作
 */
TEST_F(round_test, case_21)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRound, INPUT(self), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_22: Float16类型原地指定小数位数四舍五入运算测试
 * @details 验证Float16类型原地操作decimals=1时接口正常工作
 */
TEST_F(round_test, case_22)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t decimals = 1;

    auto ut = OP_API_UT(aclnnInplaceRoundDecimals, INPUT(self, decimals), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_23: Int32类型四舍五入运算测试
 * @details 验证Int32类型输入输出时接口正常工作
 */
TEST_F(round_test, case_23)
{
    auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRound, INPUT(self), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_24: Int32类型指定小数位数四舍五入运算测试
 * @details 验证Int32类型输入decimals=2时接口正常工作
 */
TEST_F(round_test, case_24)
{
    auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    int64_t decimals = 2;

    auto ut = OP_API_UT(aclnnRoundDecimals, INPUT(self, decimals), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_25: Int64类型四舍五入运算测试
 * @details 验证Int64类型输入输出时接口正常工作
 */
TEST_F(round_test, case_25)
{
    auto self = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRound, INPUT(self), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_26: Double类型指定小数位数四舍五入运算测试
 * @details 验证Double类型输入decimals=2时接口正常工作
 */
TEST_F(round_test, case_26)
{
    auto self = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    int64_t decimals = 2;

    auto ut = OP_API_UT(aclnnRoundDecimals, INPUT(self, decimals), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_27: Float16类型空tensor指定小数位数四舍五入运算测试
 * @details 验证Float16类型空tensor decimas=1时接口正常工作
 */
TEST_F(round_test, case_27)
{
    auto self = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t decimals = 1;

    auto ut = OP_API_UT(aclnnRoundDecimals, INPUT(self, decimals), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
