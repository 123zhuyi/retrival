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
#include "math/cast/op_api/aclnn_cast.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

/**
 * @brief Cast算子单元测试Fixture
 * @details 测试aclnnCast接口的各种数据类型转换场景，包括：
 *          1. 常规类型转换测试 (Float16, Float, Int32, Int64, Int8, Uint8, Bool)
 *          2. 空指针边界测试
 *          3. 非法参数测试 (shape不匹配、维度超限)
 *          4. Ascend 910B2特定类型测试 (bfloat16)
 *          5. 其他扩展类型测试 (Double, Int16, Uint16, Uint32, Uint64)
 */
class cast_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "cast_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "cast_test TearDown" << endl;
    }
};

/**
 * @brief case_1: Float16 -> Float 类型转换测试
 * @details 验证FP16到FP32的转换接口正常工作
 */
TEST_F(cast_test, case_1)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_2: Float -> Int32 类型转换测试
 * @details 验证FP32到INT32的转换接口正常工作
 */
TEST_F(cast_test, case_2)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_INT32), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_3: Int32 -> Float16 类型转换测试
 * @details 验证INT32到FP16的转换接口正常工作
 */
TEST_F(cast_test, case_3)
{
    auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_4: Int32 -> Float 类型转换测试
 * @details 验证INT32到FP32的转换接口正常工作
 */
TEST_F(cast_test, case_4)
{
    auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_5: Int8 -> Float16 类型转换测试
 * @details 验证INT8到FP16的转换接口正常工作
 */
TEST_F(cast_test, case_5)
{
    auto self = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_6: Uint8 -> Float16 类型转换测试
 * @details 验证UINT8到FP16的转换接口正常工作
 */
TEST_F(cast_test, case_6)
{
    auto self = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_7: Bool -> Float16 类型转换测试
 * @details 验证BOOL到FP16的转换接口正常工作
 */
TEST_F(cast_test, case_7)
{
    auto self = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_8: Float16 -> Uint8 类型转换测试
 * @details 验证FP16到UINT8的转换接口正常工作
 */
TEST_F(cast_test, case_8)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_UINT8), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_9: Float16 -> Int8 类型转换测试
 * @details 验证FP16到INT8的转换接口正常工作
 */
TEST_F(cast_test, case_9)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_INT8), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_10: Float16 -> Bool 类型转换测试
 * @details 验证FP16到BOOL的转换接口正常工作
 */
TEST_F(cast_test, case_10)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_BOOL), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_11: Float16 -> Int64 类型转换测试
 * @details 验证FP16到INT64的转换接口正常工作
 */
TEST_F(cast_test, case_11)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_INT64), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_12: Int64 -> Float16 类型转换测试
 * @details 验证INT64到FP16的转换接口正常工作
 */
TEST_F(cast_test, case_12)
{
    auto self = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_13: Int64 -> Int32 类型转换测试
 * @details 验证INT64到INT32的转换接口正常工作
 */
TEST_F(cast_test, case_13)
{
    auto self = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_INT32), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_14: Int32 -> Int64 类型转换测试
 * @details 验证INT32到INT64的转换接口正常工作
 */
TEST_F(cast_test, case_14)
{
    auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_INT64), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_15: Float -> Bool 类型转换测试
 * @details 验证FP32到BOOL的转换接口正常工作
 */
TEST_F(cast_test, case_15)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_BOOL), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_16: Bool -> Float 类型转换测试
 * @details 验证BOOL到FP32的转换接口正常工作
 */
TEST_F(cast_test, case_16)
{
    auto self = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_17: Bool -> Int32 类型转换测试
 * @details 验证BOOL到INT32的转换接口正常工作
 */
TEST_F(cast_test, case_17)
{
    auto self = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_INT32), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_18: Uint8 -> Int64 类型转换测试
 * @details 验证UINT8到INT64的转换接口正常工作
 */
TEST_F(cast_test, case_18)
{
    auto self = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_INT64), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_19: Int64 -> Uint8 类型转换测试
 * @details 验证INT64到UINT8的转换接口正常工作
 */
TEST_F(cast_test, case_19)
{
    auto self = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_UINT8), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_20: Bool -> Int64 类型转换测试
 * @details 验证BOOL到INT64的转换接口正常工作
 */
TEST_F(cast_test, case_20)
{
    auto self = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_INT64), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_21: Int64 -> Bool 类型转换测试
 * @details 验证INT64到BOOL的转换接口正常工作
 */
TEST_F(cast_test, case_21)
{
    auto self = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_BOOL), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_22: Bool -> Int8 类型转换测试
 * @details 验证BOOL到INT8的转换接口正常工作
 */
TEST_F(cast_test, case_22)
{
    auto self = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_INT8), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_23: Int8 -> Bool 类型转换测试
 * @details 验证INT8到BOOL的转换接口正常工作
 */
TEST_F(cast_test, case_23)
{
    auto self = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_BOOL), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_24: Float16 -> Float16 类型转换测试
 * @details 验证FP16到FP16的相同类型转换接口正常工作
 */
TEST_F(cast_test, case_24)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_25: 空Tensor shape (0维) 类型转换测试
 * @details 验证空tensor的转换接口正常工作
 */
TEST_F(cast_test, case_25)
{
    auto self = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_26: 输入tensor为空指针测试
 * @details 验证输入tensor为nullptr时返回ACLNN_ERR_PARAM_NULLPTR错误码
 */
TEST_F(cast_test, case_26)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(nullptr, ACL_FLOAT), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

/**
 * @brief case_27: 输出tensor为空指针测试
 * @details 验证输出tensor为nullptr时返回ACLNN_ERR_PARAM_NULLPTR错误码
 */
TEST_F(cast_test, case_27)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT), OUTPUT(nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

/**
 * @brief case_28: 输入输出shape不匹配测试
 * @details 验证输入输出shape不一致时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(cast_test, case_28)
{
    auto self = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_29: 维度超限测试
 * @details 验证输入tensor维度超过支持的最大维度时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(cast_test, case_29)
{
    auto self = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief ascend910B2_case_30: BFloat16 -> Float 类型转换测试 (Ascend 910B2)
 * @details 验证BF16到FP32的转换接口正常工作
 */
TEST_F(cast_test, ascend910B2_case_30)
{
    auto self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend910B2_case_33: Float -> BFloat16 类型转换测试 (Ascend 910B2)
 * @details 验证FP32到BF16的转换接口正常工作
 */
TEST_F(cast_test, ascend910B2_case_33)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_BF16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend910B2_case_34: BFloat16 -> Int32 类型转换测试 (Ascend 910B2)
 * @details 验证BF16到INT32的转换接口正常工作
 */
TEST_F(cast_test, ascend910B2_case_34)
{
    auto self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_INT32), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend910B2_case_35: Int32 -> BFloat16 类型转换测试 (Ascend 910B2)
 * @details 验证INT32到BF16的转换接口正常工作
 */
TEST_F(cast_test, ascend910B2_case_35)
{
    auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_BF16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend910B2_case_36: BFloat16 -> Float16 类型转换测试 (Ascend 910B2)
 * @details 验证BF16到FP16的转换接口正常工作
 */
TEST_F(cast_test, ascend910B2_case_36)
{
    auto self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend910B2_case_37: Float16 -> BFloat16 类型转换测试 (Ascend 910B2)
 * @details 验证FP16到BF16的转换接口正常工作
 */
TEST_F(cast_test, ascend910B2_case_37)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_BF16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend910B2_case_38: BFloat16 -> Bool 类型转换测试 (Ascend 910B2)
 * @details 验证BF16到BOOL的转换接口正常工作
 */
TEST_F(cast_test, ascend910B2_case_38)
{
    auto self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_BOOL), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend910B2_case_39: Bool -> BFloat16 类型转换测试 (Ascend 910B2)
 * @details 验证BOOL到BF16的转换接口正常工作
 */
TEST_F(cast_test, ascend910B2_case_39)
{
    auto self = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_BF16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend910B2_case_40: BFloat16 -> Int8 类型转换测试 (Ascend 910B2)
 * @details 验证BF16到INT8的转换接口正常工作
 */
TEST_F(cast_test, ascend910B2_case_40)
{
    auto self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_INT8), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend910B2_case_41: Int8 -> BFloat16 类型转换测试 (Ascend 910B2)
 * @details 验证INT8到BF16的转换接口正常工作
 */
TEST_F(cast_test, ascend910B2_case_41)
{
    auto self = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_BF16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend910B2_case_42: BFloat16 -> Uint8 类型转换测试 (Ascend 910B2)
 * @details 验证BF16到UINT8的转换接口正常工作
 */
TEST_F(cast_test, ascend910B2_case_42)
{
    auto self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_UINT8), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend910B2_case_43: Uint8 -> BFloat16 类型转换测试 (Ascend 910B2)
 * @details 验证UINT8到BF16的转换接口正常工作
 */
TEST_F(cast_test, ascend910B2_case_43)
{
    auto self = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_BF16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend910B2_case_44: BFloat16 -> Int64 类型转换测试 (Ascend 910B2)
 * @details 验证BF16到INT64的转换接口正常工作
 */
TEST_F(cast_test, ascend910B2_case_44)
{
    auto self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_INT64), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend910B2_case_45: Int64 -> BFloat16 类型转换测试 (Ascend 910B2)
 * @details 验证INT64到BF16的转换接口正常工作
 */
TEST_F(cast_test, ascend910B2_case_45)
{
    auto self = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_BF16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend910B2_case_46: BFloat16 -> BFloat16 类型转换测试 (Ascend 910B2)
 * @details 验证BF16到BF16的相同类型转换接口正常工作
 */
TEST_F(cast_test, ascend910B2_case_46)
{
    auto self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_BF16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_47: Double -> Float 类型转换测试
 * @details 验证DOUBLE到FP32的转换接口正常工作
 */
TEST_F(cast_test, case_47)
{
    auto self = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_48: Double -> Int32 类型转换测试
 * @details 验证DOUBLE到INT32的转换接口正常工作
 */
TEST_F(cast_test, case_48)
{
    auto self = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_INT32), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_49: Double -> Int64 类型转换测试
 * @details 验证DOUBLE到INT64的转换接口正常工作
 */
TEST_F(cast_test, case_49)
{
    auto self = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_INT64), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_50: Int16 -> Float16 类型转换测试
 * @details 验证INT16到FP16的转换接口正常工作
 */
TEST_F(cast_test, case_50)
{
    auto self = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_51: Uint16 -> Float16 类型转换测试
 * @details 验证UINT16到FP16的转换接口正常工作
 */
TEST_F(cast_test, case_51)
{
    auto self = TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_52: Uint32 -> Float16 类型转换测试
 * @details 验证UINT32到FP16的转换接口正常工作
 */
TEST_F(cast_test, case_52)
{
    auto self = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_53: Uint64 -> Float16 类型转换测试
 * @details 验证UINT64到FP16的转换接口正常工作
 */
TEST_F(cast_test, case_53)
{
    auto self = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_54: Int16 -> Float 类型转换测试
 * @details 验证INT16到FP32的转换接口正常工作
 */
TEST_F(cast_test, case_54)
{
    auto self = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_55: Uint16 -> Float 类型转换测试
 * @details 验证UINT16到FP32的转换接口正常工作
 */
TEST_F(cast_test, case_55)
{
    auto self = TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_56: Uint32 -> Float 类型转换测试
 * @details 验证UINT32到FP32的转换接口正常工作
 */
TEST_F(cast_test, case_56)
{
    auto self = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_57: Uint64 -> Float 类型转换测试
 * @details 验证UINT64到FP32的转换接口正常工作
 */
TEST_F(cast_test, case_57)
{
    auto self = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_FLOAT), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_58: Float -> Int16 类型转换测试
 * @details 验证FP32到INT16的转换接口正常工作
 */
TEST_F(cast_test, case_58)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_INT16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_59: Float -> Uint16 类型转换测试
 * @details 验证FP32到UINT16的转换接口正常工作
 */
TEST_F(cast_test, case_59)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_UINT16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_60: Float -> Uint32 类型转换测试
 * @details 验证FP32到UINT32的转换接口正常工作
 */
TEST_F(cast_test, case_60)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_UINT32), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_61: Float -> Uint64 类型转换测试
 * @details 验证FP32到UINT64的转换接口正常工作
 */
TEST_F(cast_test, case_61)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_UINT64), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_62: Float16 -> Int16 类型转换测试
 * @details 验证FP16到INT16的转换接口正常工作
 */
TEST_F(cast_test, case_62)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_INT16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_63: Float16 -> Uint16 类型转换测试
 * @details 验证FP16到UINT16的转换接口正常工作
 */
TEST_F(cast_test, case_63)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_UINT16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_64: Float16 -> Uint32 类型转换测试
 * @details 验证FP16到UINT32的转换接口正常工作
 */
TEST_F(cast_test, case_64)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_UINT32), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_65: Float16 -> Uint64 类型转换测试
 * @details 验证FP16到UINT64的转换接口正常工作
 */
TEST_F(cast_test, case_65)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_UINT64), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_66: Int32 -> Int16 类型转换测试
 * @details 验证INT32到INT16的转换接口正常工作
 */
TEST_F(cast_test, case_66)
{
    auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_INT16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_67: Int32 -> Uint16 类型转换测试
 * @details 验证INT32到UINT16的转换接口正常工作
 */
TEST_F(cast_test, case_67)
{
    auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_UINT16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_68: Int32 -> Uint32 类型转换测试
 * @details 验证INT32到UINT32的转换接口正常工作
 */
TEST_F(cast_test, case_68)
{
    auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_UINT32), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_69: Int32 -> Uint64 类型转换测试
 * @details 验证INT32到UINT64的转换接口正常工作
 */
TEST_F(cast_test, case_69)
{
    auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_UINT64), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_70: Int64 -> Int16 类型转换测试
 * @details 验证INT64到INT16的转换接口正常工作
 */
TEST_F(cast_test, case_70)
{
    auto self = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_INT16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_71: Int64 -> Uint16 类型转换测试
 * @details 验证INT64到UINT16的转换接口正常工作
 */
TEST_F(cast_test, case_71)
{
    auto self = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_UINT16), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_72: Int64 -> Uint32 类型转换测试
 * @details 验证INT64到UINT32的转换接口正常工作
 */
TEST_F(cast_test, case_72)
{
    auto self = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_UINT32), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_73: Int64 -> Uint64 类型转换测试
 * @details 验证INT64到UINT64的转换接口正常工作
 */
TEST_F(cast_test, case_73)
{
    auto self = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCast, INPUT(self, ACL_UINT64), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
