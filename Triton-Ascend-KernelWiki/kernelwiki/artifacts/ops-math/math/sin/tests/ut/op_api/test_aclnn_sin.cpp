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
#include "math/sin/op_api/aclnn_sin.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;

/**
 * @brief Sin算子单元测试Fixture
 * @details 测试aclnnSin接口的各种数据类型场景，包括：
 *          1. 常规类型测试 (Float16, Float, Double, Int, Uint, Bool, Complex)
 *          2. 空指针边界测试
 *          3. 非法参数测试 (shape不匹配、维度超限、不支持的数据类型)
 *          4. Ascend 910B2特定类型测试 (bfloat16)
 *          5. 不同输出类型测试
 */
class sin_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "sin_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "sin_test TearDown" << endl;
    }
};

/**
 * @brief case_1: Float16类型正弦运算测试
 * @details 验证Float16类型输入输出时接口正常工作
 */
TEST_F(sin_test, case_1)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_2: Float类型正弦运算测试
 * @details 验证Float类型输入输出时接口正常工作
 */
TEST_F(sin_test, case_2)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_3: Double类型正弦运算测试
 * @details 验证Double类型输入输出时接口正常工作
 */
TEST_F(sin_test, case_3)
{
    auto self = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_4: Int8 -> Float类型转换正弦运算测试
 * @details 验证Int8输入转Float输出时接口正常工作
 */
TEST_F(sin_test, case_4)
{
    auto self = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_5: Int16 -> Float类型转换正弦运算测试
 * @details 验证Int16输入转Float输出时接口正常工作
 */
TEST_F(sin_test, case_5)
{
    auto self = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_6: Int32 -> Float类型转换正弦运算测试
 * @details 验证Int32输入转Float输出时接口正常工作
 */
TEST_F(sin_test, case_6)
{
    auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_7: Int64 -> Float类型转换正弦运算测试
 * @details 验证Int64输入转Float输出时接口正常工作
 */
TEST_F(sin_test, case_7)
{
    auto self = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_8: Uint8 -> Float类型转换正弦运算测试
 * @details 验证Uint8输入转Float输出时接口正常工作
 */
TEST_F(sin_test, case_8)
{
    auto self = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_9: Bool -> Float类型转换正弦运算测试
 * @details 验证Bool输入转Float输出时接口正常工作
 */
TEST_F(sin_test, case_9)
{
    auto self = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_10: 空Tensor shape (0维) 正弦运算测试
 * @details 验证空tensor的正弦运算接口正常工作
 */
TEST_F(sin_test, case_10)
{
    auto self = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_11: 输入tensor为空指针测试
 * @details 验证输入tensor为nullptr时返回ACLNN_ERR_PARAM_NULLPTR错误码
 */
TEST_F(sin_test, case_11)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(nullptr), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

/**
 * @brief case_12: 输出tensor为空指针测试
 * @details 验证输出tensor为nullptr时返回ACLNN_ERR_PARAM_NULLPTR错误码
 */
TEST_F(sin_test, case_12)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

/**
 * @brief case_13: 输入输出shape不匹配测试
 * @details 验证输入输出shape不一致时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(sin_test, case_13)
{
    auto self = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_14: Float类型非空值范围正弦运算测试
 * @details 验证Float类型在指定值范围内(-3.14到3.14)接口正常工作
 */
TEST_F(sin_test, case_14)
{
    auto self = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-3.14, 3.14);
    auto out = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-3.14, 3.14);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_15: 维度超限测试
 * @details 验证输入tensor维度超过支持的最大维度时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(sin_test, case_15)
{
    auto self = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief ascend910B2_case_16: BFloat16类型正弦运算测试 (Ascend 910B2)
 * @details 验证BFloat16类型输入输出时接口正常工作
 */
TEST_F(sin_test, ascend910B2_case_16)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    auto self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_17: Float16 -> Double类型转换正弦运算测试
 * @details 验证Float16输入转Double输出时接口正常工作
 */
TEST_F(sin_test, case_17)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_18: Uint16类型正弦运算测试
 * @details 验证Uint16类型输入时返回ACLNN_ERR_PARAM_INVALID错误码(sin不支持Uint16)
 */
TEST_F(sin_test, case_18)
{
    auto self = TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_19: Uint32类型正弦运算测试
 * @details 验证Uint32类型输入时返回ACLNN_ERR_PARAM_INVALID错误码(sin不支持Uint32)
 */
TEST_F(sin_test, case_19)
{
    auto self = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_20: Uint64类型正弦运算测试
 * @details 验证Uint64类型输入时返回ACLNN_ERR_PARAM_INVALID错误码(sin不支持Uint64)
 */
TEST_F(sin_test, case_20)
{
    auto self = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief case_21: Float类型空Tensor正弦运算测试
 * @details 验证Float类型空tensor的正弦运算接口正常工作
 */
TEST_F(sin_test, case_21)
{
    auto self = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_22: Complex64类型正弦运算测试
 * @details 验证Complex64类型输入输出时接口正常工作
 */
TEST_F(sin_test, case_22)
{
    auto self = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief case_23: Complex128类型正弦运算测试
 * @details 验证Complex128类型输入输出时接口正常工作
 */
TEST_F(sin_test, case_23)
{
    auto self = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend950_case_inplace_1: Float类型Inplace正弦运算测试
 * @details 验证aclnnInplaceSinGetWorkspaceSize接口Float类型正常工作
 */
TEST_F(sin_test, ascend950_case_inplace_1)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);

    auto selfRef = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceSin, INPUT(selfRef), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend950_case_inplace_2: Float16类型Inplace正弦运算测试
 * @details 验证aclnnInplaceSinGetWorkspaceSize接口Float16类型正常工作
 */
TEST_F(sin_test, ascend950_case_inplace_2)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);

    auto selfRef = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceSin, INPUT(selfRef), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend950_case_inplace_3: Double类型Inplace正弦运算测试
 * @details 验证aclnnInplaceSinGetWorkspaceSize接口Double类型正常工作
 */
TEST_F(sin_test, ascend950_case_inplace_3)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);

    auto selfRef = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceSin, INPUT(selfRef), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend950_case_inplace_4: Complex64类型Inplace正弦运算测试
 * @details 验证aclnnInplaceSinGetWorkspaceSize接口Complex64类型正常工作
 */
TEST_F(sin_test, ascend950_case_inplace_4)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);

    auto selfRef = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceSin, INPUT(selfRef), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend950_case_inplace_5: Complex128类型Inplace正弦运算测试
 * @details 验证aclnnInplaceSinGetWorkspaceSize接口Complex128类型正常工作
 */
TEST_F(sin_test, ascend950_case_inplace_5)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);

    auto selfRef = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceSin, INPUT(selfRef), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend950_case_inplace_6: Inplace空指针测试
 * @details 验证Inplace版本输入为nullptr时返回ACLNN_ERR_PARAM_NULLPTR错误码
 */
TEST_F(sin_test, ascend950_case_inplace_6)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);

    auto ut = OP_API_UT(aclnnInplaceSin, INPUT((aclTensor*)nullptr), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

/**
 * @brief ascend950_case_inplace_7: BF16类型Inplace正弦运算测试
 * @details 验证aclnnInplaceSinGetWorkspaceSize接口BF16类型正常工作
 */
TEST_F(sin_test, ascend950_case_inplace_7)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);

    auto selfRef = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceSin, INPUT(selfRef), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/**
 * @brief ascend950_case_inplace_8: Inplace不支持的dtype测试
 * @details 验证Inplace版本Int8类型输入时返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(sin_test, ascend950_case_inplace_8)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);

    auto selfRef = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceSin, INPUT(selfRef), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/**
 * @brief ascend310P_case_bf16_not_supported: BF16在非支持平台测试
 * @details 验证BF16在Ascend310P平台不支持，返回ACLNN_ERR_PARAM_INVALID错误码
 */
TEST_F(sin_test, ascend310P_case_bf16_not_supported)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2002);

    auto self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSin, INPUT(self), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
