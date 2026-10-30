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
#include "../../../op_api/aclnn_zero.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2ZeroTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2ZeroTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2ZeroTest TearDown" << std::endl;
    }
};

// self的数据类型不在支持范围内
TEST_F(l2ZeroTest, l2_zero_test_unsupported_type)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_UINT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 空tensor
TEST_F(l2ZeroTest, l2_zero_test_null)
{
    auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径, aicore, int8
TEST_F(l2ZeroTest, l2_zero_test_int8)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径, aicore, int32
TEST_F(l2ZeroTest, l2_zero_test_int32)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径, aicore, int64
TEST_F(l2ZeroTest, l2_zero_test_int64)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径, aicore, uint8
TEST_F(l2ZeroTest, l2_zero_test_uint8)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径， aicore, float16
TEST_F(l2ZeroTest, l2_zero_test_float16)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径， aicore, float32
TEST_F(l2ZeroTest, l2_zero_test_float32)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径， aicore, bool
TEST_F(l2ZeroTest, l2_zero_test_bool)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径, aicpu, double
TEST_F(l2ZeroTest, l2_zero_test_double)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径, aicpu, INT16
TEST_F(l2ZeroTest, l2_zero_test_int16)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径, aicpu, UINT16
TEST_F(l2ZeroTest, l2_zero_test_uint16)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_UINT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径, aicpu, COMPLEX64
TEST_F(l2ZeroTest, l2_zero_test_complex64)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_COMPLEX64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径, aicpu, COMPLEX128
TEST_F(l2ZeroTest, l2_zero_test_complex128)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_COMPLEX128, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径, 测试 ACL_BF16 类型（仅在910B上支持aicore，其他型号走aicpu）
TEST_F(l2ZeroTest, l2_zero_test_bf16)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径, 触发 CheckDim 函数（维度为8时）
TEST_F(l2ZeroTest, l2_zero_test_max_dim)
{
    auto selfDesc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 测试多维度 aicpu 类型
TEST_F(l2ZeroTest, l2_zero_test_multi_dim_double)
{
    auto selfDesc = TensorDesc({1, 2, 3, 4}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// ============================================================================
// L2 异常 / 参数校验补充用例。
// ============================================================================

// 不支持的 dtype（UINT64 不在 L2 DTYPE_SUPPORT_LIST 13 种内）→ CheckDtypeValid 失败
TEST_F(l2ZeroTest, l2_zero_test_unsupported_type_uint64)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 空指针 selfRef → CheckNotNull / OP_CHECK_NULL → ACLNN_ERR_PARAM_NULLPTR(161001)。
// 单输入算子需传带类型的空指针 (aclTensor*)nullptr，否则 InferAclTypes 无法对 nullptr_t 进行类型推导。
TEST_F(l2ZeroTest, l2_zero_test_nullptr_self)
{
    auto ut = OP_API_UT(aclnnInplaceZero, INPUT((aclTensor*)nullptr), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// rank 越界（9 维 > MAX_DIM_LEN=8）—— AiCpu 兜底 dtype（DOUBLE）路径。
// L0 ZerosLikeAiCpu 调用 CheckDim()(OP_CHECK_MAX_DIM) 失败 → 返回 nullptr →
// L2 CHECK_RET(zeroOut != nullptr, ACLNN_ERR_INNER_NULLPTR) → 实测返回 561103。
TEST_F(l2ZeroTest, l2_zero_test_rank9_aicpu_double)
{
    auto selfDesc = TensorDesc({2, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_INNER_NULLPTR);
}

// rank 越界（9 维）—— 910b AiCore dtype（FP32）路径。
// L0 ZerosLikeAiCore 不做 CheckDim（rank 校验仅在 AiCpu 路径），9D 返回 ACLNN_SUCCESS。
TEST_F(l2ZeroTest, l2_zero_test_rank9_aicore_fp32_no_check)
{
    auto selfDesc = TensorDesc({2, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 910b AI Core 8 种 dtype 基础成功路径汇总复核（防 dtype 列表回归）。
TEST_F(l2ZeroTest, l2_zero_test_aicore_8dtype_all_success)
{
    const std::vector<aclDataType> kAiCore910bDtypes = {ACL_INT8,    ACL_INT32, ACL_INT64, ACL_UINT8,
                                                        ACL_FLOAT16, ACL_FLOAT, ACL_BOOL,  ACL_BF16};
    for (auto dt : kAiCore910bDtypes) {
        auto selfDesc = TensorDesc({2, 4}, dt, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnInplaceZero, INPUT(selfDesc), OUTPUT());
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS) << "dtype enum=" << static_cast<int>(dt);
    }
}
