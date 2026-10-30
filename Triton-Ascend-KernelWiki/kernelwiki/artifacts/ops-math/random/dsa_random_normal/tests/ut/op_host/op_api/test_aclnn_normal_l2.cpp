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

#include "../../../../op_host/op_api/aclnn_normal.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_normal_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "normal_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "normal_test TearDown" << endl;
    }
};

// 0维场景
TEST_F(l2_normal_test, case_0_dim_float_NCHW)
{
    auto tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(tensor_desc);
    float mean = 1.;
    float std = 2.;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 4维场景
TEST_F(l2_normal_test, case_4_dim_float_NCHW)
{
    auto tensor_desc = TensorDesc({3, 4, 4, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(tensor_desc);
    float mean = 1.;
    float std = 2.;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 8维场景
TEST_F(l2_normal_test, case_8_dim_float_NCHW)
{
    auto tensor_desc = TensorDesc({3, 4, 2, 3, 2, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    float mean = 2.5;
    float std = 5.6;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// data range(-1, 1) - ND
TEST_F(l2_normal_test, case_1_1_float_NCHW)
{
    auto tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    float mean = 2.;
    float std = 1.;
    int64_t seed = 1;
    int64_t offset = 3;

    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 3 - float - ND
TEST_F(l2_normal_test, case_3_4_5_float_ND)
{
    auto tensor_desc = TensorDesc({3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(tensor_desc);
    float mean = 1.;
    float std = 2.;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 3 - float16 - ND
TEST_F(l2_normal_test, case_3_4_5_float16_ND)
{
    auto tensor_desc = TensorDesc({3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(tensor_desc);
    float mean = 1.;
    float std = 2.;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 3 - float - NCHW
TEST_F(l2_normal_test, case_3_4_5_float_NCHW)
{
    auto tensor_desc = TensorDesc({3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc(tensor_desc);
    float mean = 1.;
    float std = 2.;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 3 - float - NHWC
TEST_F(l2_normal_test, case_3_4_5_float_NHWC)
{
    auto tensor_desc = TensorDesc({3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto out_tensor_desc = TensorDesc(tensor_desc);
    float mean = 1.;
    float std = 2.;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 3 - float - NWCN
TEST_F(l2_normal_test, case_3_4_5_float_HWCN)
{
    auto tensor_desc = TensorDesc({3, 4, 5}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto out_tensor_desc = TensorDesc(tensor_desc);
    float mean = 1.;
    float std = 2.;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 3 - float - NDHWC
TEST_F(l2_normal_test, case_3_4_5_float_NDHWC)
{
    auto tensor_desc = TensorDesc({3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto out_tensor_desc = TensorDesc(tensor_desc);
    float mean = 1.;
    float std = 2.;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 3 - float - NCDHW
TEST_F(l2_normal_test, case_3_4_5_float_NCDHW)
{
    auto tensor_desc = TensorDesc({3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto out_tensor_desc = TensorDesc(tensor_desc);
    float mean = 1.;
    float std = 2.;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 3 - float - std = 0 - ND
TEST_F(l2_normal_test, case_3_4_5_float_std_equal_zeros_ND)
{
    auto tensor_desc = TensorDesc({3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(tensor_desc);
    float mean = 0;
    float std = 0;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 3 - float - std < 0 - ND
TEST_F(l2_normal_test, case_3_4_5_float_std_less_zeros_ND)
{
    auto tensor_desc = TensorDesc({3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(tensor_desc);
    float mean = 0;
    float std = -2;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

//-----------------------------------------------------------------
// 3 - int16 - ND
TEST_F(l2_normal_test, case_3_4_5_int16_ND)
{
    auto tensor_desc = TensorDesc({3, 4, 5}, ACL_INT16, ACL_FORMAT_NCDHW);
    auto out_tensor_desc = TensorDesc(tensor_desc);
    float mean = 1.;
    float std = 2.;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 3 - int32 - ND
TEST_F(l2_normal_test, case_3_4_5_int32_ND)
{
    auto tensor_desc = TensorDesc({3, 4, 5}, ACL_INT32, ACL_FORMAT_NCDHW);
    auto out_tensor_desc = TensorDesc(tensor_desc);
    float mean = 1.;
    float std = 2.;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 3 - int64 - ND
TEST_F(l2_normal_test, case_3_4_5_int64_ND)
{
    auto tensor_desc = TensorDesc({3, 4, 5}, ACL_INT64, ACL_FORMAT_NCDHW);
    auto out_tensor_desc = TensorDesc(tensor_desc);
    float mean = 1.;
    float std = 2.;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 3 - uint8 - ND
TEST_F(l2_normal_test, case_3_4_5_uint86_ND)
{
    auto tensor_desc = TensorDesc({3, 4, 5}, ACL_UINT8, ACL_FORMAT_NCDHW);
    auto out_tensor_desc = TensorDesc(tensor_desc);
    float mean = 1.;
    float std = 2.;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 3 - bool - ND
TEST_F(l2_normal_test, case_3_4_5_bool_ND)
{
    auto tensor_desc = TensorDesc({3, 4, 5}, ACL_BOOL, ACL_FORMAT_NCDHW);
    auto out_tensor_desc = TensorDesc(tensor_desc);
    float mean = 1.;
    float std = 2.;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// ======================== aclnnInplaceNormal 新增用例 ========================

// double类型，走StatelessRandomNormalV2 fallback路径
TEST_F(l2_normal_test, case_3_4_double_ND)
{
    auto tensor_desc = TensorDesc({3, 4}, ACL_DOUBLE, ACL_FORMAT_ND);
    float mean = 1.;
    float std = 2.;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// bf16类型
TEST_F(l2_normal_test, case_3_4_bf16_ND)
{
    auto tensor_desc = TensorDesc({3, 4}, ACL_BF16, ACL_FORMAT_ND);
    float mean = 0.;
    float std = 1.;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 异常场景：超过8维
TEST_F(l2_normal_test, case_9_dim_float_ND_invalid)
{
    auto tensor_desc = TensorDesc({3, 4, 2, 3, 2, 3, 3, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    float mean = 1.;
    float std = 2.;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT(tensor_desc, mean, std, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：self为nullptr
TEST_F(l2_normal_test, case_self_nullptr)
{
    float mean = 1.;
    float std = 2.;
    int64_t seed = 1;
    int64_t offset = 1;
    auto ut = OP_API_UT(aclnnInplaceNormal, INPUT((aclTensor*)nullptr, mean, std, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ======================== aclnnInplaceNormalTensor 新增用例 ========================

// 正常场景：float类型
TEST_F(l2_normal_test, case_normalTensor_3_4_float_ND)
{
    float mean = 1.;
    float std = 2.;
    int64_t offset = 0;

    auto tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seedTensorDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetTensorDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceNormalTensor,
        INPUT(tensor_desc, mean, std, seedTensorDesc, offsetTensorDesc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：float16类型
TEST_F(l2_normal_test, case_normalTensor_3_4_5_float16_ND)
{
    float mean = 0.;
    float std = 1.;
    int64_t offset = 0;

    auto tensor_desc = TensorDesc({3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto seedTensorDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetTensorDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceNormalTensor,
        INPUT(tensor_desc, mean, std, seedTensorDesc, offsetTensorDesc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：bf16类型
TEST_F(l2_normal_test, case_normalTensor_3_4_bf16_ND)
{
    float mean = 0.;
    float std = 1.;
    int64_t offset = 0;

    auto tensor_desc = TensorDesc({3, 4}, ACL_BF16, ACL_FORMAT_ND);
    auto seedTensorDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetTensorDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceNormalTensor,
        INPUT(tensor_desc, mean, std, seedTensorDesc, offsetTensorDesc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：double类型，走StatelessRandomNormalV2 fallback路径
TEST_F(l2_normal_test, case_normalTensor_3_4_double_ND)
{
    float mean = 1.;
    float std = 2.;
    int64_t offset = 0;

    auto tensor_desc = TensorDesc({3, 4}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto seedTensorDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetTensorDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceNormalTensor,
        INPUT(tensor_desc, mean, std, seedTensorDesc, offsetTensorDesc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：int32类型（非浮点输出，会Cast到FLOAT再计算）
TEST_F(l2_normal_test, case_normalTensor_3_4_int32_ND)
{
    float mean = 1.;
    float std = 2.;
    int64_t offset = 0;

    auto tensor_desc = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto seedTensorDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetTensorDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceNormalTensor,
        INPUT(tensor_desc, mean, std, seedTensorDesc, offsetTensorDesc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 异常场景：std < 0
TEST_F(l2_normal_test, case_normalTensor_std_less_zeros)
{
    float mean = 0.;
    float std = -2.;
    int64_t offset = 0;

    auto tensor_desc = TensorDesc({3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seedTensorDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetTensorDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceNormalTensor,
        INPUT(tensor_desc, mean, std, seedTensorDesc, offsetTensorDesc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景：self为nullptr
TEST_F(l2_normal_test, case_normalTensor_self_nullptr)
{
    float mean = 1.;
    float std = 2.;
    int64_t offset = 0;

    auto seedTensorDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetTensorDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceNormalTensor,
        INPUT((aclTensor*)nullptr, mean, std, seedTensorDesc, offsetTensorDesc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景：超过8维
TEST_F(l2_normal_test, case_normalTensor_9_dim_invalid)
{
    float mean = 1.;
    float std = 2.;
    int64_t offset = 0;

    auto tensor_desc = TensorDesc({3, 4, 2, 3, 2, 3, 3, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seedTensorDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetTensorDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceNormalTensor,
        INPUT(tensor_desc, mean, std, seedTensorDesc, offsetTensorDesc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常场景：带非零offset
TEST_F(l2_normal_test, case_normalTensor_3_4_float_with_offset)
{
    float mean = 0.;
    float std = 1.;
    int64_t offset = 100;

    auto tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seedTensorDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetTensorDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceNormalTensor,
        INPUT(tensor_desc, mean, std, seedTensorDesc, offsetTensorDesc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：std = 0，均值固定
TEST_F(l2_normal_test, case_normalTensor_std_equal_zero)
{
    float mean = 5.;
    float std = 0.;
    int64_t offset = 0;

    auto tensor_desc = TensorDesc({3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seedTensorDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetTensorDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceNormalTensor,
        INPUT(tensor_desc, mean, std, seedTensorDesc, offsetTensorDesc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}