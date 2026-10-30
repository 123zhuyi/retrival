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
#include "math/reduce_std_v2/op_api/aclnn_std.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_std_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_std_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_std_test TearDown" << std::endl;
    }

public:
    template <typename T>
    int CreateAclScalarTensor(
        const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
        aclTensor** tensor)
    {
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(
            0, 0, dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
        return 0;
    }
};

// 正常场景_float32_nd_dim为0_keepdim为true
TEST_F(l2_std_test, normal_dtype_float32_format_nd_dim_0_keepdim_true)
{
    auto selfDesc = TensorDesc({2, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 1;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 正常场景_float32_nchw_dim为1_keepdim为false
TEST_F(l2_std_test, normal_dtype_float32_format_nchw_dim_1_keepdim_false)
{
    auto selfDesc = TensorDesc({1, 2, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{1});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 正常场景_float32_nhwc_dim为2_keepdim为false
TEST_F(l2_std_test, normal_dtype_float32_format_nhwc_dim_2_keepdim_false)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 正常场景_float16_hwcn_dim为2_keepdim为true
TEST_F(l2_std_test, normal_dtype_float16_format_hwcn_dim_2_keepdim_true)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT16, ACL_FORMAT_HWCN).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2});
    int64_t correction = 1;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 2, 1, 4}, ACL_FLOAT16, ACL_FORMAT_HWCN).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 正常场景_bfloat16_hwcn_dim为2_keepdim为true
TEST_F(l2_std_test, ascend910B2_normal_dtype_bfloat16_format_hwcn_dim_2_keepdim_true)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_BF16, ACL_FORMAT_HWCN).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2});
    int64_t correction = 1;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 2, 1, 4}, ACL_BF16, ACL_FORMAT_HWCN).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 正常场景_float32_ncdhw_dim为2_3_keepdim为true
TEST_F(l2_std_test, normal_dtype_float32_format_ncdhw_dim_2_3_keepdim_true)
{
    auto selfDesc = TensorDesc({1, 2, 8, 6, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2, 3});
    int64_t correction = 1;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 2, 1, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 正常场景_float16_ndhwc_dim为1_4_keepdim为true
TEST_F(l2_std_test, normal_dtype_float16_format_ndhwc_dim_1_4_keepdim_true)
{
    auto selfDesc = TensorDesc({1, 2, 6, 2, 4}, ACL_FLOAT16, ACL_FORMAT_NDHWC).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{1, 4});
    int64_t correction = 1;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 1, 6, 2, 1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 正常场景_correction为0
TEST_F(l2_std_test, normal_correction_0)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{1, 2});
    int64_t correction = 0;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 1, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 正常场景_correction为2
TEST_F(l2_std_test, normal_correction_2)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{3});
    int64_t correction = 2;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 2, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 所有维度都做reduce_keepdim为false
TEST_F(l2_std_test, normal_all_reduce_keepdim_false)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0, 1, 2, 3});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 所有维度reduce且输入为空dim_keepdim为false
TEST_F(l2_std_test, normal_all_reduce_empty_dim_keepdim_false)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 0维tensor场景
TEST_F(l2_std_test, normal_shape_dim_0)
{
    auto selfDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 1;
    bool keepdim = true;
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 8维tensor场景
TEST_F(l2_std_test, normal_shape_dim_8)
{
    auto selfDesc = TensorDesc({2, 1, 9, 9, 4, 3, 9, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0, 3, 4});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({1, 9, 3, 9, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// dim为空指针
TEST_F(l2_std_test, normal_dim_nullptr)
{
    auto selfDesc = TensorDesc({2, 1, 9, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2, 3});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, nullptr, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dim为负且对应shape乘积为1_shape_prod等于correction
TEST_F(l2_std_test, normal_neg_dim)
{
    auto selfDesc = TensorDesc({6, 30, 17, 32, 32, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{-1});
    int64_t correction = 1;
    bool keepdim = true;
    auto outDesc = TensorDesc({6, 30, 17, 32, 32, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// 空tensor场景
TEST_F(l2_std_test, normal_empty_tensor)
{
    auto selfDesc = TensorDesc({2, 0, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2});
    int64_t correction = 1;
    bool keepdim = true;
    auto outDesc = TensorDesc({2, 0, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// dim对应shape乘积为1_shape_prod等于correction
TEST_F(l2_std_test, normal_shape_prod_equal_1_correction_equal)
{
    auto selfDesc = TensorDesc({1, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 1;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// dim对应shape乘积为1_shape_prod小于correction
TEST_F(l2_std_test, normal_shape_prod_equal_1_correction_above)
{
    auto selfDesc = TensorDesc({1, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 2;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// correction大于1_shape_prod等于correction
TEST_F(l2_std_test, normal_correction_greater_than_1_correction_equal)
{
    auto selfDesc = TensorDesc({1, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0, 1});
    int64_t correction = 2;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// correction大于1_shape_prod小于correction
TEST_F(l2_std_test, normal_correction_greater_than_1_correction_above)
{
    auto selfDesc = TensorDesc({2, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0, 1});
    int64_t correction = 3;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, 0);

    // ut.TestPrecision();
}

// CheckNotNull_1
TEST_F(l2_std_test, abnormal_self_nullptr)
{
    auto selfDesc = TensorDesc({2, 1, 9, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2, 3});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnStd, INPUT(nullptr, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_2
TEST_F(l2_std_test, abnormal_out_nullptr)
{
    auto selfDesc = TensorDesc({2, 1, 9, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2, 3});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid_1
TEST_F(l2_std_test, abnormal_dtype_int32)
{
    auto selfDesc = TensorDesc({2, 1, 9, 9}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2, 3});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({2, 1}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_2
TEST_F(l2_std_test, abnormal_dtype_uint8)
{
    auto selfDesc = TensorDesc({2, 1, 9, 9}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2, 3});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({2, 1}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDimValid_1
TEST_F(l2_std_test, abnormal_dim_out_of_range)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{7});
    int64_t correction = 1;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDimValid_2
TEST_F(l2_std_test, abnormal_dim_repeated)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2, 2});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDimValid_3
TEST_F(l2_std_test, abnormal_dim_neg_out_of_range)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{-5});
    int64_t correction = 1;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDimValid_4
TEST_F(l2_std_test, abnormal_neg_dim_repeated)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{-1, -1});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_1
TEST_F(l2_std_test, abnormal_shape_dim_greater_than_threshold)
{
    auto selfDesc = TensorDesc({2, 1, 9, 9, 4, 3, 9, 9, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2, 3, 5, 7});
    int64_t correction = 1;
    bool keepdim = true;
    auto outDesc = TensorDesc({2, 1, 1, 1, 4, 1, 9, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_std_test, l2_std_self_scalar_tensor)
{
    std::vector<float> selfHostData = {0.0};
    std::vector<int64_t> selfShape = {1};
    void* selfDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    auto ret = CreateAclScalarTensor<float>(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 0;
    bool keepdim = false;
    auto outDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(self, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_std_test, l2_std_self_scalar_tensor_keepdim)
{
    std::vector<float> selfHostData = {0.0};
    std::vector<int64_t> selfShape = {1};
    void* selfDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    auto ret = CreateAclScalarTensor<float>(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 0;
    bool keepdim = true;
    auto outDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(self, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 950
TEST_F(l2_std_test, ascend950_normal_dtype_float16_format_hwcn_dim_2_keepdim_true)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT16, ACL_FORMAT_HWCN).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{2});
    int64_t correction = 1;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 2, 1, 4}, ACL_FLOAT16, ACL_FORMAT_HWCN).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckShape_2: out shape 不匹配
TEST_F(l2_std_test, abnormal_shape_not_match)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{1});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_3: keepdim为true时out shape不匹配
TEST_F(l2_std_test, abnormal_shape_not_match_keepdim_true)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{1});
    int64_t correction = 1;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dim为负数场景
TEST_F(l2_std_test, normal_neg_dim_valid)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{-2});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({1, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 负dim转换测试
TEST_F(l2_std_test, normal_neg_dim_convert)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{-1, -2, -3});
    int64_t correction = 1;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 2, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}


// correction > 1 且 shapeProd < correction 返回INF
TEST_F(l2_std_test, normal_correction_above_shape_prod)
{
    auto selfDesc = TensorDesc({1, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0, 1});
    int64_t correction = 3;
    bool keepdim = true;
    auto outDesc = TensorDesc({1, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 混合正负dim
TEST_F(l2_std_test, normal_mixed_pos_neg_dim)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{1, -2});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 5维tensor场景
TEST_F(l2_std_test, normal_shape_dim_5)
{
    auto selfDesc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0, 2, 4});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试最后一个维度
TEST_F(l2_std_test, normal_last_dim)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{3});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({1, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试第一个维度
TEST_F(l2_std_test, normal_first_dim)
{
    auto selfDesc = TensorDesc({1, 2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({2, 6, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dtype float32
TEST_F(l2_std_test, normal_dtype_float32)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{1});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dtype float16
TEST_F(l2_std_test, normal_dtype_float16)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto dimDesc = IntArrayDesc(vector<int64_t>{1});
    int64_t correction = 1;
    bool keepdim = false;
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnStd, INPUT(selfDesc, dimDesc, correction, keepdim), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}