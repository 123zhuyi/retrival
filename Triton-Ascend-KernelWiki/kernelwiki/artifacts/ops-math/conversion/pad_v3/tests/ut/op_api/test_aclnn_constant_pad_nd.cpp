/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_constant_pad_nd.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2ConstantPadNd : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2ConstantPadNd SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2ConstantPadNd TearDown" << std::endl; }
};

TEST_F(l2ConstantPadNd, case_norm_int64)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({6, 6}, ACL_INT64, ACL_FORMAT_ND);
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2ConstantPadNd, case_norm_int32)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{0, 0, 0, 0});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2ConstantPadNd, case_norm_int32_new)
{
    auto tensor_desc = TensorDesc({22948, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{0, 1, 0, 0});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({22948, 4}, ACL_INT32, ACL_FORMAT_ND);
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2ConstantPadNd, case_norm_int16)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{1, 0, 0, -1});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_INT16, ACL_FORMAT_ND);
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2ConstantPadNd, case_norm_int8)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{1, -1, 1, -1});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_INT8, ACL_FORMAT_ND);
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2ConstantPadNd, case_norm_uint8)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{1, -1, 1, -1});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_UINT8, ACL_FORMAT_ND);
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2ConstantPadNd, case_norm_bool)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{1, -1, 1, -1});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2ConstantPadNd, case_norm_float32_aicpu)
{
    auto tensor_desc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({1, 1, 1, 1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2ConstantPadNd, case_pad_odd)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({6, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2ConstantPadNd, case_pad_not_equal_dim)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{0, 0, 1, 1, 1, 1});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({6, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2ConstantPadNd, case_pad_abnormal_1)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{1, -10, 1, 1});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({6, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2ConstantPadNd, case_pad_abnormal_3)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{-2, -2, 1, 1});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({6, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2ConstantPadNd, case_pad_normal_minus)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{-2, -2});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({4, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2ConstantPadNd, case_out_abnormal)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2ConstantPadNd, case_format_abnormal)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({6, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2ConstantPadNd, case_out_shape_abnormal)
{
    auto tensor_desc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1, 1, 1, 1});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({3, 3, 3, 3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2ConstantPadNd, case_nullptr_input)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({6, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT((aclTensor*)nullptr, pad, (aclScalar*)nullptr),
                        OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2ConstantPadNd, case_nullptr_output)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({6, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT((aclTensor*)nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2ConstantPadNd, case_zero_value_exists_within_padcover_range)
{
    auto tensor_desc = TensorDesc({1, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{-2, 4, 0, 0});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({1, 0, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2ConstantPadNd, case_zero_value_not_exists_within_padcover_range)
{
    auto tensor_desc = TensorDesc({1, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{-2, 4});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({1, 0, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2ConstantPadNd, case_norm_empty_neg)
{
    auto tensor_desc = TensorDesc({4, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{0, 0, -1, -1});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2ConstantPadNd, case_norm_empty_pos)
{
    auto tensor_desc = TensorDesc({4, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({6, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2ConstantPadNd, case_norm_empty_zeros)
{
    auto tensor_desc = TensorDesc({4, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{0, 0, 0, 0});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({4, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2ConstantPadNd, case_norm_empty_bool)
{
    auto tensor_desc = TensorDesc({0}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{0, 1});
    auto value = ScalarDesc(static_cast<int64_t>(-2147483648));
    auto out_tensor_desc = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2ConstantPadNd, case_abnormal_empty_pos)
{
    auto tensor_desc = TensorDesc({4, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{0, 0, 1, 1});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({6, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2ConstantPadNd, case_norm_strided_zeros)
{
    auto tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto pad = IntArrayDesc(vector<int64_t>{0, 0});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2ConstantPadNd, ascend950_case_norm_float32)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto pad = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto value = ScalarDesc(static_cast<int64_t>(2));
    auto out_tensor_desc = TensorDesc({6, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnConstantPadNd, INPUT(tensor_desc, pad, value), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
