/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "reduce_std_with_mean/op_host/op_api/aclnn_std_mean_correction.h"

#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_std_mean_correction_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_std_mean_correction_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_std_mean_correction_test TearDown" << endl;
    }
};

TEST_F(l2_std_mean_correction_test, std_mean_correction_dtype_float)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto dim = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 1;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    ut.TestPrecision();
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_dtype_float16)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto dim = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 1;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({1, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({1, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    ut.TestPrecision();
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_dtype_int8)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto dim = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 1;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_dtype_int32)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto dim = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 1;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_dtype_uint8)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto dim = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 1;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_all_format)
{
    vector<aclFormat> format_list{ACL_FORMAT_NC1HWC0, ACL_FORMAT_NCHW,  ACL_FORMAT_NHWC, ACL_FORMAT_ND,
                                  ACL_FORMAT_HWCN,    ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};
    for (auto format : format_list) {
        cout << "+++++++++++++++++++++++ start to test format " << format << endl;
        auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, format).Value(vector<float>{1, 2, 3, 4, 5, 6});
        auto dim = IntArrayDesc(vector<int64_t>{0});
        int64_t correction = 1;
        bool keepdim = true;
        auto stdOutDesc = TensorDesc({1, 3}, ACL_FLOAT, format);
        auto meanOutDesc = TensorDesc({1, 3}, ACL_FLOAT, format);

        auto ut = OP_API_UT(
            aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        if (format == ACL_FORMAT_NC1HWC0) {
            EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
        } else {
            EXPECT_EQ(aclRet, ACLNN_SUCCESS);
            ut.TestPrecision();
        }
    }
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_nullptr_self)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto dim = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 1;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(nullptr, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_nullptr_stdout)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto dim = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 1;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(nullptr, meanOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_nullptr_meanout)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto dim = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 1;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_empty_tensor)
{
    auto selfDesc = TensorDesc({0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 1;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    selfDesc = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut2 =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_non_contiguous)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {3, 2}).ValueRange(-1, 1);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 1;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_invalid_dim)
{
    auto selfDesc = TensorDesc({2, 3, 3, 2, 2, 3, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 1;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    selfDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut2 =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_dim_neg)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto dim = IntArrayDesc(vector<int64_t>{-1});
    int64_t correction = 1;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_dim_multi)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 24);
    auto dim = IntArrayDesc(vector<int64_t>{0, 1});
    int64_t correction = 1;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({1, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_dim_empty)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto dim = IntArrayDesc(vector<int64_t>{});
    int64_t correction = 1;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_dim_nullptr)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    int64_t correction = 1;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnStdMeanCorrection, INPUT(selfDesc, nullptr, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_keepdim_false)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto dim = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 1;
    bool keepdim = false;
    auto stdOutDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_correction_0)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto dim = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 0;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_correction_2)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto dim = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 2;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_shape_prod_1_correction_1)
{
    auto selfDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto dim = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 1;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_shape_prod_lt_correction)
{
    auto selfDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto dim = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 2;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_dim_repeat)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto dim = IntArrayDesc(vector<int64_t>{0, 0});
    int64_t correction = 1;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_dim_out_of_range)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto dim = IntArrayDesc(vector<int64_t>{5});
    int64_t correction = 1;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_output_shape_mismatch)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto dim = IntArrayDesc(vector<int64_t>{0});
    int64_t correction = 1;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_std_mean_correction_test, std_mean_correction_5d_tensor)
{
    auto selfDesc = TensorDesc({2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{2, 3});
    int64_t correction = 1;
    bool keepdim = true;
    auto stdOutDesc = TensorDesc({2, 3, 1, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanOutDesc = TensorDesc({2, 3, 1, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnStdMeanCorrection, INPUT(selfDesc, dim, correction, keepdim), OUTPUT(stdOutDesc, meanOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
