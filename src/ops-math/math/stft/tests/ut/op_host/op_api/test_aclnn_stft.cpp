/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_stft.cpp
 * \brief
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/acl_stft.h"

#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_stft_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "stft_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "stft_test TearDown" << endl;
    }
};

TEST_F(l2_stft_test, ascend910B2_case_aicpu)
{
    auto self_tensor_desc = TensorDesc({16, 25000}, ACL_DOUBLE, ACL_FORMAT_ND);
    int64_t nFft = 400L;
    int64_t hopLength = 160L;
    int64_t winLength = 400L;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    auto out_tensor_desc =
        TensorDesc({16, nFft / 2 + 1, (25000 - winLength) / hopLength + 1, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclStft,
        INPUT(
            self_tensor_desc, (aclTensor*)nullptr, out_tensor_desc, nFft, hopLength, winLength, normalized, onesided,
            returnComplex),
        OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_stft_test, ascend910B2_case_aiccore)
{
    auto self_tensor_desc = TensorDesc({16, 25000}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t nFft = 400L;
    int64_t hopLength = 160L;
    int64_t winLength = 400L;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    auto out_tensor_desc =
        TensorDesc({16, nFft / 2 + 1, (25000 - winLength) / hopLength + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclStft,
        INPUT(
            self_tensor_desc, (aclTensor*)nullptr, out_tensor_desc, nFft, hopLength, winLength, normalized, onesided,
            returnComplex),
        OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_stft_test, ascend910B2_normallize_case_aiccore)
{
    auto self_tensor_desc = TensorDesc({16, 25000}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t nFft = 500L;
    int64_t hopLength = 160L;
    int64_t winLength = 400L;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    auto out_tensor_desc =
        TensorDesc({16, nFft / 2 + 1, (25000 - winLength) / hopLength + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclStft,
        INPUT(
            self_tensor_desc, (aclTensor*)nullptr, out_tensor_desc, nFft, hopLength, winLength, normalized, onesided,
            returnComplex),
        OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_stft_test, ascend910B2_complex_case_aiccore)
{
    auto self_tensor_desc = TensorDesc({16, 25000}, ACL_COMPLEX64, ACL_FORMAT_ND);
    int64_t nFft = 500L;
    int64_t hopLength = 160L;
    int64_t winLength = 400L;
    bool normalized = false;
    bool onesided = false;
    bool returnComplex = false;
    auto out_tensor_desc = TensorDesc({16, nFft, (25000 - winLength) / hopLength + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclStft,
        INPUT(
            self_tensor_desc, (aclTensor*)nullptr, out_tensor_desc, nFft, hopLength, winLength, normalized, onesided,
            returnComplex),
        OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_stft_test, ascend310P_normallize_case_aiccore)
{
    auto self_tensor_desc = TensorDesc({16, 25000}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t nFft = 500L;
    int64_t hopLength = 160L;
    int64_t winLength = 400L;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    auto out_tensor_desc =
        TensorDesc({16, nFft / 2 + 1, (25000 - winLength) / hopLength + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclStft,
        INPUT(
            self_tensor_desc, (aclTensor*)nullptr, out_tensor_desc, nFft, hopLength, winLength, normalized, onesided,
            returnComplex),
        OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_stft_test, ascend910B2_case_self_nullptr)
{
    auto self_tensor_desc = TensorDesc({16, 25000}, ACL_DOUBLE, ACL_FORMAT_ND);
    int64_t nFft = 400L;
    int64_t hopLength = 160L;
    int64_t winLength = 400L;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    auto out_tensor_desc =
        TensorDesc({16, nFft / 2 + 1, (25000 - winLength) / hopLength + 1, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclStft,
        INPUT(
            (aclTensor*)nullptr, (aclTensor*)nullptr, out_tensor_desc, nFft, hopLength, winLength, normalized, onesided,
            returnComplex),
        OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_stft_test, ascend910B2_case_out_nullptr)
{
    auto self_tensor_desc = TensorDesc({16, 25000}, ACL_DOUBLE, ACL_FORMAT_ND);
    int64_t nFft = 400L;
    int64_t hopLength = 160L;
    int64_t winLength = 400L;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    auto out_tensor_desc =
        TensorDesc({16, nFft / 2 + 1, (25000 - winLength) / hopLength + 1, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclStft,
        INPUT(
            self_tensor_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, nFft, hopLength, winLength, normalized,
            onesided, returnComplex),
        OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_stft_test, ascend910B2_case_invalid_dtype)
{
    auto self_tensor_desc = TensorDesc({16, 25000}, ACL_INT32, ACL_FORMAT_ND);
    int64_t nFft = 400L;
    int64_t hopLength = 160L;
    int64_t winLength = 400L;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    auto out_tensor_desc =
        TensorDesc({16, nFft / 2 + 1, (25000 - winLength) / hopLength + 1, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclStft,
        INPUT(
            self_tensor_desc, (aclTensor*)nullptr, out_tensor_desc, nFft, hopLength, winLength, normalized, onesided,
            returnComplex),
        OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_stft_test, ascend910B2_case_invalid_format)
{
    auto self_tensor_desc = TensorDesc({16, 25000}, ACL_DOUBLE, ACL_FORMAT_HWCN);
    int64_t nFft = 400L;
    int64_t hopLength = 160L;
    int64_t winLength = 400L;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    auto out_tensor_desc =
        TensorDesc({16, nFft / 2 + 1, (25000 - winLength) / hopLength + 1, 2}, ACL_DOUBLE, ACL_FORMAT_HWCN);
    auto ut = OP_API_UT(
        aclStft,
        INPUT(
            self_tensor_desc, (aclTensor*)nullptr, out_tensor_desc, nFft, hopLength, winLength, normalized, onesided,
            returnComplex),
        OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_stft_test, ascend910B2_case_invalid_out_less_shape)
{
    auto self_tensor_desc = TensorDesc({16, 25}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t nFft = 400L;
    int64_t hopLength = 160L;
    int64_t winLength = 400L;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    auto out_tensor_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclStft,
        INPUT(
            self_tensor_desc, (aclTensor*)nullptr, out_tensor_desc, nFft, hopLength, winLength, normalized, onesided,
            returnComplex),
        OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_stft_test, ascend910B2_case_invalid_out_more_shape)
{
    auto self_tensor_desc = TensorDesc({16, 25}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t nFft = 400L;
    int64_t hopLength = 160L;
    int64_t winLength = 400L;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    auto out_tensor_desc = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclStft,
        INPUT(
            self_tensor_desc, (aclTensor*)nullptr, out_tensor_desc, nFft, hopLength, winLength, normalized, onesided,
            returnComplex),
        OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_stft_test, ascend910B2_case_plan_cache)
{
    auto self_tensor_desc = TensorDesc({16, 25000}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t nFft = 400L;
    int64_t hopLength = 160L;
    int64_t winLength = 400L;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    auto out_tensor_desc =
        TensorDesc({16, nFft / 2 + 1, (25000 - winLength) / hopLength + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    for (int i = 0; i < 5; i++) {
        auto ut = OP_API_UT(
            aclStft,
            INPUT(
                self_tensor_desc, (aclTensor*)nullptr, out_tensor_desc, nFft, hopLength, winLength, normalized,
                onesided, returnComplex),
            OUTPUT());

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);

        // SAMPLE: precision simulate
        // ut.TestPrecision();
    }
}

TEST_F(l2_stft_test, ascend910B3_case_plan_cache)
{
    auto self_tensor_desc = TensorDesc({16, 25000}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t nFft = 400L;
    int64_t hopLength = 160L;
    int64_t winLength = 400L;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    auto out_tensor_desc =
        TensorDesc({16, nFft / 2 + 1, (25000 - winLength) / hopLength + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    for (int i = 0; i < 5; i++) {
        auto ut = OP_API_UT(
            aclStft,
            INPUT(
                self_tensor_desc, (aclTensor*)nullptr, out_tensor_desc, nFft, hopLength, winLength, normalized,
                onesided, returnComplex),
            OUTPUT());

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);

        // SAMPLE: precision simulate
        // ut.TestPrecision();
    }
}

TEST_F(l2_stft_test, ascend910B2_case_invalid_nfft)
{
    auto self_tensor_desc = TensorDesc({16, 25000}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t nFft = 0;
    int64_t hopLength = 100L;
    int64_t winLength = 400L;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    auto out_tensor_desc = TensorDesc({16, nFft / 2 + 1, (25000 - winLength) / 160L + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclStft,
        INPUT(
            self_tensor_desc, (aclTensor*)nullptr, out_tensor_desc, nFft, hopLength, winLength, normalized, onesided,
            returnComplex),
        OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_stft_test, ascend910B2_case_invalid_hop_length)
{
    auto self_tensor_desc = TensorDesc({16, 25000}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t nFft = 400L;
    int64_t hopLength = 0L;
    int64_t winLength = 400L;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    auto out_tensor_desc = TensorDesc({16, nFft / 2 + 1, (25000 - winLength) / 160L + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclStft,
        INPUT(
            self_tensor_desc, (aclTensor*)nullptr, out_tensor_desc, nFft, hopLength, winLength, normalized, onesided,
            returnComplex),
        OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_stft_test, ascend910B2_case_invalid_win_length)
{
    auto self_tensor_desc = TensorDesc({16, 25000}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t nFft = 400L;
    int64_t hopLength = 160L;
    int64_t winLength = 0L;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    auto out_tensor_desc = TensorDesc({16, nFft / 2 + 1, (25000 - 400L) / hopLength + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclStft,
        INPUT(
            self_tensor_desc, (aclTensor*)nullptr, out_tensor_desc, nFft, hopLength, winLength, normalized, onesided,
            returnComplex),
        OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_stft_test, ascend910B2_case_invalid_win_length_with_window)
{
    auto self_tensor_desc = TensorDesc({16, 25000}, ACL_FLOAT, ACL_FORMAT_ND);
    auto win_tensor_desc = TensorDesc({100}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t nFft = 400L;
    int64_t hopLength = 160L;
    int64_t winLength = 300L;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    auto out_tensor_desc =
        TensorDesc({16, nFft / 2 + 1, (25000 - winLength) / hopLength + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclStft,
        INPUT(
            self_tensor_desc, win_tensor_desc, out_tensor_desc, nFft, hopLength, winLength, normalized, onesided,
            returnComplex),
        OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_stft_test, ascend910B2_case_invalid_input_length)
{
    auto self_tensor_desc = TensorDesc({16, 25000}, ACL_FLOAT, ACL_FORMAT_ND);
    auto win_tensor_desc = TensorDesc({100}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t nFft = 40000L;
    int64_t hopLength = 160L;
    int64_t winLength = 300L;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    auto out_tensor_desc =
        TensorDesc({16, nFft / 2 + 1, (25000 - winLength) / hopLength + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclStft,
        INPUT(
            self_tensor_desc, win_tensor_desc, out_tensor_desc, nFft, hopLength, winLength, normalized, onesided,
            returnComplex),
        OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_stft_test, ascend910B2_case_invalid_onesided)
{
    auto self_tensor_desc = TensorDesc({16, 25000}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto win_tensor_desc = TensorDesc({100}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t nFft = 20000L;
    int64_t hopLength = 160L;
    int64_t winLength = 300L;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    auto out_tensor_desc =
        TensorDesc({16, nFft / 2 + 1, (25000 - winLength) / hopLength + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclStft,
        INPUT(
            self_tensor_desc, win_tensor_desc, out_tensor_desc, nFft, hopLength, winLength, normalized, onesided,
            returnComplex),
        OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_stft_test, ascend910B2_case_with_window)
{
    auto self_tensor_desc = TensorDesc({16, 25000}, ACL_FLOAT, ACL_FORMAT_ND);
    auto win_tensor_desc = TensorDesc({400}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t nFft = 400L;
    int64_t hopLength = 160L;
    int64_t winLength = 400L;
    bool normalized = false;
    bool onesided = true;
    bool returnComplex = false;
    auto out_tensor_desc =
        TensorDesc({16, nFft / 2 + 1, (25000 - winLength) / hopLength + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclStft,
        INPUT(
            self_tensor_desc, win_tensor_desc, out_tensor_desc, nFft, hopLength, winLength, normalized, onesided,
            returnComplex),
        OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}