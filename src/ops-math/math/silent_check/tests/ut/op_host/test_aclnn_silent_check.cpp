/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_silent_check.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op; 
using namespace std;

class l2_silent_check_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "SilentCheck Test Setup" << endl; }
    static void TearDownTestCase() { cout << "SilentCheck Test TearDown" << endl; }
};

TEST_F(l2_silent_check_test, case_check_shape_failed)
{
    const int32_t c_min_steps = 2;
    const float c_thresh_l1 = 300;
    const float c_coeff_l1 = 1;
    const float c_thresh_l2 = 0.8;
    const float c_coeff_l2 = 0.5;
    const int32_t npu_asd_detect = 1;
    auto val_tensor_desc = TensorDesc({1, 2}, ACL_FLOAT).Value(vector<float>{1, 2});
    auto input_grad_tensor_desc = TensorDesc({1, 2}, ACL_FLOAT).Value(vector<float>{1, 2});
    auto sfda_tensor_desc = TensorDesc({1, 2}, ACL_FLOAT).Value(vector<float>{1, 2});
    auto step_tensor_desc = TensorDesc({1, 2}, ACL_INT64).Value(vector<int>{1, 2});
    auto result_tensor_desc = TensorDesc({1}, ACL_INT32);
    auto ut = OP_API_UT(aclnnSilentCheck,
                        INPUT(val_tensor_desc, input_grad_tensor_desc, sfda_tensor_desc, step_tensor_desc, c_min_steps, c_thresh_l1, c_coeff_l1, c_thresh_l2, c_coeff_l2, npu_asd_detect),
                        OUTPUT(result_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_silent_check_test, case_check_not_null_failed) {
    const int32_t c_min_steps = 2;
    const float c_thresh_l1 = 300;
    const float c_coeff_l1 = 1;
    const float c_thresh_l2 = 0.8;
    const float c_coeff_l2 = 0.5;
    const int32_t npu_asd_detect = 1;
    auto result_tensor_desc = TensorDesc({1}, ACL_INT32);
    auto ut = OP_API_UT(aclnnSilentCheck,
                        INPUT(nullptr, nullptr, nullptr, nullptr, c_min_steps, c_thresh_l1, c_coeff_l1, c_thresh_l2, c_coeff_l2, npu_asd_detect),
                        OUTPUT(result_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_silent_check_test, case_check_dtype_failed) {
    const int32_t c_min_steps = 2;
    const float c_thresh_l1 = 300;
    const float c_coeff_l1 = 1;
    const float c_thresh_l2 = 0.8;
    const float c_coeff_l2 = 0.5;
    const int32_t npu_asd_detect = 1;
    auto val_tensor_desc = TensorDesc({1}, ACL_DOUBLE).Value(vector<double>{160});
    auto input_grad_tensor_desc = TensorDesc({1, 2}, ACL_FLOAT).Value(vector<float>{1, 2});
    auto sfda_tensor_desc = TensorDesc({3}, ACL_DOUBLE).Value(vector<double>{70, 70, 70});
    auto step_tensor_desc = TensorDesc({1}, ACL_INT64).Value(vector<int>{7});
    auto result_tensor_desc = TensorDesc({1}, ACL_INT32);
    auto ut = OP_API_UT(aclnnSilentCheck,
                        INPUT(val_tensor_desc, input_grad_tensor_desc, sfda_tensor_desc, step_tensor_desc, c_min_steps, c_thresh_l1, c_coeff_l1, c_thresh_l2, c_coeff_l2, npu_asd_detect),
                        OUTPUT(result_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_silent_check_test, case_check_shape_failed_1)
{
    const int32_t c_min_steps = 2;
    const float c_thresh_l1 = 300;
    const float c_coeff_l1 = 1;
    const float c_thresh_l2 = 0.8;
    const float c_coeff_l2 = 0.5;
    const int32_t npu_asd_detect = 1;
    auto val_tensor_desc = TensorDesc({}, ACL_FLOAT).Value(vector<float>{1});
    auto input_grad_tensor_desc = TensorDesc({2}, ACL_FLOAT).Value(vector<float>{1, 2});
    auto sfda_tensor_desc = TensorDesc({2}, ACL_FLOAT).Value(vector<float>{1, 2});
    auto step_tensor_desc = TensorDesc({2}, ACL_INT64).Value(vector<int>{1, 2});
    auto result_tensor_desc = TensorDesc({1}, ACL_INT32);
    auto ut = OP_API_UT(aclnnSilentCheck,
                        INPUT(val_tensor_desc, input_grad_tensor_desc, sfda_tensor_desc, step_tensor_desc, c_min_steps, c_thresh_l1, c_coeff_l1, c_thresh_l2, c_coeff_l2, npu_asd_detect),
                        OUTPUT(result_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
