/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <thread>
#include <gmock/gmock.h>
#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"
#include "../../../../op_host/op_api/aclnn_grouped_matmul_swiglu_quant_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_GroupedMatmulSwigluQuantV2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_GroupedMatmulSwigluQuantV2_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_GroupedMatmulSwigluQuantV2_test TearDown" << endl;
    }
};

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w8a8_normal_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w8a8_normal_nz_workspace_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight = TensorDesc({e, n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w8a8_wrong_nd_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w8a8_wrong_nd_no_nz_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight = TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w4a8_without_weight_assist_matrix_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w4a4_redundant_weight_assist_matrix_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc weight_assist_matrix = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_assist_matrix_desc = TensorListDesc({weight_assist_matrix});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, weight_assist_matrix_desc, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w4a4_normal_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w4a4_smoothscale_1d_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    TensorDesc smoothScale_desc = TensorDesc({e}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              smoothScale_desc, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w4a4_smoothscale_2d_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    TensorDesc smoothScale_desc = TensorDesc({e, n / 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              smoothScale_desc, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w4a4_smoothscale_invalid_dim_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    TensorDesc smoothScale_desc = TensorDesc({e, n / 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              smoothScale_desc, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w4a4_smoothscale_wrong_shape_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {}, 0, {e, n / 64, k / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    TensorDesc smoothScale_desc = TensorDesc({e + 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              smoothScale_desc, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w4a4_wtrans_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT4, ACL_FORMAT_ND, {e, 1, k}, 0, {e, k / 64, n / 16, 16, 64}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w8a8_multi_weight_normal_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight0 =
        TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc weight1 =
        TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc weight2 =
        TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc weight3 =
        TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight0, weight1, weight2, weight3});
    TensorDesc weight_sacle0 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle1 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle2 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle3 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle0, weight_sacle1, weight_sacle2, weight_sacle3});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w8a8_multi_weight_normal_nz_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight0 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc weight1 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc weight2 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc weight3 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight0, weight1, weight2, weight3});
    TensorDesc weight_sacle0 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle1 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle2 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle3 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle0, weight_sacle1, weight_sacle2, weight_sacle3});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w8a8_multi_weight_normal_nz_workspace_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight0 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc weight1 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc weight2 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc weight3 = TensorDesc({n / 32, k / 16, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight0, weight1, weight2, weight3});
    TensorDesc weight_sacle0 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle1 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle2 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle3 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle0, weight_sacle1, weight_sacle2, weight_sacle3});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 0);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend910B2_test_opapi_w8a8_multi_weight_wrong_nd_no_nz_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t quantGroupSize = 256;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight0 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight1 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight2 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight3 = TensorDesc({k, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorListDesc weight_desc = TensorListDesc({weight0, weight1, weight2, weight3});
    TensorDesc weight_sacle0 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle1 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle2 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc weight_sacle3 = TensorDesc({n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle0, weight_sacle1, weight_sacle2, weight_sacle3});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 64);
    vector<int64_t> tuningConfigVal = { 10 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, 1024}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 0, 0, 0, 0, tuningConfig),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend91095_test_opapi_normal_case)
{
    int64_t m = 2048;
    int64_t k = 7168;
    int64_t n = 4096;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, k / 64, n, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m, k / 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m, k /64 / 2 , 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend91095_test_opapi_illegal_case)
{
    int64_t m = 2048;
    int64_t k = 7168;
    int64_t n = 4096;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, k / 64, n, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m, k / 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m, k /64 / 2 , 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, weight_desc, nullptr, xScale_desc,
                              nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend950_test_opapi_pertoken_normal_case)
{
    int64_t m = 2048;
    int64_t k = 7168;
    int64_t n = 4096;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend950_test_opapi_pertoken_illegal_case_1_wscale_nullptr)
{
    int64_t m = 2048;
    int64_t k = 7168;
    int64_t n = 4096;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, nullptr, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161001);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend950_test_opapi_pertoken_illegal_case_2_shape_mismatch)
{
    int64_t m = 2048;
    int64_t k = 7168;
    int64_t n = 4096;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, n, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend950_test_opapi_pertoken_illegal_case_3_dtype_mismatch)
{
    int64_t m = 2048;
    int64_t k = 7168;
    int64_t n = 4096;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend950_test_opapi_pertoken_illegal_case_4_invalid_format)
{
    int64_t m = 2048;
    int64_t k = 7168;
    int64_t n = 4096;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_FRACTAL_NZ).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                              nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, 161002);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend91095_test_opapi_m0_case)
{
    int64_t m = 0;
    int64_t k = 7168;
    int64_t n = 4096;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, k / 64, n, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m, k / 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m, k /64 / 2 , 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                            nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend91095_test_opapi_n0_case)
{
    int64_t m = 2048;
    int64_t k = 7168;
    int64_t n = 0;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, k / 64, n, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m, k / 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m, k /64 / 2 , 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                            nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_GroupedMatmulSwigluQuantV2_test, ascend91095_test_opapi_k0_case)
{
    int64_t m = 2048;
    int64_t k = 0;
    int64_t n = 4096;
    int64_t e = 8;

    TensorDesc x_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_desc = TensorListDesc({weight});
    TensorDesc weight_sacle = TensorDesc({e, k / 64, n, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorListDesc weight_scale_desc = TensorListDesc({weight_sacle});
    TensorDesc xScale_desc = TensorDesc({m, k / 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc groupList_desc = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    vector<int64_t> tuningConfigVal = { 1 };
    aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
    TensorDesc out1_desc = TensorDesc({m, n / 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-10, 10);
    TensorDesc out2_desc = TensorDesc({m, k /64 / 2 , 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantV2,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, xScale_desc,
                            nullptr, groupList_desc, 2, 0, 2, 2, nullptr),
                        OUTPUT(out1_desc, out2_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}