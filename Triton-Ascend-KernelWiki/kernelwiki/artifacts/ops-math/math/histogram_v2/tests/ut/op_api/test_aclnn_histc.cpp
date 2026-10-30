/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <limits>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_histc.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class l2_histc_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "histc_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "histc_test TearDown" << endl; }

    // Restore the default platform after every test (some tests switch platform to
    // exercise the AiCore / AiCPU / RegBase branches in histogram.cpp).
    void TearDown() override { op::SetPlatformSocVersion(op::SocVersion::ASCEND910B); }
};

TEST_F(l2_histc_test, case_000_workspace)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);

    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_001_float32_normal)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);

    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_002_float16_normal)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);

    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_003_int32_normal)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);

    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_008_1_dim_input_tensor)
{
    auto selfTensor = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);

    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_009_3_dim_input_tensor)
{
    auto selfTensor = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);

    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_010_5_dim_input_tensor)
{
    auto selfTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);

    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_011_8_dim_input_tensor)
{
    auto selfTensor = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);

    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_012_bins_coverage)
{
    auto selfTensor = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 10;
    auto minScalar = ScalarDesc(-10.0f);
    auto maxScalar = ScalarDesc(10.0f);

    auto ut1 = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    outTensor = TensorDesc({12}, ACL_FLOAT, ACL_FORMAT_ND);
    bins = 12;
    auto ut2 = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    workspaceSize = 0;
    aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    outTensor = TensorDesc({14}, ACL_FLOAT, ACL_FORMAT_ND);
    bins = 14;
    auto ut3 = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    workspaceSize = 0;
    aclRet = ut3.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_014_NHWC)
{
    auto selfTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NHWC);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-10.0f);
    auto maxScalar = ScalarDesc(10.0f);

    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_015_NCHW)
{
    auto selfTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-10.0f);
    auto maxScalar = ScalarDesc(10.0f);

    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_016_HWCN)
{
    auto selfTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_HWCN);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-10.0f);
    auto maxScalar = ScalarDesc(10.0f);

    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_018_empty_tensor)
{
    auto selfTensor = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 2;
    auto minScalar = ScalarDesc(-10.0f);
    auto maxScalar = ScalarDesc(10.0f);

    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_histc_test, case_019_float32_min_greater_max)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(9.0f);
    auto maxScalar = ScalarDesc(-9.0f);

    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_histc_test, case_020_min_greater_max)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(9.0f);
    auto maxScalar = ScalarDesc(-9.0f);

    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ---------------------- nullptr checks (CheckNotNull) ----------------------
TEST_F(l2_histc_test, case_021_null_self)
{
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT((aclTensor*)nullptr, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_histc_test, case_022_null_out)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT((aclTensor*)nullptr));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_histc_test, case_023_null_min)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto maxScalar = ScalarDesc(9.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, (aclScalar*)nullptr, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_histc_test, case_024_null_max)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, (aclScalar*)nullptr), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

// ---------------------- dtype validity (CheckDtypeValid) ----------------------
TEST_F(l2_histc_test, case_025_invalid_self_dtype)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_histc_test, case_026_invalid_out_dtype)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_DOUBLE, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// ---------------------- bins / shape checks ----------------------
TEST_F(l2_histc_test, case_027_bins_non_positive)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 0;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_histc_test, case_028_out_wrong_dim)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_histc_test, case_029_out_size_ne_bins)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_histc_test, case_030_self_exceed_max_dim)
{
    auto selfTensor = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// ---------------------- min/max inf & nan (CheckMinMaxIsInfNan) ----------------------
TEST_F(l2_histc_test, case_031_min_pos_inf_only)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(std::numeric_limits<float>::infinity());
    auto maxScalar = ScalarDesc(9.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_histc_test, case_032_max_nan)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(std::numeric_limits<float>::quiet_NaN());
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_histc_test, case_033_min_max_both_pos_inf)
{
    // min == max == +inf is a valid equal range (CheckMinMaxInfEqual) and triggers min/max recompute.
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(std::numeric_limits<float>::infinity());
    auto maxScalar = ScalarDesc(std::numeric_limits<float>::infinity());
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_034_min_max_both_neg_inf)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-std::numeric_limits<float>::infinity());
    auto maxScalar = ScalarDesc(-std::numeric_limits<float>::infinity());
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// ---------------------- min == max recompute (NeedComputeMinMax / AllMinMax) ----------------------
TEST_F(l2_histc_test, case_035_float_min_eq_max)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(0.0f);
    auto maxScalar = ScalarDesc(0.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_036_int_min_eq_max)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(0.0f);
    auto maxScalar = ScalarDesc(0.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_037_scalar_self_min_eq_max)
{
    // 0-dim self exercises the AllMinMax dimNum == 0 branch.
    auto selfTensor = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(0.0f);
    auto maxScalar = ScalarDesc(0.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// ---------------------- remaining integer dtypes ----------------------
TEST_F(l2_histc_test, case_038_int8_normal)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_039_uint8_normal)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 10);
    auto outTensor = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(0.0f);
    auto maxScalar = ScalarDesc(9.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_040_int16_normal)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(l2_histc_test, case_041_int64_normal)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// ---------------------- platform-specific dispatch (histogram.cpp) ----------------------
// RegBase (ascend950) with fp32 output exercises the AiCore RegBase desDtype branch.
TEST_F(l2_histc_test, case_042_regbase_out_fp32)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// RegBase with min == max exercises the NeedComputeMinMax RegBase recompute path.
TEST_F(l2_histc_test, case_043_regbase_min_eq_max)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(0.0f);
    auto maxScalar = ScalarDesc(0.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// A non-AiCore-supported arch (ascend910 / DAV_1001) routes to the AiCPU path.
TEST_F(l2_histc_test, case_044_aicpu_path)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910);
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// AiCPU path with fp16 input exercises the desDtype fp16->fp32 promotion branch.
TEST_F(l2_histc_test, case_045_aicpu_fp16)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910);
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// ascend310p (DAV_2002) is AiCore-supported and covers that npuArch branch.
TEST_F(l2_histc_test, case_046_dav2002_path)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND310P);
    auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);
    int64_t bins = 3;
    auto minScalar = ScalarDesc(-9.0f);
    auto maxScalar = ScalarDesc(9.0f);
    auto ut = OP_API_UT(aclnnHistc, INPUT(selfTensor, bins, minScalar, maxScalar), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}
