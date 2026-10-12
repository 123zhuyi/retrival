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

#include "../../../op_api/aclnn_mul.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_mul_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "mul_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "mul_test TearDown" << endl;
    }
};

TEST_F(l2_mul_test, case_nullptr)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMul, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr), OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_mul_test, case_001)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnMul, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mul_test, case_002)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnMul, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mul_test, case_003)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnMul, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mul_test, case_004)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnMul, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mul_test, case_005)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnMul, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mul_test, case_006)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnMul, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mul_test, case_007)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnMul, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_mul_test, case_008)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);

    auto ut = OP_API_UT(aclnnMul, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// ---------------------------------------------------------------------------------------
// White-box coverage of the MulPreferContiguous routing predicate (host-side dispatch).
// It fires for an operand X (vs Y) iff ALL of:
//   A: broadcast factor(X vs Y) >= 320   B: numel(X) >= 256
//   C: inner contiguous run(X) < 32      D: gather stride(X) >= 24
// The HIT path needs a sparse strided view into a larger buffer (storage > view), which the
// op_api UT TensorDesc (storage == product(view_shape)) cannot represent; the HIT decision,
// thresholds and no-regression are verified on-device + by the offline truth-table. The op_api
// UT below uses storage-fitting views (transpose / stride-0) to exercise each MISS branch and
// guard the routing helpers against crashes; all must return ACL_SUCCESS.
// ---------------------------------------------------------------------------------------

// MISS (A fails): transposed view, no broadcast -> factor == 1
TEST_F(l2_mul_test, case_wb_miss_no_broadcast)
{
    auto self_desc = TensorDesc({8, 64, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {16, 128, 1}, 0).ValueRange(-1, 1);
    auto other_desc = TensorDesc({8, 64, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_desc = TensorDesc({8, 64, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMul, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}

// MISS (B fails): broadcast transposed view but payload (64) below threshold
TEST_F(l2_mul_test, case_wb_miss_small_payload)
{
    auto self_desc = TensorDesc({8, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_desc = TensorDesc({1, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND, {64, 1, 8}, 0).ValueRange(-1, 1);
    auto out_desc = TensorDesc({8, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMul, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}

// MISS (C fails): transposed broadcast with a long contiguous run (64) -> vectorizable, keep non-contig.
// (Order-independent run detection: the stride-1 dim is not innermost here.)
TEST_F(l2_mul_test, case_wb_miss_contiguous_run)
{
    auto self_desc = TensorDesc({320, 64, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto other_desc = TensorDesc({1, 64, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {1024, 1, 64}, 0).ValueRange(-1, 1);
    auto out_desc = TensorDesc({320, 64, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMul, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}

// MISS (C fails): pure stride-0 broadcast with an inner contiguous run == 32
TEST_F(l2_mul_test, case_wb_miss_zero_stride_broadcast)
{
    auto self_desc = TensorDesc({64, 64, 32}, ACL_FLOAT16, ACL_FORMAT_ND, {32, 0, 1}, 0).ValueRange(-1, 1);
    auto other_desc = TensorDesc({64, 64, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_desc = TensorDesc({64, 64, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMul, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
