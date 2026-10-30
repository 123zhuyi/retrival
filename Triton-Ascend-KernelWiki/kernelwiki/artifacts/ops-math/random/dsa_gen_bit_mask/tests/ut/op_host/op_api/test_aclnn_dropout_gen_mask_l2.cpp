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
 * \file test_dropout_gen_mask.cpp
 * \brief
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_dropout_gen_mask.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_dropout_gen_mask_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "dropout_gen_mask_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "dropout_gen_mask_test TearDown" << endl;
    }
};

static inline int64_t ComputeByteSizeAlign(const vector<int64_t>& shape)
{
    int64_t num = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        num *= shape[i];
    }
    return (num + 128 - 1) / 128 * 128 / 8;
}

// check dtype valid
TEST_F(l2_dropout_gen_mask_test, case_001)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto shapeArray = IntArrayDesc(shape);
    double p = 1;

    int64_t seed = 1234;
    int64_t offset = 0;
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutGenMask, INPUT(shapeArray, p, seed, offset), OUTPUT(mask_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check p
TEST_F(l2_dropout_gen_mask_test, case_002)
{
    const vector<int64_t> shape = {2, 1, 3};
    auto shapeArray = IntArrayDesc(shape);
    double p = 1.5;

    int64_t seed = 213;
    int64_t offset = 0;
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutGenMask, INPUT(shapeArray, p, seed, offset), OUTPUT(mask_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check shape
TEST_F(l2_dropout_gen_mask_test, case_003)
{
    const vector<int64_t> shape = {2, 1, 3};
    auto shapeArray = IntArrayDesc(shape);
    double p = 0.5;

    int64_t seed = 213;
    int64_t offset = 0;
    auto mask_desc = TensorDesc({32}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutGenMask, INPUT(shapeArray, p, seed, offset), OUTPUT(mask_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}