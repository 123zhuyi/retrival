/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include "gtest/gtest.h"

#include "experimental/conversion/split_v/op_api/aclnn_split_tensor.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class SplitTensorApiTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "SplitTensorApiTest SetUp" << endl; }

    static void TearDownTestCase() { cout << "SplitTensorApiTest TearDown" << endl; }
};

TEST_F(SplitTensorApiTest, aclnnSplitTensor_float32_dim0)
{
    auto selfDesc = TensorDesc({6, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t splitSections = 4;
    int64_t dim = 0;
    auto outList = TensorListDesc({
        TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001),
        TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001),
    });

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(selfDesc, splitSections, dim), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(SplitTensorApiTest, aclnnSplitTensor_negative_dim)
{
    auto selfDesc = TensorDesc({4, 8, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t splitSections = 2;
    int64_t dim = -2;
    auto outList = TensorListDesc({
        TensorDesc({4, 2, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.01, 0.01),
        TensorDesc({4, 2, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.01, 0.01),
        TensorDesc({4, 2, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.01, 0.01),
        TensorDesc({4, 2, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.01, 0.01),
    });

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(selfDesc, splitSections, dim), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(SplitTensorApiTest, aclnnSplitTensor_int64)
{
    auto selfDesc = TensorDesc({6, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    uint64_t splitSections = 2;
    int64_t dim = 1;
    auto outList = TensorListDesc({
        TensorDesc({6, 2}, ACL_INT64, ACL_FORMAT_ND),
        TensorDesc({6, 2}, ACL_INT64, ACL_FORMAT_ND),
    });

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(selfDesc, splitSections, dim), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(SplitTensorApiTest, aclnnSplitTensor_empty_tensor)
{
    auto selfDesc = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t splitSections = 0;
    int64_t dim = 0;
    auto outList = TensorListDesc({
        TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND),
        TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND),
    });

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(selfDesc, splitSections, dim), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(SplitTensorApiTest, aclnnSplitTensor_invalid_dim_exceeds)
{
    auto selfDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t splitSections = 2;
    int64_t dim = 2;
    auto outList = TensorListDesc({
        TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001),
        TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001),
    });

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(selfDesc, splitSections, dim), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(SplitTensorApiTest, aclnnSplitTensor_invalid_split_sections_zero)
{
    auto selfDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    uint64_t splitSections = 0;
    int64_t dim = 0;
    auto outList = TensorListDesc({
        TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001),
    });

    auto ut = OP_API_UT(aclnnSplitTensor, INPUT(selfDesc, splitSections, dim), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
