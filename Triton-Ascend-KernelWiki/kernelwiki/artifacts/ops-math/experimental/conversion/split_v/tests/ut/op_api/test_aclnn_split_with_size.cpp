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

#include "experimental/conversion/split_v/op_api/aclnn_split_with_size.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class SplitWithSizeApiTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "SplitWithSizeApiTest SetUp" << endl; }

    static void TearDownTestCase() { cout << "SplitWithSizeApiTest TearDown" << endl; }
};

TEST_F(SplitWithSizeApiTest, aclnnSplitWithSize_float32_dim0)
{
    auto selfDesc = TensorDesc({20, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto splitSize = IntArrayDesc(vector<int64_t>{15, 5});
    int64_t dim = 0;
    auto outList = TensorListDesc({
        TensorDesc({15, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001),
        TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001),
    });

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(selfDesc, splitSize, dim), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(SplitWithSizeApiTest, aclnnSplitWithSize_negative_dim)
{
    auto selfDesc = TensorDesc({8, 16, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto splitSize = IntArrayDesc(vector<int64_t>{4, 4, 4, 4});
    int64_t dim = -2;
    auto outList = TensorListDesc({
        TensorDesc({8, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.01, 0.01),
        TensorDesc({8, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.01, 0.01),
        TensorDesc({8, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.01, 0.01),
        TensorDesc({8, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.01, 0.01),
    });

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(selfDesc, splitSize, dim), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(SplitWithSizeApiTest, aclnnSplitWithSize_int32)
{
    auto selfDesc = TensorDesc({10, 5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto splitSize = IntArrayDesc(vector<int64_t>{2, 3, 5});
    int64_t dim = 0;
    auto outList = TensorListDesc({
        TensorDesc({2, 5}, ACL_INT32, ACL_FORMAT_ND),
        TensorDesc({3, 5}, ACL_INT32, ACL_FORMAT_ND),
        TensorDesc({5, 5}, ACL_INT32, ACL_FORMAT_ND),
    });

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(selfDesc, splitSize, dim), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(SplitWithSizeApiTest, aclnnSplitWithSize_empty_tensor)
{
    auto selfDesc = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto splitSize = IntArrayDesc(vector<int64_t>{0, 0});
    int64_t dim = 0;
    auto outList = TensorListDesc({
        TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND),
        TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND),
    });

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(selfDesc, splitSize, dim), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(SplitWithSizeApiTest, aclnnSplitWithSize_invalid_dim_exceeds)
{
    auto selfDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto splitSize = IntArrayDesc(vector<int64_t>{2, 2});
    int64_t dim = 2;
    auto outList = TensorListDesc({
        TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001),
        TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001),
    });

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(selfDesc, splitSize, dim), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(SplitWithSizeApiTest, aclnnSplitWithSize_invalid_split_size_sum)
{
    auto selfDesc = TensorDesc({8, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto splitSize = IntArrayDesc(vector<int64_t>{3, 3});
    int64_t dim = 0;
    auto outList = TensorListDesc({
        TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001),
        TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001),
    });

    auto ut = OP_API_UT(aclnnSplitWithSize, INPUT(selfDesc, splitSize, dim), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
