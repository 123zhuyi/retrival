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

#include "../../../op_api/aclnn_view_copy.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_view_copy_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_view_copy_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_view_copy_test TearDown" << endl; }
};

static TensorDesc MakeMeta(const vector<int64_t>& values, aclDataType dtype = ACL_INT64)
{
    TensorDesc desc({static_cast<int64_t>(values.size())}, dtype, ACL_FORMAT_ND);
    desc.Value(values);
    return desc;
}

TEST_F(l2_view_copy_test, view_copy_empty_tensor_success)
{
    auto dstDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0f, 1.0f);
    auto dstSizeDesc = MakeMeta({1});
    auto dstStrideDesc = MakeMeta({1});
    auto dstOffsetDesc = MakeMeta({0});
    auto srcDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1.0f, 2.0f);
    auto srcSizeDesc = MakeMeta({1});
    auto srcStrideDesc = MakeMeta({1});
    auto srcOffsetDesc = MakeMeta({0});
    auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnViewCopy,
        INPUT(dstDesc, dstSizeDesc, dstStrideDesc, dstOffsetDesc, srcDesc, srcSizeDesc, srcStrideDesc, srcOffsetDesc),
        OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_view_copy_test, view_copy_nullptr)
{
    auto metaDesc = MakeMeta({1});
    auto srcDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnViewCopy,
                        INPUT(nullptr, metaDesc, metaDesc, metaDesc, srcDesc, metaDesc, metaDesc, metaDesc),
                        OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_view_copy_test, view_copy_data_dtype_mismatch)
{
    auto dstDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto metaDesc = MakeMeta({1});
    auto srcDesc = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnViewCopy,
                        INPUT(dstDesc, metaDesc, metaDesc, metaDesc, srcDesc, metaDesc, metaDesc, metaDesc),
                        OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_view_copy_test, view_copy_metadata_shape_mismatch)
{
    auto dstDesc = TensorDesc({12}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dstSizeDesc = MakeMeta({2, 3});
    auto badStrideDesc = MakeMeta({1});
    auto offsetDesc = MakeMeta({0});
    auto srcDesc = TensorDesc({12}, ACL_FLOAT, ACL_FORMAT_ND);
    auto srcSizeDesc = MakeMeta({2, 3});
    auto srcStrideDesc = MakeMeta({3, 1});
    auto outDesc = TensorDesc({12}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnViewCopy,
        INPUT(dstDesc, dstSizeDesc, badStrideDesc, offsetDesc, srcDesc, srcSizeDesc, srcStrideDesc, offsetDesc),
        OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_view_copy_test, view_copy_metadata_dtype_mismatch)
{
    auto dstDesc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND);
    auto int64MetaDesc = MakeMeta({1}, ACL_INT64);
    auto int32MetaDesc = MakeMeta({1}, ACL_INT32);
    auto srcDesc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnViewCopy,
                        INPUT(dstDesc, int64MetaDesc, int32MetaDesc, int64MetaDesc, srcDesc, int64MetaDesc,
                              int64MetaDesc, int64MetaDesc),
                        OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
