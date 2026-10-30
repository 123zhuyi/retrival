/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "math/reduce_log_sum/op_api/aclnn_reduce_log_sum.h"
#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "acl/acl.h"

using namespace std;

class l2_reduce_log_sum_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "reduce_log_sum_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "reduce_log_sum_test TearDown" << endl;
    }
};

// 正常场景 - float16
TEST_F(l2_reduce_log_sum_test, l2_reduce_log_sum_float16)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 8);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    auto outTensorDesc = TensorDesc({1, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValidCount(4);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 - float32
TEST_F(l2_reduce_log_sum_test, l2_reduce_log_sum_float32)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 8);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    auto outTensorDesc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(4);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 - dim为-1
TEST_F(l2_reduce_log_sum_test, l2_reduce_log_sum_dim_negative_one)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 8);
    auto dim = IntArrayDesc(vector<int64_t>{-1});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    auto outTensorDesc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(2);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 - keep_dim为false
TEST_F(l2_reduce_log_sum_test, l2_reduce_log_sum_keep_dim_false)
{
    auto xDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 8);
    auto dim = IntArrayDesc(vector<int64_t>{-1, 0});
    bool keep_dim = false;
    bool noopWithEmptyAxes = false;
    auto outTensorDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(3);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 - 多维度reduce
TEST_F(l2_reduce_log_sum_test, l2_reduce_log_sum_multi_dim_keep_dim_true)
{
    auto xDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 8);
    auto dim = IntArrayDesc(vector<int64_t>{1, 2});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    auto outTensorDesc = TensorDesc({2, 1, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(10);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 - 多维度reduce keep_dim为false
TEST_F(l2_reduce_log_sum_test, l2_reduce_log_sum_multi_dim_keep_dim_false)
{
    auto xDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 8);
    auto dim = IntArrayDesc(vector<int64_t>{1, 2});
    bool keep_dim = false;
    bool noopWithEmptyAxes = false;
    auto outTensorDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(10);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 - dim为空 noopWithEmptyAxes为true
TEST_F(l2_reduce_log_sum_test, l2_reduce_log_sum_empty_dim_noop_true)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 8);
    auto dim = IntArrayDesc(vector<int64_t>{});
    bool keep_dim = false;
    bool noopWithEmptyAxes = true;
    auto outTensorDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(8);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 - dim为空 noopWithEmptyAxes为false
TEST_F(l2_reduce_log_sum_test, l2_reduce_log_sum_empty_dim_noop_false)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 8);
    auto dim = IntArrayDesc(vector<int64_t>{});
    bool keep_dim = false;
    bool noopWithEmptyAxes = false;
    auto outTensorDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(1);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 - 所有Format
TEST_F(l2_reduce_log_sum_test, l2_reduce_log_sum_all_format)
{
    vector<aclFormat> formats{ACL_FORMAT_ND, ACL_FORMAT_NCHW, ACL_FORMAT_NC, ACL_FORMAT_NCL};
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    for (auto format : formats) {
        auto xDesc = TensorDesc({2, 2, 2, 3}, ACL_FLOAT, format).ValueRange(-50, 50);
        auto dim = IntArrayDesc(vector<int64_t>{0});
        auto outTensorDesc = TensorDesc({1, 2, 2, 3}, ACL_FLOAT, format).ValidCount(12);
        auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

// 异常场景 - data为空指针
TEST_F(l2_reduce_log_sum_test, l2_reduce_log_sum_nullptr_self)
{
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(nullptr, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(getWorkspaceResult, ACLNN_ERR_INNER_NULLPTR);
}

// 异常场景 - axes为空指针
TEST_F(l2_reduce_log_sum_test, l2_reduce_log_sum_nullptr_dim)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    auto outDesc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(selfDesc, nullptr, keep_dim, noopWithEmptyAxes), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(getWorkspaceResult, ACLNN_ERR_INNER_NULLPTR);
}

// 异常场景 - reduce为空指针
TEST_F(l2_reduce_log_sum_test, l2_reduce_log_sum_nullptr_reduce)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(selfDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(getWorkspaceResult, ACLNN_ERR_INNER_NULLPTR);
}

// 异常场景 - 不支持的self数据类型
TEST_F(l2_reduce_log_sum_test, l2_reduce_log_sum_self_dtype_not_support)
{
    vector<aclDataType> dtypes{ACL_INT64, ACL_INT8,  ACL_BOOL,      ACL_DOUBLE,    ACL_INT32,
                               ACL_UINT8, ACL_INT16, ACL_COMPLEX64, ACL_COMPLEX128};
    for (auto dtype : dtypes) {
        auto selfDesc = TensorDesc({2, 4}, dtype, ACL_FORMAT_ND);
        auto dim = IntArrayDesc(vector<int64_t>{0});
        bool keep_dim = true;
        bool noopWithEmptyAxes = false;
        auto outDesc = TensorDesc({1, 4}, dtype, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(selfDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outDesc));
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }
}

// 异常场景 - dim超出范围
TEST_F(l2_reduce_log_sum_test, l2_reduce_log_sum_dim_out_of_range)
{
    auto xDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 8);
    auto dim = IntArrayDesc(vector<int64_t>{1, 3});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    auto outTensorDesc = TensorDesc({2, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(8);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景 - 维度超过8维
TEST_F(l2_reduce_log_sum_test, l2_reduce_log_sum_dim_exceeds_limit)
{
    auto xDesc = TensorDesc({2, 2, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 8);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    auto outTensorDesc = TensorDesc({1, 2, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(2);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景 - reduce shape不匹配
TEST_F(l2_reduce_log_sum_test, l2_reduce_log_sum_reduce_shape_not_match)
{
    auto xDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 8);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    auto outTensorDesc = TensorDesc({99, 99}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(10);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 边界场景 - 空tensor
TEST_F(l2_reduce_log_sum_test, l2_reduce_log_sum_empty_tensor)
{
    auto xDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    auto outTensorDesc = TensorDesc({1, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 边界场景 - 0维tensor
TEST_F(l2_reduce_log_sum_test, l2_reduce_log_sum_zero_dim_tensor)
{
    auto xDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    auto outTensorDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 不连续场景 - self discontinues
TEST_F(l2_reduce_log_sum_test, l2_reduce_log_sum_self_discontinues)
{
    auto xDesc =
        TensorDesc({2, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {40, 20, 1, 5}, 0, {2, 2, 4, 5}).ValueRange(-50, 50);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    auto outTensorDesc = TensorDesc({1, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(40);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 不连续场景 - out discontinues
TEST_F(l2_reduce_log_sum_test, l2_reduce_log_sum_out_discontinues)
{
    auto xDesc = TensorDesc({2, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto dim = IntArrayDesc(vector<int64_t>{0});
    bool keep_dim = true;
    bool noopWithEmptyAxes = false;
    auto outTensorDesc =
        TensorDesc({1, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {40, 20, 1, 5}, 0, {1, 2, 4, 5}).ValidCount(40);
    auto ut = OP_API_UT(aclnnReduceLogSum, INPUT(xDesc, dim, keep_dim, noopWithEmptyAxes), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
