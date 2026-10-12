/*
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "conversion/strided_slice_v3/op_api/aclnn_slice_v2.h"
#include "../../../../../tests/ut/op_api/op_api_ut_common/inc/op_api_ut_common/op_api_ut.h"
#include "../../../../../tests/ut/op_api/op_api_ut_common/inc/op_api_ut_common/scalar_desc.h"
#include "../../../../../tests/ut/op_api/op_api_ut_common/inc/op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_slicev2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "slicev2_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "slicev2_test TearDown" << endl;
    }
};

TEST_F(l2_slicev2_test, aclnnSliceV2_float32)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dims = IntArrayDesc(vector<int64_t>{0, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});
    auto out = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_float64)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dims = IntArrayDesc(vector<int64_t>{0, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});
    auto out = TensorDesc({3, 2}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_out_float64)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dims = IntArrayDesc(vector<int64_t>{0, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});
    auto out = TensorDesc({3, 2}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_float16)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dims = IntArrayDesc(vector<int64_t>{0, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});
    auto out = TensorDesc({3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_int64)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dims = IntArrayDesc(vector<int64_t>{0, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});
    auto out = TensorDesc({3, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_int32)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dims = IntArrayDesc(vector<int64_t>{0, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});
    auto out = TensorDesc({3, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_int16)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_INT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dims = IntArrayDesc(vector<int64_t>{0, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});
    auto out = TensorDesc({3, 2}, ACL_INT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_int8)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dims = IntArrayDesc(vector<int64_t>{0, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});
    auto out = TensorDesc({3, 2}, ACL_INT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_uint8)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dims = IntArrayDesc(vector<int64_t>{0, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});
    auto out = TensorDesc({3, 2}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_bool)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 5);
    auto dims = IntArrayDesc(vector<int64_t>{0, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});
    auto out = TensorDesc({3, 2}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_self_nullptr)
{
    auto dims = IntArrayDesc(vector<int64_t>{0, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});
    auto out = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT((aclTensor*)nullptr, starts, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_starts_nullptr)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 5);
    auto dims = IntArrayDesc(vector<int64_t>{0, 1});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});
    auto out = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, (aclIntArray*)nullptr, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_ends_nullptr)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 5);
    auto dims = IntArrayDesc(vector<int64_t>{0, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});
    auto out = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, (aclIntArray*)nullptr, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_dims_nullptr)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 5);
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});
    auto out = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, (aclIntArray*)nullptr, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_steps_nullptr)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 5);
    auto dims = IntArrayDesc(vector<int64_t>{0, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto out = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, (aclIntArray*)nullptr), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_out_nullptr)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 5);
    auto dims = IntArrayDesc(vector<int64_t>{0, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, steps), OUTPUT((aclTensor*)nullptr));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_large_shape)
{
    const vector<int64_t>& self_shape = {6, 4, 2, 2, 2, 2, 2, 2, 2};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dims = IntArrayDesc(vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2, 1, 1, 1, 1, 1, 1, 1});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto out = TensorDesc({3, 2, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_error_dims)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dims = IntArrayDesc(vector<int64_t>{5, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});
    auto out = TensorDesc({3, 4}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_error_dims1)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dims = IntArrayDesc(vector<int64_t>{-5, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});
    auto out = TensorDesc({3, 4}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_error_steps)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dims = IntArrayDesc(vector<int64_t>{0, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{-1, 1});
    auto out = TensorDesc({3, 4}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_error_out)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dims = IntArrayDesc(vector<int64_t>{0, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});
    auto out = TensorDesc({3, 4}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slicev2_test, Ascend910B2_aclnnSliceV2_BF16_case)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dims = IntArrayDesc(vector<int64_t>{0, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});
    auto out = TensorDesc({3, 2}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slicev2_test, case_BF16_910)
{
    const vector<int64_t>& self_shape = {6, 4};
    auto self_desc = TensorDesc(self_shape, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dims = IntArrayDesc(vector<int64_t>{0, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});
    auto out = TensorDesc({3, 4}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_empty_float32)
{
    const vector<int64_t>& self_shape = {6, 4, 0};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dims = IntArrayDesc(vector<int64_t>{0, 1});
    auto starts = IntArrayDesc(vector<int64_t>{0, 0});
    auto ends = IntArrayDesc(vector<int64_t>{3, 2});
    auto steps = IntArrayDesc(vector<int64_t>{1, 1});
    auto out = TensorDesc({3, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_slicev2_test, aclnnSliceV2_neg_attr_float32)
{
    const vector<int64_t>& self_shape = {31, 1, 1, 12851, 1, 1, 1};
    auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto dims = IntArrayDesc(vector<int64_t>{3, 2, 0, 6, 5, 4});
    auto starts = IntArrayDesc(vector<int64_t>{-8, -2, 4, -5, 6, -10});
    auto ends = IntArrayDesc(vector<int64_t>{9, 3, -7, 4, -8, -1});
    auto steps = IntArrayDesc(vector<int64_t>{4, 15, 16, 18, 1, 3});
    auto out = TensorDesc({2, 1, 1, 0, 0, 0, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSliceV2, INPUT(self_desc, starts, ends, dims, steps), OUTPUT(out));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}