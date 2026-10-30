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
#include "platform/platform_info.h"
#include "../../../op_api/aclnn_repeat.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_repeat_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "repeat_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "repeat_test TearDown" << std::endl;
    }

    void test_run(
        vector<int64_t> selfDims, aclDataType selfDtype, aclFormat selfFormat, vector<int64_t> selfRange,
        vector<int64_t> repeatsDims, vector<int64_t> outDims, aclDataType outDtype, aclFormat outFormat)
    {
        auto self = TensorDesc(selfDims, selfDtype, selfFormat).ValueRange(selfRange[0], selfRange[1]);
        auto repeats = IntArrayDesc(repeatsDims);
        auto out = TensorDesc(outDims, outDtype, outFormat).Precision(0.001, 0.001);

        auto ut = OP_API_UT(aclnnRepeat, INPUT(self, repeats), OUTPUT(out));
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
        // ut.TestPrecision();
    }

    void test_run_invalid(
        vector<int64_t> selfDims, aclDataType selfDtype, aclFormat selfFormat, vector<int64_t> selfRange,
        vector<int64_t> repeatsDims, vector<int64_t> outDims, aclDataType outDtype, aclFormat outFormat)
    {
        auto self = TensorDesc(selfDims, selfDtype, selfFormat).ValueRange(selfRange[0], selfRange[1]);
        auto repeats = IntArrayDesc(repeatsDims);
        auto out = TensorDesc(outDims, outDtype, outFormat).Precision(0.0001, 0.0001);

        auto ut = OP_API_UT(aclnnRepeat, INPUT(self, repeats), OUTPUT(out));
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }
};

///////////////////////////////////////
/////          检查dtype          /////
///////////////////////////////////////

TEST_F(l2_repeat_test, l2_repeat_test_dtype_bf16_error)
{
    test_run({1, 2}, ACL_BF16, ACL_FORMAT_ND, {-10, 10}, {3, 2, 3}, {3, 2, 6}, ACL_BF16, ACL_FORMAT_ND);
}

///////////////////////////////////////
/////         检查format          /////
///////////////////////////////////////

// 所有format都支持
TEST_F(l2_repeat_test, l2_repeat_test_format)
{
    vector<aclFormat> ValidList = {ACL_FORMAT_ND,   ACL_FORMAT_NCHW,  ACL_FORMAT_NHWC,
                                   ACL_FORMAT_HWCN, ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};

    int64_t length = ValidList.size();
    vector<int64_t> input_dim = {1, 2};
    vector<int64_t> repeats_dim = {3, 2, 3};
    vector<int64_t> result_dim = {3, 2, 6};

    for (int i = 0; i < length; i++) {
        auto inputDesc = TensorDesc(input_dim, ACL_FLOAT16, ValidList[i]).ValueRange(-2, 2);
        auto repeatsDesc = IntArrayDesc(repeats_dim);
        auto outDesc = TensorDesc(result_dim, ACL_FLOAT16, ValidList[i]).Precision(0.001, 0.001);

        auto ut = OP_API_UT(aclnnRepeat, INPUT(inputDesc, repeatsDesc), OUTPUT(outDesc));
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        // ut.TestPrecision();
    }
}

// self.dtype = out.dtype
TEST_F(l2_repeat_test, l2_repeat_test_same_dtype_error)
{
    test_run_invalid({1, 2}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 2, 3}, {3, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
}

///////////////////////////////////////
/////          检查空指针          /////
///////////////////////////////////////

TEST_F(l2_repeat_test, l2_repeat_test_nullptr)
{
    uint64_t workspaceSize = 0;
    auto self = TensorDesc({1, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto repeats = IntArrayDesc({3, 2, 3});
    auto out = TensorDesc({3, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int64_t outputSize = 36;

    auto ut1 = OP_API_UT(aclnnRepeat, INPUT(nullptr, repeats), OUTPUT(out));
    aclnnStatus getWorkspaceResult = ut1.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnRepeat, INPUT(self, nullptr), OUTPUT(out));
    getWorkspaceResult = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

///////////////////////////////////////
/////         支持空tensor         /////
///////////////////////////////////////

TEST_F(l2_repeat_test, l2_repeat_test_null_tensor)
{
    // self
    test_run({0, 2}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 2, 3}, {3, 0, 6}, ACL_FLOAT16, ACL_FORMAT_ND);

    // repeats
    test_run({1, 2}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 0, 3}, {3, 0, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
}

///////////////////////////////////////
/////       支持非连续tensor       /////
///////////////////////////////////////

TEST_F(l2_repeat_test, l2_repeat_test_uncontiguous)
{
    auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {3, 2}).ValueRange(-1, 1);
    auto repeats = IntArrayDesc({3, 2, 3});
    auto out = TensorDesc({3, 4, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    // self not contiguous
    auto ut = OP_API_UT(aclnnRepeat, INPUT(self, repeats), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    // ut.TestPrecision();
}

///////////////////////////////////////
/////         检查repeats         /////
///////////////////////////////////////

TEST_F(l2_repeat_test, l2_repeat_repeats_size)
{
    // repeats的维度数应大于self的维度书
    test_run_invalid({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {1, 3}, {2, 1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);

    // repeats中的值应大于等于0
    test_run_invalid({1, 2}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {-1, 3}, {-1, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
}

///////////////////////////////////////
/////          0维 ~ 8维          /////
///////////////////////////////////////

TEST_F(l2_repeat_test, l2_repeat_test_dim_8)
{
    // 0维
    test_run({}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 3}, {2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    // 8维
    test_run(
        {1, 2, 1, 2, 1, 1, 2, 1}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 2, 2, 2, 2, 2, 2, 2},
        {2, 4, 2, 4, 2, 2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    // repeats szie > 8
    test_run_invalid(
        {1, 2, 1, 2, 1, 1, 2, 1}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 2, 2, 2, 2, 2, 2, 2, 2},
        {2, 2, 4, 2, 4, 2, 2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    // self dim > 8
    test_run_invalid(
        {1, 2, 1, 2, 1, 1, 2, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 2, 2, 2, 2, 2, 2, 2, 2},
        {2, 4, 2, 4, 2, 2, 4, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
}

////////////////////////////////////////////////
///// self及repeats均为scalar时self直接搬出 /////
////////////////////////////////////////////////

TEST_F(l2_repeat_test, l2_repeat_test_repeats_zero_size)
{
    // self及repeats均为scalar
    test_run({}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {}, {}, ACL_FLOAT16, ACL_FORMAT_ND);
}