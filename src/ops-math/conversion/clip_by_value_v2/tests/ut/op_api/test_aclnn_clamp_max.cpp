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

#include "conversion/clip_by_value_v2/op_api/aclnn_clamp.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_clamp_max_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "clamp_max_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "clamp_max_test TearDown" << std::endl;
    }
};

// 各元素特殊类型覆盖用例
// 空tensor   self ,out为空时，空进空出
TEST_F(l2_clamp_max_test, aclnnClampMax_float_nd_empty_tensor)
{
    // self input
    const vector<int64_t>& selfShape = {0};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;

    // out input
    const vector<int64_t>& outShape = {0};
    aclDataType outDtype = selfDtype;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maxScalar = ScalarDesc(static_cast<float>(2));
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnClampMax, INPUT(selfTensorDesc, maxScalar), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 报错类型覆盖用例
// 空指针
TEST_F(l2_clamp_max_test, aclnnClampMax_input_nullptr)
{
    auto tensor_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxScalar = ScalarDesc(static_cast<float>(2));

    auto ut_self_nullptr = OP_API_UT(aclnnClampMax, INPUT((aclTensor*)nullptr, maxScalar), OUTPUT(tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut_self_nullptr.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_max_nullptr = OP_API_UT(aclnnClampMax, INPUT(tensor_desc, (aclScalar*)nullptr), OUTPUT(tensor_desc));

    aclRet = ut_max_nullptr.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_out_nullptr = OP_API_UT(aclnnClampMax, INPUT(tensor_desc, maxScalar), OUTPUT((aclTensor*)nullptr));

    aclRet = ut_out_nullptr.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// self形状大于8
TEST_F(l2_clamp_max_test, aclnnClampMax_self_shape_out_of_8)
{
    // self input
    const vector<int64_t>& selfShape = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;

    // out input
    const vector<int64_t>& outShape = selfShape;
    aclDataType outDtype = selfDtype;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maxScalar = ScalarDesc(static_cast<float>(2));
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnClampMax, INPUT(selfTensorDesc, maxScalar), OUTPUT(outTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_clamp_max_test, ascend950pr_9589_aclnnClampMax_input_diff)
{
    auto tensor_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxScalar = ScalarDesc(static_cast<float>(2));

    auto ut_self_nullptr = OP_API_UT(aclnnClampMax, INPUT((aclTensor*)nullptr, maxScalar), OUTPUT(tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut_self_nullptr.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_max_nullptr = OP_API_UT(aclnnClampMax, INPUT(tensor_desc, (aclScalar*)nullptr), OUTPUT(tensor_desc));

    aclRet = ut_max_nullptr.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_out_nullptr = OP_API_UT(aclnnClampMax, INPUT(tensor_desc, maxScalar), OUTPUT((aclTensor*)nullptr));

    aclRet = ut_out_nullptr.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}