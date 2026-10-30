/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../op_api/aclnn_confusion_transpose.h"

#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_confusion_transpose_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_confusion_transpose_test Setup" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "l2_confusion_transpose_test TearDown" << std::endl;
    }
};

// 测试所有支持的类型, transpose_first=true
TEST_F(l2_confusion_transpose_test, ascend950_float16_transpose_first)
{
    vector<int64_t> x_shape = {16, 64};
    vector<int64_t> perm_data = {1, 0};
    vector<int64_t> shape_data = {16, 64};
    vector<int64_t> out_shape = {16, 64};

    auto x_desc = TensorDesc(x_shape, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto perm_desc = IntArrayDesc(perm_data);
    auto shape_desc = IntArrayDesc(shape_data);
    bool transpose_first = true;
    auto out_desc = TensorDesc(out_shape, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);

    auto ut = OP_API_UT(
        aclnnConfusionTranspose,                               // host api第二段接口名称
        INPUT(x_desc, perm_desc, shape_desc, transpose_first), // host api输入
        OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    std::cout << "test>> current soc version is " << curSoc << ", aclRet is " << aclRet << std::endl;
    if (curSoc == SocVersion::ASCEND950) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_confusion_transpose_test, ascend950_empty_input)
{
    vector<int64_t> x_shape = {16, 0};
    vector<int64_t> perm_data = {1, 0};
    vector<int64_t> shape_data = {16, 64};
    vector<int64_t> out_shape = {16, 0};

    auto x_desc = TensorDesc(x_shape, ACL_FLOAT16, ACL_FORMAT_ND);
    auto perm_desc = IntArrayDesc(perm_data);
    auto shape_desc = IntArrayDesc(shape_data);
    auto out_desc = TensorDesc(out_shape, ACL_FLOAT16, ACL_FORMAT_ND);
    bool transpose_first = true;

    auto ut = OP_API_UT(
        aclnnConfusionTranspose,                               // host api第二段接口名称
        INPUT(x_desc, perm_desc, shape_desc, transpose_first), // host api输入
        OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    if (curSoc == SocVersion::ASCEND950) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}