/**
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

#include "../../../../op_host/op_api/aclnn_segsum.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_segsum_a5_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "segsum_a5_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "segsum_a5_test TearDown" << endl; }
};

// 空张量早退必须排在入参校验之后: 顺序颠倒会让 dtype 不一致这类非法入参被整段跳过、返回 SUCCESS
TEST_F(l2_segsum_a5_test, ascend950_case_empty_tensor_dtype_invalid)
{
    auto self_desc = TensorDesc({1, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 2, 0, 0}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExpSegsum, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 入参合法的空张量仍按空处理: workspace 为 0 且返回成功
TEST_F(l2_segsum_a5_test, ascend950_case_empty_tensor_valid)
{
    auto self_desc = TensorDesc({1, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 2, 0, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExpSegsum, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_EQ(workspace_size, 0U);
}

// self 为空指针应返回 161001, 而不是在 self->IsEmpty() 处解引用崩溃
TEST_F(l2_segsum_a5_test, ascend950_case_self_nullptr)
{
    auto self_desc = nullptr;
    auto out_desc = TensorDesc({1, 2, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExpSegsum, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
