/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_matmul_reduce_scatter.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace MatmulReduceScatter {
class L2AclnnMatmulReduceScatterTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2AclnnMatmulReduceScatterTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2AclnnMatmulReduceScatterTest TearDown" << endl;
    }
};

TEST_F(L2AclnnMatmulReduceScatterTest, TestAclnnMatmulReduceScatterFirstApi)
{
  TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2Desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
  EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnMatmulReduceScatterTest, TestAclnnMatmulReduceScatterFirstApi2)
{
  TensorDesc x1Desc = TensorDesc({0, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2Desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc outDesc = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
  EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnMatmulReduceScatterTest, TestAclnnMatmulReduceScatterFirstApi3)
{
  TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2Desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
  EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2AclnnMatmulReduceScatterTest, TestSixApi950)
{
  TensorDesc x1Desc = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc x2Desc = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc outDesc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMatmulReduceScatter, INPUT(x1Desc, x2Desc, bias, "test_group", "sum", 8, 1), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
  EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}
} // MatmulReduceScatter