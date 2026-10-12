/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>
#include "gtest/gtest.h"

#include "conversion/im2col/op_api/aclnn_im2col.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

namespace {
class Im2colApiTest : public testing::Test {
protected:
    static void SetUpTestCase() { op::SetPlatformSocVersion(op::SocVersion::ASCEND910B); }
};

TEST_F(Im2colApiTest, supports_bool_rank4)
{
    auto self = TensorDesc({1, 2, 2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 8, 4}, ACL_BOOL, ACL_FORMAT_ND);
    auto kernel = IntArrayDesc(std::vector<int64_t>{2, 2});
    auto dilation = IntArrayDesc(std::vector<int64_t>{1, 1});
    auto padding = IntArrayDesc(std::vector<int64_t>{1, 1});
    auto stride = IntArrayDesc(std::vector<int64_t>{2, 2});

    auto test = OP_API_UT(aclnnIm2col, INPUT(self, kernel, dilation, padding, stride), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(test.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(Im2colApiTest, supports_float16_rank3)
{
    auto self = TensorDesc({2, 5, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({12, 15}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto kernel = IntArrayDesc(std::vector<int64_t>{3, 2});
    auto dilation = IntArrayDesc(std::vector<int64_t>{1, 1});
    auto padding = IntArrayDesc(std::vector<int64_t>{1, 0});
    auto stride = IntArrayDesc(std::vector<int64_t>{2, 1});

    auto test = OP_API_UT(aclnnIm2col, INPUT(self, kernel, dilation, padding, stride), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(test.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(Im2colApiTest, rejects_unsupported_dtype)
{
    auto self = TensorDesc({1, 2, 2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 8, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto kernel = IntArrayDesc(std::vector<int64_t>{2, 2});
    auto dilation = IntArrayDesc(std::vector<int64_t>{1, 1});
    auto padding = IntArrayDesc(std::vector<int64_t>{1, 1});
    auto stride = IntArrayDesc(std::vector<int64_t>{2, 2});

    auto test = OP_API_UT(aclnnIm2col, INPUT(self, kernel, dilation, padding, stride), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(test.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(Im2colApiTest, rejects_wrong_output_shape)
{
    auto self = TensorDesc({1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 8, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto kernel = IntArrayDesc(std::vector<int64_t>{2, 2});
    auto dilation = IntArrayDesc(std::vector<int64_t>{1, 1});
    auto padding = IntArrayDesc(std::vector<int64_t>{1, 1});
    auto stride = IntArrayDesc(std::vector<int64_t>{2, 2});

    auto test = OP_API_UT(aclnnIm2col, INPUT(self, kernel, dilation, padding, stride), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(test.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(Im2colApiTest, rejects_invalid_attribute_length)
{
    auto self = TensorDesc({1, 2, 2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 8, 4}, ACL_BF16, ACL_FORMAT_ND);
    auto kernel = IntArrayDesc(std::vector<int64_t>{2});
    auto dilation = IntArrayDesc(std::vector<int64_t>{1, 1});
    auto padding = IntArrayDesc(std::vector<int64_t>{1, 1});
    auto stride = IntArrayDesc(std::vector<int64_t>{2, 2});

    auto test = OP_API_UT(aclnnIm2col, INPUT(self, kernel, dilation, padding, stride), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(test.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(Im2colApiTest, rejects_output_channel_overflow)
{
    auto self = TensorDesc({1, 2, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto kernel = IntArrayDesc(std::vector<int64_t>{3037000500LL, 3037000500LL});
    auto dilation = IntArrayDesc(std::vector<int64_t>{1, 1});
    auto padding = IntArrayDesc(std::vector<int64_t>{3037000500LL, 3037000500LL});
    auto stride = IntArrayDesc(std::vector<int64_t>{1, 1});

    auto test = OP_API_UT(aclnnIm2col, INPUT(self, kernel, dilation, padding, stride), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(test.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}
} // namespace
