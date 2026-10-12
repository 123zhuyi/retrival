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

#include "conversion/npu_format_cast/op_host/op_api/aclnn_npu_format_cast.h"
#include "gtest/gtest.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;

class NpuFormatCastEmptyTest : public testing::Test {
protected:
    void SetUp() override { SetPlatformNpuArch(NpuArch::DAV_3510); }
};

TEST_F(NpuFormatCastEmptyTest, NdToNzInfersZeroStorageShapeAndNeedsNoWorkspace)
{
    const std::vector<int64_t> viewShape = {9, 0};
    auto srcTensor = TensorDesc(viewShape, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    int64_t* dstShape = nullptr;
    uint64_t dstShapeSize = 0;
    int actualFormat = ACL_FORMAT_UNDEFINED;

    auto ret = aclnnNpuFormatCastCalculateSizeAndFormat(srcTensor.get(), ACL_FORMAT_FRACTAL_NZ, ACL_FLOAT16, &dstShape,
                                                        &dstShapeSize, &actualFormat);
    ASSERT_EQ(ret, ACLNN_SUCCESS);
    ASSERT_NE(dstShape, nullptr);
    ASSERT_EQ(dstShapeSize, 4);
    EXPECT_EQ(std::vector<int64_t>(dstShape, dstShape + dstShapeSize), (std::vector<int64_t>{0, 1, 16, 16}));
    EXPECT_EQ(actualFormat, ACL_FORMAT_FRACTAL_NZ);

    const std::vector<int64_t> storageShape(dstShape, dstShape + dstShapeSize);
    delete[] dstShape;
    auto dstTensor = TensorDesc(viewShape, ACL_FLOAT16, static_cast<aclFormat>(actualFormat), {}, 0, storageShape)
                         .ToAclType();
    uint64_t workspaceSize = 1;
    aclOpExecutor* executor = nullptr;

    ret = aclnnNpuFormatCastGetWorkspaceSize(srcTensor.get(), dstTensor.get(), &workspaceSize, &executor);
    ASSERT_EQ(ret, ACLNN_SUCCESS);
    ASSERT_EQ(workspaceSize, 0);
    ASSERT_NE(executor, nullptr);
    EXPECT_EQ(aclnnNpuFormatCast(nullptr, workspaceSize, executor, nullptr), ACLNN_SUCCESS);
}

TEST_F(NpuFormatCastEmptyTest, NdToNzRejectsStorageShapeThatDoesNotMatchEmptyView)
{
    const std::vector<int64_t> viewShape = {9, 0};
    auto srcTensor = TensorDesc(viewShape, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    auto dstTensor = TensorDesc(viewShape, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16}).ToAclType();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = aclnnNpuFormatCastGetWorkspaceSize(srcTensor.get(), dstTensor.get(), &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(NpuFormatCastEmptyTest, NzToNdNeedsNoWorkspace)
{
    const std::vector<int64_t> viewShape = {9, 0};
    auto srcTensor = TensorDesc(viewShape, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {0, 1, 16, 16}).ToAclType();
    auto dstTensor = TensorDesc(viewShape, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    uint64_t workspaceSize = 1;
    aclOpExecutor* executor = nullptr;

    auto ret = aclnnNpuFormatCastGetWorkspaceSize(srcTensor.get(), dstTensor.get(), &workspaceSize, &executor);
    ASSERT_EQ(ret, ACLNN_SUCCESS);
    ASSERT_EQ(workspaceSize, 0);
    ASSERT_NE(executor, nullptr);
    EXPECT_EQ(aclnnNpuFormatCast(nullptr, workspaceSize, executor, nullptr), ACLNN_SUCCESS);
}
