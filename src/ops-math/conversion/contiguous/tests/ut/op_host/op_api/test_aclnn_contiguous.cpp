/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#ifdef __cplusplus
extern "C" {
#endif
extern aclnnStatus aclnnContiguousGetWorkspaceSize(
    const aclTensorList* self, uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnContiguous(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);
extern aclnnStatus aclnnViewCopyGetWorkspaceSize(
    const aclTensorList* in, const aclTensorList* out, uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnViewCopy(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);
#ifdef __cplusplus
}
#endif

using namespace std;
using namespace op;

static constexpr int64_t CONTIGUOUS_DATA_SIZE = 1024 * 1024;

class l2_contiguous_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "contiguous_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "contiguous_test TearDown" << endl;
    }

public:
    aclTensor* CreateAclTensor(
        vector<int64_t> view_shape, vector<int64_t> stride, int64_t offset, vector<int64_t> storage_shape,
        aclDataType dataType = ACL_FLOAT, int64_t dataOffset = 0)
    {
        return aclCreateTensor(
            view_shape.data(), view_shape.size(), dataType, stride.data(), offset, ACL_FORMAT_ND, storage_shape.data(),
            storage_shape.size(), data + dataOffset);
    }

    int64_t data[CONTIGUOUS_DATA_SIZE] = {0};
};

TEST_F(l2_contiguous_test, test_seccess)
{
    auto tensor = CreateAclTensor({3, 5, 7, 6}, {210, 42, 1, 7}, 0, {4, 5, 6, 7});
    tensor->SetViewFormat(static_cast<op::Format>(ACL_FORMAT_NCHW));
    tensor->SetOriginalFormat(static_cast<op::Format>(ACL_FORMAT_NCHW));
    tensor->SetStorageFormat(static_cast<op::Format>(ACL_FORMAT_NCHW));
    auto tensor_list = aclCreateTensorList(&tensor, 1);
    uint64_t workspaceSize = 0U;
    aclOpExecutor* exe = nullptr;
    auto aclRet = aclnnContiguousGetWorkspaceSize(tensor_list, &workspaceSize, &exe);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(exe, nullptr);
}

TEST_F(l2_contiguous_test, test_viewcopy)
{
    auto unContTensor = CreateAclTensor({4, 5, 6, 7}, {210, 42, 1, 7}, 0, {4, 5, 7, 6});
    unContTensor->SetViewFormat(static_cast<op::Format>(ACL_FORMAT_NCHW));
    unContTensor->SetOriginalFormat(static_cast<op::Format>(ACL_FORMAT_NCHW));
    unContTensor->SetStorageFormat(static_cast<op::Format>(ACL_FORMAT_NCHW));
    auto unContTensorList = aclCreateTensorList(&unContTensor, 1);

    auto contTensor = CreateAclTensor({4, 5, 6, 7}, {210, 42, 7, 1}, 0, {4, 5, 6, 7});
    contTensor->SetViewFormat(static_cast<op::Format>(ACL_FORMAT_NCHW));
    contTensor->SetOriginalFormat(static_cast<op::Format>(ACL_FORMAT_NCHW));
    contTensor->SetStorageFormat(static_cast<op::Format>(ACL_FORMAT_NCHW));
    auto contTensorList = aclCreateTensorList(&contTensor, 1);

    uint64_t workspaceSize = 0U;
    aclOpExecutor* exe = nullptr;
    auto aclRet = aclnnViewCopyGetWorkspaceSize(contTensorList, unContTensorList, &workspaceSize, &exe);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(exe, nullptr);
}
