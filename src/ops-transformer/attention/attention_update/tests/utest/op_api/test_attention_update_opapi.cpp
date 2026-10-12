/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../op_host/op_api/aclnn_attention_update.h"


#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "opdev/op_log.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "common/op_api_def.h"

using namespace op;
using namespace std;

class l2_attention_update_test : public testing::Test {
protected:
 static void SetUpTestCase() { std::cout << "attention_update_test Setup" << std::endl; }
 static void TearDownTestCase() { std::cout << "attention_update_test TearDown" << std::endl; }
};

TEST_F(l2_attention_update_test, ascend910B2_test_attention_update_1) {

    vector<int64_t> lse_dims = {128};
    vector<int64_t> localOut_dims = {128, 256};
    vector<int64_t> out_dims = {128, 256};

    int64_t updateType = 2;

    auto lse1_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto lse_list_desc = TensorListDesc({lse1_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc});

    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate,  // host api第二段接口名称
                        INPUT(lse_list_desc, localOut_list_desc, updateType),  // host api输入
                        OUTPUT(out_desc, nullptr));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_attention_update_test, ascend910B2_test_attention_update_2) {

    vector<int64_t> lse_dims = {128};
    vector<int64_t> localOut_dims = {128, 256};
    vector<int64_t> out_dims = {128, 128};

    int64_t updateType = 0;

    auto lse1_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut2_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc, localOut2_desc});

    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate,  // host api第二段接口名称
                        INPUT(lse_list_desc, localOut_list_desc, updateType),  // host api输入
                        OUTPUT(out_desc, nullptr));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_attention_update_test, ascend910B2_test_attention_update_3) {

    vector<int64_t> lse_dims = {128};
    vector<int64_t> localOut_dims = {128, 256};
    vector<int64_t> out_dims = {128, 256};

    int64_t updateType = 0;

    auto lse1_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut2_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc, localOut2_desc});

    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto outLse_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate,  // host api第二段接口名称
                        INPUT(lse_list_desc, localOut_list_desc, updateType),  // host api输入
                        OUTPUT(out_desc, outLse_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_attention_update_test, ascend910B2_test_attention_update_4) {

    vector<int64_t> lse_dims = {128};
    vector<int64_t> localOut_dims = {128, 256};
    vector<int64_t> out_dims = {128, 256};

    int64_t updateType = 0;

    auto lse1_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut2_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc, localOut2_desc});

    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate,  // host api第二段接口名称
                        INPUT(lse_list_desc, localOut_list_desc, updateType),  // host api输入
                        OUTPUT(out_desc, nullptr));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_attention_update_test, ascend910B2_test_attention_update_5) {

    vector<int64_t> lse_dims = {128};
    vector<int64_t> localOut_dims = {128, 256};
    vector<int64_t> out_dims = {128, 256};
    vector<int64_t> lseout_dims = {128};

    int64_t updateType = 1;

    auto lse1_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut2_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc, localOut2_desc});

    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto lseout_desc = TensorDesc(lseout_dims, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate,  // host api第二段接口名称
                        INPUT(lse_list_desc, localOut_list_desc, updateType),  // host api输入
                        OUTPUT(out_desc, lseout_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_attention_update_test, Ascend910_9589_test_attention_update_1) {

    vector<int64_t> lse_dims = {128};
    vector<int64_t> localOut_dims = {128, 256};
    vector<int64_t> out_dims = {128, 256};

    int64_t updateType = 0;

    auto lse1_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut2_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc, localOut2_desc});

    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate,  // host api第二段接口名称
                        INPUT(lse_list_desc, localOut_list_desc, updateType),  // host api输入
                        OUTPUT(out_desc, nullptr));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_attention_update_test, Ascend910_9589_test_attention_update_2) {

    vector<int64_t> lse_dims = {128};
    vector<int64_t> localOut_dims = {128, 256};
    vector<int64_t> out_dims = {128, 256};

    int64_t updateType = 0;

    auto lse1_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut2_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc, localOut2_desc});

    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto outLse_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate,  // host api第二段接口名称
                        INPUT(lse_list_desc, localOut_list_desc, updateType),  // host api输入
                        OUTPUT(out_desc, outLse_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_attention_update_test, Ascend910_9589_test_attention_update_3) {

    vector<int64_t> lse_dims = {128};
    vector<int64_t> localOut_dims = {128, 256};
    vector<int64_t> out_dims = {128, 256};

    int64_t updateType = 1;

    auto lse1_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto lse2_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut1_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto localOut2_desc = TensorDesc(localOut_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto lse_list_desc = TensorListDesc({lse1_desc, lse2_desc});
    auto localOut_list_desc = TensorListDesc({localOut1_desc, localOut2_desc});

    auto out_desc = TensorDesc(out_dims, ACL_FLOAT, ACL_FORMAT_ND);
    auto go_desc = TensorDesc(lse_dims, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAttentionUpdate,  // host api第二段接口名称
                        INPUT(lse_list_desc, localOut_list_desc, updateType),  // host api输入
                        OUTPUT(out_desc, go_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size); // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}