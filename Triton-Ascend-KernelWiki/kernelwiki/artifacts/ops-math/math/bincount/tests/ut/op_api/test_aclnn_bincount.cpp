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

#include "../../../op_api/aclnn_bincount.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_bincount_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_bincount_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_bincount_test TearDown" << std::endl;
    }
};

TEST_F(l2_bincount_test, ascend950PR_99_input_int8)
{
    auto self_tensor_desc =
        TensorDesc({39}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                                                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                                                    27, 28, 29, 30, 1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto weights_tensor_desc = TensorDesc({39}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(2, 2);
    int64_t minlength = 0;
    auto out_tensor_desc = TensorDesc({31}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut =
        OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));
}

TEST_F(l2_bincount_test, ascend950PR_99_input_uint8)
{
    auto self_tensor_desc =
        TensorDesc({39}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<int>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                                     1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto weights_tensor_desc = TensorDesc({39}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(2, 2);
    int64_t minlength = 0;
    auto out_tensor_desc = TensorDesc({31}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut =
        OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));
}

TEST_F(l2_bincount_test, ascend950PR_99_input_int16)
{
    auto self_tensor_desc =
        TensorDesc({39}, ACL_INT16, ACL_FORMAT_ND).Value(vector<int>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                                     1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto weights_tensor_desc = TensorDesc({39}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(2, 2);
    int64_t minlength = 0;
    auto out_tensor_desc = TensorDesc({31}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut =
        OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));
}

TEST_F(l2_bincount_test, ascend950PR_99_input_int32)
{
    auto self_tensor_desc =
        TensorDesc({39}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                                     1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto weights_tensor_desc = TensorDesc({39}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(2, 2);
    int64_t minlength = 0;
    auto out_tensor_desc = TensorDesc({31}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut =
        OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));
}

TEST_F(l2_bincount_test, ascend950PR_99_input_int64)
{
    auto self_tensor_desc =
        TensorDesc({39}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                                     1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto weights_tensor_desc = TensorDesc({39}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(2, 2);
    int64_t minlength = 0;
    auto out_tensor_desc = TensorDesc({31}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut =
        OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));
}

TEST_F(l2_bincount_test, ascend950PR_99_input_int8_minlength)
{
    auto self_tensor_desc =
        TensorDesc({39}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                                                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                                                    27, 28, 29, 30, 1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto weights_tensor_desc = TensorDesc({39}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(2, 2);
    int64_t minlength = 100;
    auto out_tensor_desc = TensorDesc({100}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut =
        OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));
}

TEST_F(l2_bincount_test, ascend950PR_99_input_uint8_minlength)
{
    auto self_tensor_desc =
        TensorDesc({39}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<int>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                                     1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto weights_tensor_desc = TensorDesc({39}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(2, 2);
    int64_t minlength = 100;
    auto out_tensor_desc = TensorDesc({100}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut =
        OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));
}

TEST_F(l2_bincount_test, ascend950PR_99_input_int16_minlength)
{
    auto self_tensor_desc =
        TensorDesc({39}, ACL_INT16, ACL_FORMAT_ND).Value(vector<int>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                                     1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto weights_tensor_desc = TensorDesc({39}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(2, 2);
    int64_t minlength = 100;
    auto out_tensor_desc = TensorDesc({100}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut =
        OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));
}

TEST_F(l2_bincount_test, ascend950PR_99_input_int32_minlength)
{
    auto self_tensor_desc =
        TensorDesc({39}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                                     1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto weights_tensor_desc = TensorDesc({39}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(2, 2);
    int64_t minlength = 100;
    auto out_tensor_desc = TensorDesc({100}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut =
        OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));
}

TEST_F(l2_bincount_test, ascend950PR_99_input_int64_minlength)
{
    auto self_tensor_desc =
        TensorDesc({39}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                                     1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto weights_tensor_desc = TensorDesc({39}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(2, 2);
    int64_t minlength = 100;
    auto out_tensor_desc = TensorDesc({100}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut =
        OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));
}

// weights dtype test
TEST_F(l2_bincount_test, ascend950PR_99_input_int8_BOOL)
{
    auto self_tensor_desc =
        TensorDesc({39}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                                                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                                                    27, 28, 29, 30, 1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto weights_tensor_desc = TensorDesc({39}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(false, true);
    int64_t minlength = 0;
    auto out_tensor_desc = TensorDesc({31}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut =
        OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));
}

TEST_F(l2_bincount_test, ascend950PR_99_input_int8_weights_dtypes)
{
    vector<aclDataType> dtypes{ACL_INT8, ACL_UINT8, ACL_INT16, ACL_INT32, ACL_INT64, ACL_FLOAT16, ACL_DOUBLE};
    for (auto dtype : dtypes) {
        auto self_tensor_desc =
            TensorDesc({39}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                                        1,  2,  3,  4,  5,  6,  7,  8,  9});
        auto weights_tensor_desc = TensorDesc({39}, dtype, ACL_FORMAT_ND).ValueRange(false, true);
        int64_t minlength = 0;
        auto out_tensor_desc = TensorDesc({31}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);
        auto ut =
            OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));
    }
}

// weights nullptr
TEST_F(l2_bincount_test, ascend950PR_99_input_int8_weights_nullptr)
{
    auto self_tensor_desc =
        TensorDesc({39}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                                                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                                                    27, 28, 29, 30, 1,  2,  3,  4,  5,  6,  7,  8,  9});
    aclTensor* weights_tensor_desc = nullptr;
    int64_t minlength = 0;
    auto out_tensor_desc = TensorDesc({31}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut =
        OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));
}

// input  null tensor
TEST_F(l2_bincount_test, ascend950PR_99_input_null_tensor)
{
    auto self_tensor_desc = TensorDesc({0}, ACL_INT8, ACL_FORMAT_ND);
    auto weights_tensor_desc = TensorDesc({0}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t minlength = 0;
    auto out_tensor_desc = TensorDesc({0}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut =
        OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));
}

TEST_F(l2_bincount_test, ascend950PR_99_input_null_tensor_with_minlength)
{
    auto self_tensor_desc = TensorDesc({0}, ACL_INT8, ACL_FORMAT_ND);
    auto weights_tensor_desc = TensorDesc({0}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t minlength = 2;
    auto out_tensor_desc = TensorDesc({2}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut =
        OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));
}

// Abnormal Scenarios
TEST_F(l2_bincount_test, ascend950PR_99_input_weights_shape_not_same)
{
    auto self_tensor_desc = TensorDesc({16}, ACL_INT8, ACL_FORMAT_ND)
                                .Value(vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto weights_tensor_desc = TensorDesc({10}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t minlength = 0;
    auto out_tensor_desc = TensorDesc({17}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut =
        OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_bincount_test, ascend950PR_99_input_weights_not_1_dims)
{
    auto self_tensor_desc = TensorDesc({16, 1}, ACL_INT8, ACL_FORMAT_ND)
                                .Value(vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto weights_tensor_desc = TensorDesc({16, 1}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t minlength = 0;
    auto out_tensor_desc = TensorDesc({17}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut =
        OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_bincount_test, ascend950PR_99_input_dtype_not_support)
{
    auto self_tensor_desc = TensorDesc({16}, ACL_DOUBLE, ACL_FORMAT_ND)
                                .Value(vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto weights_tensor_desc = TensorDesc({16}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t minlength = 0;
    auto out_tensor_desc = TensorDesc({17}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut =
        OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_bincount_test, ascend950PR_99_weights_dtype_not_support)
{
    auto self_tensor_desc = TensorDesc({16}, ACL_DOUBLE, ACL_FORMAT_ND)
                                .Value(vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto weights_tensor_desc = TensorDesc({16}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t minlength = 0;
    auto out_tensor_desc = TensorDesc({17}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut =
        OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_bincount_test, ascend950PR_99_input_nullptr)
{
    aclTensor* self_tensor_desc = nullptr;
    auto weights_tensor_desc = TensorDesc({16}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t minlength = 0;
    auto out_tensor_desc = TensorDesc({17}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut =
        OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_bincount_test, ascend950PR_99_output_nullptr)
{
    auto self_tensor_desc = TensorDesc({16}, ACL_DOUBLE, ACL_FORMAT_ND)
                                .Value(vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto weights_tensor_desc = TensorDesc({16}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t minlength = 0;
    aclTensor* out_tensor_desc = nullptr;
    auto ut =
        OP_API_UT(aclnnBincount, INPUT(self_tensor_desc, weights_tensor_desc, minlength), OUTPUT(out_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
