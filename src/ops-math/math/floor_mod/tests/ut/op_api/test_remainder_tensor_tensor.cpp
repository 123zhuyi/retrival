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

#include "math/floor_mod/op_api/aclnn_remainder.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "acl/acl.h"

using namespace op;
using namespace std;

class l2_remainder_tensor_tensor_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "remainder_tensor_tensor_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "remainder_tensor_tensor_test TearDown" << std::endl;
    }

    void test_run(
        vector<int64_t> selfDims, aclDataType selfDtype, aclFormat selfFormat, vector<int64_t> selfRange,
        vector<int64_t> otherDims, aclDataType otherDtype, aclFormat otherFormat, vector<int64_t> otherRange,
        vector<int64_t> outDims, aclDataType outDtype, aclFormat outFormat)
    {
        auto self = TensorDesc(selfDims, selfDtype, selfFormat).ValueRange(selfRange[0], selfRange[1]);
        auto other = TensorDesc(otherDims, otherDtype, otherFormat).ValueRange(otherRange[0], otherRange[1]);
        auto out = TensorDesc(outDims, outDtype, outFormat).Precision(0.00001, 0.00001);

        auto ut = OP_API_UT(aclnnRemainderTensorTensor, INPUT(self, other), OUTPUT(out));
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
        // ut.TestPrecision();
    }

    void test_run_invalid(
        vector<int64_t> selfDims, aclDataType selfDtype, aclFormat selfFormat, vector<int64_t> selfRange,
        vector<int64_t> otherDims, aclDataType otherDtype, aclFormat otherFormat, vector<int64_t> otherRange,
        vector<int64_t> outDims, aclDataType outDtype, aclFormat outFormat)
    {
        auto self = TensorDesc(selfDims, selfDtype, selfFormat).ValueRange(selfRange[0], selfRange[1]);
        auto other = TensorDesc(otherDims, otherDtype, otherFormat).ValueRange(otherRange[0], otherRange[1]);
        auto out = TensorDesc(outDims, outDtype, outFormat).Precision(0.00001, 0.00001);

        auto ut = OP_API_UT(aclnnRemainderTensorTensor, INPUT(self, other), OUTPUT(out));
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }
};

///////////////////////////////////////
/////          检查dtype          /////
///////////////////////////////////////

// self + other + out: int32
TEST_F(l2_remainder_tensor_tensor_test, l2_remainder_tensor_tensor_test_01)
{
    test_run(
        {2, 3, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_INT32, ACL_FORMAT_ND, {-15, -10}, {2, 3, 3},
        ACL_INT32, ACL_FORMAT_ND);
}

// self + other + out: int64
TEST_F(l2_remainder_tensor_tensor_test, l2_remainder_tensor_tensor_test_02)
{
    test_run(
        {2, 3, 3}, ACL_INT64, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_INT64, ACL_FORMAT_ND, {-15, -10}, {2, 3, 3},
        ACL_INT64, ACL_FORMAT_ND);
}

// self + other + out: float16
TEST_F(l2_remainder_tensor_tensor_test, l2_remainder_tensor_tensor_test_03)
{
    test_run(
        {2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-15, -10}, {2, 3, 3},
        ACL_FLOAT16, ACL_FORMAT_ND);
}

// self + other + out: float32
TEST_F(l2_remainder_tensor_tensor_test, l2_remainder_tensor_tensor_test_04)
{
    test_run(
        {2, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND, {-15, -10}, {2, 3, 3},
        ACL_FLOAT, ACL_FORMAT_ND);
}

// self + other + out: float64
TEST_F(l2_remainder_tensor_tensor_test, l2_remainder_tensor_tensor_test_05)
{
    test_run(
        {2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {-15, -10}, {2, 3, 3},
        ACL_DOUBLE, ACL_FORMAT_ND);
}

// self + other + out: 不支持bool、uint8、int8、int16、bfloat16、complex64、complex128
TEST_F(l2_remainder_tensor_tensor_test, l2_remainder_tensor_tensor_test_06)
{
    test_run_invalid(
        {2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND, {-15, -10}, {2, 3, 3},
        ACL_BOOL, ACL_FORMAT_ND);
    test_run_invalid(
        {2, 3, 3}, ACL_UINT8, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_UINT8, ACL_FORMAT_ND, {-15, -10}, {2, 3, 3},
        ACL_UINT8, ACL_FORMAT_ND);
    test_run_invalid(
        {2, 3, 3}, ACL_INT8, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_INT8, ACL_FORMAT_ND, {-15, -10}, {2, 3, 3},
        ACL_INT8, ACL_FORMAT_ND);
    test_run_invalid(
        {2, 3, 3}, ACL_INT16, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_INT16, ACL_FORMAT_ND, {-15, -10}, {2, 3, 3},
        ACL_INT16, ACL_FORMAT_ND);
    // test_run_invalid({2, 3, 3}, ACL_BF16, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_BF16, ACL_FORMAT_ND, {-15, -10},
    //     {2, 3, 3}, ACL_BF16, ACL_FORMAT_ND);
    test_run_invalid(
        {2, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_ND, {-15, -10},
        {2, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
    test_run_invalid(
        {2, 3, 3}, ACL_COMPLEX128, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_COMPLEX128, ACL_FORMAT_ND, {-15, -10},
        {2, 3, 3}, ACL_COMPLEX128, ACL_FORMAT_ND);
}

// self.dtype != other.dtype 会进行promote type  promoteType必须符合预期
TEST_F(l2_remainder_tensor_tensor_test, l2_remainder_tensor_tensor_test_07)
{
    // int16 + float16 -> float16
    test_run(
        {2, 3, 3}, ACL_INT16, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {3, 10}, {2, 3, 3},
        ACL_FLOAT16, ACL_FORMAT_ND);

    // int32 + uint8 -> int32
    test_run(
        {2, 3, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_UINT8, ACL_FORMAT_ND, {5, 10}, {2, 3, 3},
        ACL_INT32, ACL_FORMAT_ND);
    // int32 + int64 -> int64
    test_run(
        {2, 3, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_INT64, ACL_FORMAT_ND, {5, 10}, {2, 3, 3},
        ACL_INT64, ACL_FORMAT_ND);
    // int32 + float64 -> float64
    test_run(
        {2, 3, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {5, 10}, {2, 3, 3},
        ACL_DOUBLE, ACL_FORMAT_ND);

    // int64 + uint8 -> int64
    test_run(
        {2, 3, 3}, ACL_INT64, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_UINT8, ACL_FORMAT_ND, {5, 10}, {2, 3, 3},
        ACL_INT64, ACL_FORMAT_ND);
    // int64 + float64 -> float64
    test_run(
        {2, 3, 3}, ACL_INT64, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {5, 10}, {2, 3, 3},
        ACL_DOUBLE, ACL_FORMAT_ND);

    // fp16 + uint8 -> fp16
    test_run(
        {2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_UINT8, ACL_FORMAT_ND, {3, 10}, {2, 3, 3},
        ACL_FLOAT16, ACL_FORMAT_ND);
    // fp16 + int64 -> fp16
    test_run(
        {2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_INT64, ACL_FORMAT_ND, {3, 10}, {2, 3, 3},
        ACL_FLOAT16, ACL_FORMAT_ND);
    // fp16 + fp64 -> fp64
    test_run(
        {2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {3, 10}, {2, 3, 3},
        ACL_DOUBLE, ACL_FORMAT_ND);

    // fp64 + uin8 -> fp64
    test_run(
        {2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_UINT8, ACL_FORMAT_ND, {3, 10}, {2, 3, 3},
        ACL_DOUBLE, ACL_FORMAT_ND);
    // fp64 + int64-> fp64
    test_run(
        {2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_INT64, ACL_FORMAT_ND, {3, 10}, {2, 3, 3},
        ACL_DOUBLE, ACL_FORMAT_ND);
    // fp64 + fp32 -> fp64
    test_run(
        {2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND, {3, 10}, {2, 3, 3},
        ACL_DOUBLE, ACL_FORMAT_ND);

    // promote以后的dtype不符合预期
    test_run_invalid(
        {2, 3, 3}, ACL_INT8, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_INT16, ACL_FORMAT_ND, {3, 10}, {2, 3, 3},
        ACL_DOUBLE, ACL_FORMAT_ND);
    test_run_invalid(
        {2, 3, 3}, ACL_INT16, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_INT8, ACL_FORMAT_ND, {3, 10}, {2, 3, 3},
        ACL_DOUBLE, ACL_FORMAT_ND);
}

// out.dtype不一致时会进行cast, 不能cast的场景会报错
TEST_F(l2_remainder_tensor_tensor_test, l2_remainder_tensor_tensor_test_08)
{
    // 向同类型的最顶部 cast
    // int32 + uint8 -> int64
    test_run(
        {2, 3, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_UINT8, ACL_FORMAT_ND, {2, 10}, {2, 3, 3},
        ACL_INT64, ACL_FORMAT_ND);

    // 向同类型的最底部 cast   精度异常
    // int32 + int8 -> int8
    test_run(
        {2, 3, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_INT8, ACL_FORMAT_ND, {2, 10}, {2, 3, 3},
        ACL_INT8, ACL_FORMAT_ND);

    // 向不同类型的最顶部 cast
    // int32 + int64 -> float64
    test_run(
        {2, 3, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_INT64, ACL_FORMAT_ND, {2, 10}, {2, 3, 3},
        ACL_DOUBLE, ACL_FORMAT_ND);

    // 向不同类型的最底部 cast
    // int32 + uint8 -> float64
    test_run(
        {2, 3, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_UINT8, ACL_FORMAT_ND, {2, 10}, {2, 3, 3},
        ACL_FLOAT16, ACL_FORMAT_ND);

    // 向同类型的最顶部 cast
    // float32 + uint8 -> float64
    test_run(
        {2, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_UINT8, ACL_FORMAT_ND, {2, 10}, {2, 3, 3},
        ACL_DOUBLE, ACL_FORMAT_ND);

    // 向同类型的最底部 cast
    // float32 + float64 -> float16
    test_run(
        {2, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {5, 10}, {2, 3, 3},
        ACL_FLOAT16, ACL_FORMAT_ND);

    // 向不同类型的最顶部 cast   精度异常  CPU不允许cast
    // float16 + int8 -> int64
    test_run_invalid(
        {2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_INT8, ACL_FORMAT_ND, {5, 10}, {2, 3, 3},
        ACL_INT64, ACL_FORMAT_ND);

    // 向不同类型的最底部 cast  精度异常   CPU不允许cast
    // float16 + float64 -> int8
    test_run_invalid(
        {2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {5, 10}, {2, 3, 3},
        ACL_INT8, ACL_FORMAT_ND);

    // 支持out为complex
    test_run(
        {2, 3, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {5, 10}, {2, 3, 3},
        ACL_COMPLEX64, ACL_FORMAT_ND);
}

// ///////////////////////////////////////
// /////          检查空指针          /////
// ///////////////////////////////////////

TEST_F(l2_remainder_tensor_tensor_test, l2_remainder_tensor_tensor_test_09)
{
    uint64_t workspaceSize = 0;
    auto self = TensorDesc({2, 3, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto other = TensorDesc({2, 3, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(3, 10);
    auto out = TensorDesc({2, 3, 3}, ACL_INT64, ACL_FORMAT_ND).Precision(0.00001, 0.00001);

    auto ut = OP_API_UT(aclnnRemainderTensorTensor, INPUT(nullptr, other), OUTPUT(out));
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnRemainderTensorTensor, INPUT(self, nullptr), OUTPUT(out));
    getWorkspaceResult = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnRemainderTensorTensor, INPUT(self, other), OUTPUT(nullptr));
    getWorkspaceResult = ut3.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

///////////////////////////////////////
/////         支持空tensor         /////
///////////////////////////////////////

TEST_F(l2_remainder_tensor_tensor_test, l2_remainder_tensor_tensor_test_10)
{
    // self empty + other empty
    test_run(
        {2, 3, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {2, 3, 0}, ACL_INT8, ACL_FORMAT_ND, {7, 10}, {2, 3, 0},
        ACL_FLOAT, ACL_FORMAT_ND);
    // 不支持空tensor时混着不支持的数据类型
    test_run_invalid(
        {2, 0, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, {2, 0, 3}, ACL_COMPLEX128, ACL_FORMAT_ND, {7, 10}, {2, 0, 3},
        ACL_FLOAT, ACL_FORMAT_ND);

    // self empty
    // self empty + other 1维
    test_run(
        {0}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {1}, ACL_INT8, ACL_FORMAT_ND, {7, 10}, {0}, ACL_FLOAT, ACL_FORMAT_ND);
    // self empty + other 0维
    test_run(
        {0}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {}, ACL_INT8, ACL_FORMAT_ND, {7, 10}, {0}, ACL_FLOAT, ACL_FORMAT_ND);

    // other empty
    // self 1维 + other empty
    test_run(
        {1}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {0}, ACL_INT8, ACL_FORMAT_ND, {7, 10}, {0}, ACL_FLOAT, ACL_FORMAT_ND);
    // self 0维 + other empty
    test_run(
        {}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {0}, ACL_INT8, ACL_FORMAT_ND, {7, 10}, {0}, ACL_FLOAT, ACL_FORMAT_ND);

    // 因为不能broadcast，所以运行失败
    test_run_invalid(
        {2, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_INT8, ACL_FORMAT_ND, {7, 10}, {2, 0, 3},
        ACL_FLOAT, ACL_FORMAT_ND);
}

///////////////////////////////////////
/////       支持非连续tensor       /////
///////////////////////////////////////

TEST_F(l2_remainder_tensor_tensor_test, l2_remainder_tensor_tensor_test_11)
{
    uint64_t workspaceSize = 0;
    auto self = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto selfT = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 10);
    auto other = TensorDesc({5, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto otherT = TensorDesc({5, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(6, 10);
    auto out = TensorDesc({5, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);

    // self not contiguous
    auto ut = OP_API_UT(aclnnRemainderTensorTensor, INPUT(selfT, other), OUTPUT(out));
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    // ut.TestPrecision();

    // other not contiguous
    ut = OP_API_UT(aclnnRemainderTensorTensor, INPUT(self, otherT), OUTPUT(out));
    getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    // ut.TestPrecision();
}

///////////////////////////////////////
/////          检查shape          /////
///////////////////////////////////////

// out.shape不为broadcast以后的shape
TEST_F(l2_remainder_tensor_tensor_test, l2_remainder_tensor_tensor_test_12)
{
    // out.shape为broadcast以后的shape
    // 3维 broadcast成4维
    test_run(
        {3, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {5, 3, 3, 5}, ACL_DOUBLE, ACL_FORMAT_ND, {1, 10}, {5, 3, 3, 5},
        ACL_DOUBLE, ACL_FORMAT_ND);
    test_run(
        {5, 3, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {3, 3, 5}, ACL_DOUBLE, ACL_FORMAT_ND, {1, 10}, {5, 3, 3, 5},
        ACL_DOUBLE, ACL_FORMAT_ND);
    // 4维 broadcast成5维
    test_run(
        {3, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {5, 3, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND, {1, 10},
        {5, 3, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND);
    test_run(
        {5, 3, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {3, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND, {1, 10},
        {5, 3, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND);

    // out.shape不为broadcast以后的shape
    test_run_invalid(
        {2, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {2, 2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {1, 10}, {2, 2, 3, 4},
        ACL_DOUBLE, ACL_FORMAT_ND);
}

///////////////////////////////////////
/////          0维 ~ 8维          /////
///////////////////////////////////////

TEST_F(l2_remainder_tensor_tensor_test, l2_remainder_tensor_tensor_test_13)
{
    // 0维
    // self 0维， other 0维, out 0维
    test_run(
        {}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {}, ACL_DOUBLE, ACL_FORMAT_ND, {6, 10}, {}, ACL_DOUBLE, ACL_FORMAT_ND);
    // self 0维， other 0维, out 1维
    test_run(
        {}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {}, ACL_DOUBLE, ACL_FORMAT_ND, {6, 10}, {1}, ACL_DOUBLE,
        ACL_FORMAT_ND);
    // self 0维,  other 1维, out 0维
    test_run(
        {}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {1}, ACL_DOUBLE, ACL_FORMAT_ND, {6, 10}, {}, ACL_DOUBLE,
        ACL_FORMAT_ND);
    // self 0维,  other 1维, out 1维
    test_run(
        {}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {1}, ACL_DOUBLE, ACL_FORMAT_ND, {6, 10}, {1}, ACL_DOUBLE,
        ACL_FORMAT_ND);
    // self 1维,  other 0维, out 0维
    test_run(
        {1}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {}, ACL_DOUBLE, ACL_FORMAT_ND, {6, 10}, {}, ACL_DOUBLE,
        ACL_FORMAT_ND);
    // self 1维,  other 0维, out 1维
    test_run(
        {1}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {}, ACL_DOUBLE, ACL_FORMAT_ND, {6, 10}, {1}, ACL_DOUBLE,
        ACL_FORMAT_ND);
    // self 1维,  other 1维, out 0维
    test_run(
        {1}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {1}, ACL_DOUBLE, ACL_FORMAT_ND, {6, 10}, {}, ACL_DOUBLE,
        ACL_FORMAT_ND);

    // 1维
    test_run(
        {3}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {3}, ACL_DOUBLE, ACL_FORMAT_ND, {6, 10}, {3}, ACL_DOUBLE,
        ACL_FORMAT_ND);
    // 3维
    test_run(
        {3, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {3, 3, 4}, ACL_DOUBLE, ACL_FORMAT_ND, {6, 10}, {3, 3, 4},
        ACL_DOUBLE, ACL_FORMAT_ND);
    // 4维
    test_run(
        {1, 2, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {1, 2, 1, 1}, ACL_DOUBLE, ACL_FORMAT_ND, {6, 10},
        {1, 2, 1, 1}, ACL_DOUBLE, ACL_FORMAT_ND);

    // 8维
    test_run(
        {1, 3, 1, 2, 3, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {1, 3, 1, 2, 3, 2, 2, 2}, ACL_DOUBLE,
        ACL_FORMAT_ND, {6, 10}, {1, 3, 1, 2, 3, 2, 2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
    test_run(
        {1, 3, 1, 2, 3, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {3, 1, 2, 3, 2, 2, 2}, ACL_DOUBLE, ACL_FORMAT_ND,
        {6, 10}, {1, 3, 1, 2, 3, 2, 2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);

    // 9维
    test_run_invalid(
        {1, 3, 1, 2, 3, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {1, 3, 1, 2, 3, 2, 2, 2, 2}, ACL_DOUBLE,
        ACL_FORMAT_ND, {6, 10}, {1, 3, 1, 2, 3, 2, 2, 2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
    test_run_invalid(
        {1, 3, 1, 2, 3, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {3, 1, 2, 3, 2, 2, 2, 2}, ACL_DOUBLE,
        ACL_FORMAT_ND, {6, 10}, {1, 3, 1, 2, 3, 2, 2, 2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
}

///////////////////////////////////////
/////        tensor正负值         /////
///////////////////////////////////////

// 支持正负值
TEST_F(l2_remainder_tensor_tensor_test, l2_remainder_tensor_tensor_test_14)
{
    // self > 0, other > 0
    test_run(
        {3, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND, {3, 10}, {3, 3, 5}, ACL_DOUBLE, ACL_FORMAT_ND, {1, 10}, {3, 3, 5},
        ACL_DOUBLE, ACL_FORMAT_ND);
    // self > 0, other < 0
    test_run(
        {3, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND, {3, 10}, {3, 3, 5}, ACL_DOUBLE, ACL_FORMAT_ND, {-10, -1}, {3, 3, 5},
        ACL_DOUBLE, ACL_FORMAT_ND);
    // self < 0, other > 0
    test_run(
        {3, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND, {-10, -1}, {3, 3, 5}, ACL_DOUBLE, ACL_FORMAT_ND, {1, 10}, {3, 3, 5},
        ACL_DOUBLE, ACL_FORMAT_ND);
    // self < 0, other < 0
    test_run(
        {3, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND, {-10, -1}, {3, 3, 5}, ACL_DOUBLE, ACL_FORMAT_ND, {-10, -1}, {3, 3, 5},
        ACL_DOUBLE, ACL_FORMAT_ND);
}

TEST_F(l2_remainder_tensor_tensor_test, ascend310P_l2_remainder_tensor_tensor_test_15)
{
    // 0维
    // self 0维， other 0维, out 0维
    test_run(
        {}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {}, ACL_DOUBLE, ACL_FORMAT_ND, {6, 10}, {}, ACL_DOUBLE, ACL_FORMAT_ND);
}

TEST_F(l2_remainder_tensor_tensor_test, l2_remainder_tensor_tensor_test_16)
{
    // 0维
    // self 0维， other 0维, out 0维
    test_run_invalid(
        {}, ACL_BF16, ACL_FORMAT_ND, {-10, 10}, {}, ACL_BF16, ACL_FORMAT_ND, {6, 10}, {}, ACL_BF16, ACL_FORMAT_ND);
}

TEST_F(l2_remainder_tensor_tensor_test, ascend910B2_l2_remainder_tensor_tensor_test_17)
{
    // 0维
    // self 0维， other 0维, out 0维
    test_run(
        {}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {}, ACL_DOUBLE, ACL_FORMAT_ND, {6, 10}, {}, ACL_DOUBLE, ACL_FORMAT_ND);
}

TEST_F(l2_remainder_tensor_tensor_test, Ascend950PR_99_l2_remainder_tensor_tensor_test_03)
{
    test_run(
        {2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-15, -10}, {2, 3, 3},
        ACL_FLOAT16, ACL_FORMAT_ND);
}
