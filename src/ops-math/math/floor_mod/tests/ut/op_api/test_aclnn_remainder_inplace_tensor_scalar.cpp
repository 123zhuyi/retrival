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

class l2_remainder_inplace_tensor_scalar_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "remainder_inplace_tensor_scalar_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "remainder_inplace_tensor_scalar_test TearDown" << std::endl;
    }

    void test_run(
        vector<int64_t> selfDims, aclDataType selfDtype, aclFormat selfFormat, vector<int64_t> selfRange,
        ScalarDesc other)
    {
        auto self = TensorDesc(selfDims, selfDtype, selfFormat)
                        .ValueRange(selfRange[0], selfRange[1])
                        .Precision(0.00001, 0.00001);

        auto ut = OP_API_UT(aclnnInplaceRemainderTensorScalar, INPUT(self, other), OUTPUT());
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
        // ut.TestPrecision();
    }

    void test_run_invalid(
        vector<int64_t> selfDims, aclDataType selfDtype, aclFormat selfFormat, vector<int64_t> selfRange,
        ScalarDesc other)
    {
        auto self = TensorDesc(selfDims, selfDtype, selfFormat)
                        .ValueRange(selfRange[0], selfRange[1])
                        .Precision(0.00001, 0.00001);

        auto ut = OP_API_UT(aclnnInplaceRemainderTensorScalar, INPUT(self, other), OUTPUT());
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }
};

///////////////////////////////////////
/////          检查dtype          /////
///////////////////////////////////////

// self + other + out: int32
TEST_F(l2_remainder_inplace_tensor_scalar_test, l2_remainder_inplace_tensor_scalar_test_01)
{
    int32_t value = 3;
    auto other_desc = ScalarDesc(value);
    test_run({2, 3, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, other_desc);
}

// self + other + out: int64
// TEST_F(l2_remainder_inplace_tensor_scalar_test, l2_remainder_inplace_tensor_scalar_test_02)
// {
//     int64_t value = 3;
//     auto other_desc = ScalarDesc(value);
//     test_run({2, 3, 3}, ACL_INT64, ACL_FORMAT_ND, {-10, 10}, other_desc);
// }

// // self + other(int32) + out: float16
// TEST_F(l2_remainder_inplace_tensor_scalar_test, l2_remainder_inplace_tensor_scalar_test_03)
// {
//     int32_t value = 2;
//     auto other_desc = ScalarDesc(value);
//     test_run({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, other_desc);
// }

// // self + other + out: float32
// TEST_F(l2_remainder_inplace_tensor_scalar_test, l2_remainder_inplace_tensor_scalar_test_04)
// {
//     float value = 3.5;
//     auto other_desc = ScalarDesc(value);
//     test_run({2, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, other_desc);
// }

// // self + other + out: float64
// TEST_F(l2_remainder_inplace_tensor_scalar_test, l2_remainder_inplace_tensor_scalar_test_05)
// {
//     double value = 3.14;
//     auto other_desc = ScalarDesc(value);
//     test_run({2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {-10, 10}, other_desc);
// }

// self + other + out: 不支持self和other都为bool、uint8、int8、int16、bfloat16、complex64、complex128
TEST_F(l2_remainder_inplace_tensor_scalar_test, l2_remainder_inplace_tensor_scalar_test_06)
{
    bool value_bool = true;
    uint8_t value_uint8 = 3;
    int8_t value_int8 = 3;
    int16_t value_int16 = 3;
    auto other_desc_bool = ScalarDesc(value_bool);
    auto other_desc_uint8 = ScalarDesc(value_uint8);
    auto other_desc_int8 = ScalarDesc(value_int8);
    auto other_desc_int16 = ScalarDesc(value_int16);

    test_run_invalid({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND, {-10, 10}, other_desc_bool);
    test_run_invalid({2, 3, 3}, ACL_UINT8, ACL_FORMAT_ND, {-10, 10}, other_desc_uint8);
    test_run_invalid({2, 3, 3}, ACL_INT8, ACL_FORMAT_ND, {-10, 10}, other_desc_int8);
    test_run_invalid({2, 3, 3}, ACL_INT16, ACL_FORMAT_ND, {-10, 10}, other_desc_int16);
    test_run_invalid({2, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_ND, {-10, 10}, other_desc_uint8);
    test_run_invalid({2, 3, 3}, ACL_COMPLEX128, ACL_FORMAT_ND, {-10, 10}, other_desc_uint8);
}

// self dtype必须属于支持的数据类型 + other能cast成对应的数据类型
// TEST_F(l2_remainder_inplace_tensor_scalar_test, l2_remainder_inplace_tensor_scalar_test_07)
// {
//     int64_t value_int64 = 3;
//     auto other_desc_int64 = ScalarDesc(value_int64);
//     double value_float64 = 3.5;
//     auto other_desc_float64 = ScalarDesc(value_float64);
//     // self不属于，other属于 -> inplace不支持
//     test_run_invalid({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND, {-10, 10}, other_desc_int64);
//     test_run_invalid({2, 3, 3}, ACL_UINT8, ACL_FORMAT_ND, {-10, 10}, other_desc_int64);
//     test_run_invalid({2, 3, 3}, ACL_INT8, ACL_FORMAT_ND, {-10, 10}, other_desc_int64);
//     test_run_invalid({2, 3, 3}, ACL_INT16, ACL_FORMAT_ND, {-10, 10}, other_desc_int64);
//     test_run_invalid({2, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_ND, {-10, 10}, other_desc_int64);
//     test_run_invalid({2, 3, 3}, ACL_COMPLEX128, ACL_FORMAT_ND, {-10, 10}, other_desc_int64);

//     // self属于，other不属于
//     bool value_bool = true;
//     uint8_t value_uint8 = 3;
//     int8_t value_int8 = 3;
//     int16_t value_int16 = 3;
//     auto other_desc_bool = ScalarDesc(value_bool);
//     auto other_desc_uint8 = ScalarDesc(value_uint8);
//     auto other_desc_int8 = ScalarDesc(value_int8);
//     auto other_desc_int16 = ScalarDesc(value_int16);
//     test_run({2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {-10, 10}, other_desc_bool);
//     test_run({2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {-10, 10}, other_desc_uint8);
//     test_run({2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {-10, 10}, other_desc_int8);
//     test_run({2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {-10, 10}, other_desc_int16);

//     // other能cast成对应的数据类型
//     test_run({2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {-10, 10}, other_desc_int64);
//     test_run_invalid({2, 3, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, other_desc_float64);
// }

// // self.dtype != other.dtype 会进行promote type   inplace必须self为promoteType
// TEST_F(l2_remainder_inplace_tensor_scalar_test, l2_remainder_inplace_tensor_scalar_test_08)
// {
//     // int32 + uint8 -> int32
//     uint8_t value_uint8 = 3;
//     auto other_desc = ScalarDesc(value_uint8);
//     test_run({2, 3, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, other_desc);
//     // int32 + int64 -> int32
//     int64_t value_int64 = 3;
//     other_desc = ScalarDesc(value_int64);
//     test_run({2, 3, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, other_desc);
//     // int32 + float64 -> int32
//     double value_fp64 = 3.55;
//     other_desc = ScalarDesc(value_fp64);
//     test_run_invalid({2, 3, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, other_desc);

//     // int64 + uint8 -> int64
//     other_desc = ScalarDesc(value_uint8);
//     test_run({2, 3, 3}, ACL_INT64, ACL_FORMAT_ND, {-10, 10}, other_desc);
//     // int64 + float64 -> int64
//     other_desc = ScalarDesc(value_fp64);
//     test_run_invalid({2, 3, 3}, ACL_INT64, ACL_FORMAT_ND, {-10, 10}, other_desc);

//     // fp16 + uin8 -> fp16
//     other_desc = ScalarDesc(value_uint8);
//     test_run({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, other_desc);
//     // fp16 + int64 -> fp16
//     other_desc = ScalarDesc(3);
//     test_run({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, other_desc);
//     // fp16 + fp64 -> fp16
//     other_desc = ScalarDesc(value_fp64);
//     test_run({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, other_desc);

//     // fp64 + uin8 -> fp64
//     other_desc = ScalarDesc(value_uint8);
//     test_run({2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {-10, 10}, other_desc);
//     // fp64 + int64-> fp64
//     other_desc = ScalarDesc(3);
//     test_run({2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {-10, 10}, other_desc);
//     // fp64 + fp32 -> fp64
//     other_desc = ScalarDesc(3.5f);
//     test_run({2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {-10, 10}, other_desc);
// }

///////////////////////////////////////
/////          检查空指针          /////
///////////////////////////////////////

TEST_F(l2_remainder_inplace_tensor_scalar_test, l2_remainder_inplace_tensor_scalar_test_09)
{
    uint64_t workspaceSize = 0;
    auto other_desc = ScalarDesc(7.86f);
    auto self = TensorDesc({2, 3, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10).Precision(0.00001, 0.00001);

    auto ut = OP_API_UT(aclnnInplaceRemainderTensorScalar, INPUT(nullptr, other_desc), OUTPUT());
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnInplaceRemainderTensorScalar, INPUT(self, nullptr), OUTPUT());
    getWorkspaceResult = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

///////////////////////////////////////
/////         支持空tensor         /////
///////////////////////////////////////

TEST_F(l2_remainder_inplace_tensor_scalar_test, l2_remainder_inplace_tensor_scalar_test_10)
{
    // self emtpy, self.dtype = promoteType
    int32_t value_int32 = 7;
    auto other_desc = ScalarDesc(value_int32);
    test_run({2, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, other_desc);

    // self emtpy, self.dtype != promoteType
    int64_t value_int64 = 7;
    other_desc = ScalarDesc(value_int64);
    test_run({2, 0, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, other_desc);
}

///////////////////////////////////////
/////       支持非连续tensor       /////
///////////////////////////////////////

// TEST_F(l2_remainder_inplace_tensor_scalar_test, l2_remainder_inplace_tensor_scalar_test_11)
// {
//     uint64_t workspaceSize = 0;
//     auto self = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10).Precision(0.00001, 0.00001);
//     auto selfT =
//         TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 10).Precision(0.00001, 0.00001);
//     auto other_desc = ScalarDesc(7);

//     // self not contiguous
//     auto ut = OP_API_UT(aclnnInplaceRemainderTensorScalar, INPUT(selfT, other_desc), OUTPUT());
//     aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
//     // ut.TestPrecision();
// }

// ///////////////////////////////////////
// /////        检查other的值         /////
// ///////////////////////////////////////

// // self为空时，允许other为0。self不为空时，允许other为0
// TEST_F(l2_remainder_inplace_tensor_scalar_test, l2_remainder_inplace_tensor_scalar_test_12)
// {
//     // self为空时，允许other为0
//     auto other_desc = ScalarDesc(0);
//     test_run({2, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, other_desc);

//     // self不为空时，不允许other为0
//     test_run({2, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, other_desc);
// }

// // other的值为负数时，out也为负数
// TEST_F(l2_remainder_inplace_tensor_scalar_test, l2_remainder_inplace_tensor_scalar_test_13)
// {
//     auto other_desc = ScalarDesc(-1.52f);
//     test_run({2, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, other_desc);
// }

// ///////////////////////////////////////
// /////          0维 ~ 8维          /////
// ///////////////////////////////////////

// TEST_F(l2_remainder_inplace_tensor_scalar_test, l2_remainder_inplace_tensor_scalar_test_14)
// {
//     float value = 3.3;
//     auto other_desc = ScalarDesc(value);
//     // 0维
//     test_run({}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, other_desc);
//     // 1维
//     test_run({5}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, other_desc);
//     // 3维
//     test_run({3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, other_desc);
//     // 4维
//     test_run({1, 2, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, other_desc);
//     // 8维 精度概率性有问题
//     test_run({3, 2, 2, 2, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, other_desc);
//     // 9维
//     test_run_invalid({1, 2, 1, 2, 1, 1, 1, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, other_desc);
// }
