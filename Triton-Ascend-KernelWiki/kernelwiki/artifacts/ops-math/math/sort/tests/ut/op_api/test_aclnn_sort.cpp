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

#include "../../../op_api/aclnn_sort.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_sort_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "sort_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "sort_test TearDown" << endl; }

    vector<int64_t> GetViewDims(int num_of_dim)
    {
        vector<int64_t> view_dims = {1, 2, 2, 2}; // NCHW + NHWC + HWCN
        if (num_of_dim == 5)                      // NDHWC + NCDHW
            view_dims = {1, 4, 1, 1, 2};
        else if (num_of_dim == 0) // 空tensor
            view_dims = {0};
        else if (num_of_dim == 1) // 1维
            view_dims = {10};
        else if (num_of_dim == 8) // 8维
            view_dims = {2, 1, 2, 1, 2, 4, 5, 7};
        else if (num_of_dim == 10) // > 8维
            view_dims = {2, 1, 4, 4, 3, 1, 2, 4, 5, 7};
        else if (num_of_dim != 4) // ND
            view_dims = {1, 4, 1, 1, 2, 1};

        return view_dims;
    }

    vector<float> tensor_value = {1.3232, 2.123, 3.234, -234.4, 23.52, -1.6, 71.8, 823};

    // num_of_dim: 这个tensor是几维的
    void test_run(aclDataType test_dtype, aclFormat test_format, int num_of_dim, bool stable = false, int64_t dim = 1,
                  bool descending = false)
    {
        auto view_dims = GetViewDims(num_of_dim);
        const vector<int64_t>& view_dims_final = const_cast<vector<int64_t>&>(view_dims);

        auto self = TensorDesc(view_dims_final, test_dtype, test_format).ValueRange(-2, 2).Value(tensor_value);
        auto values = TensorDesc(view_dims_final, test_dtype, test_format).Precision(0.00001, 0.00001);
        auto indices = TensorDesc(view_dims_final, ACL_INT64, test_format).Precision(0.00001, 0.00001);

        auto ut = OP_API_UT(aclnnSort, INPUT(self, stable, dim, descending), OUTPUT(values, indices));

        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
        // ut.TestPrecision();
    }

    void test_run_invalid(aclDataType test_dtype, aclFormat test_format, int num_of_dim, bool stable = false,
                          int64_t dim = 1, bool descending = false)
    {
        auto view_dims = GetViewDims(num_of_dim);
        const vector<int64_t>& view_dims_final = const_cast<vector<int64_t>&>(view_dims);

        auto self = TensorDesc(view_dims_final, test_dtype, test_format)
                        .ValueRange(-2, 2)
                        .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
        auto values = TensorDesc(view_dims_final, test_dtype, test_format);
        auto indices = TensorDesc(view_dims_final, ACL_INT64, test_format);

        auto ut = OP_API_UT(aclnnSort, INPUT(self, stable, dim, descending), OUTPUT(values, indices));

        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }

    void test_run_dtype_not_consistent(aclDataType self_dtype, aclDataType values_dtype, aclDataType indices_dtype,
                                       aclFormat test_format, int num_of_dim, bool stable = false, int64_t dim = 1,
                                       bool descending = false, bool invalid = true)
    {
        auto view_dims = GetViewDims(num_of_dim);
        const vector<int64_t>& view_dims_final = const_cast<vector<int64_t>&>(view_dims);

        auto self = TensorDesc(view_dims_final, self_dtype, test_format)
                        .ValueRange(-2, 2)
                        .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
        auto values = TensorDesc(view_dims_final, values_dtype, test_format);
        auto indices = TensorDesc(view_dims_final, indices_dtype, test_format);

        auto ut = OP_API_UT(aclnnSort, INPUT(self, stable, dim, descending), OUTPUT(values, indices));

        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        if (invalid) {
            EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
        } else {
            EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
            // ut.TestPrecision();
        }
    }

    void test_run_format_not_consistent(aclDataType test_dtype, aclFormat self_format, aclFormat values_format,
                                        aclFormat indices_format, int num_of_dim, bool stable = false, int64_t dim = 1,
                                        bool descending = false)
    {
        auto view_dims = GetViewDims(num_of_dim);
        const vector<int64_t>& view_dims_final = const_cast<vector<int64_t>&>(view_dims);

        auto self = TensorDesc(view_dims_final, test_dtype, self_format)
                        .ValueRange(-2, 2)
                        .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
        auto values = TensorDesc(view_dims_final, test_dtype, values_format);
        auto indices = TensorDesc(view_dims_final, ACL_INT64, indices_format);

        auto ut = OP_API_UT(aclnnSort, INPUT(self, stable, dim, descending), OUTPUT(values, indices));

        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }

    void test_run_nullptr(aclDataType test_dtype, aclFormat test_format, int num_of_dim)
    {
        int64_t dim = 1;
        bool stable = false;
        bool descending = false;

        auto view_dims = GetViewDims(num_of_dim);
        const vector<int64_t>& view_dims_final = const_cast<vector<int64_t>&>(view_dims);

        auto self = TensorDesc(view_dims_final, test_dtype, test_format)
                        .ValueRange(-2, 2)
                        .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
        auto values = TensorDesc(view_dims_final, test_dtype, test_format);
        auto indices = TensorDesc(view_dims_final, ACL_INT64, test_format);

        uint64_t workspaceSize = 0;

        auto ut1 = OP_API_UT(aclnnSort, INPUT(nullptr, stable, dim, descending), OUTPUT(values, indices));
        aclnnStatus getWorkspaceResult1 = ut1.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult1, ACLNN_ERR_PARAM_NULLPTR);

        auto ut2 = OP_API_UT(aclnnSort, INPUT(self, stable, dim, descending), OUTPUT(nullptr, indices));
        aclnnStatus getWorkspaceResult2 = ut2.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult2, ACLNN_ERR_PARAM_NULLPTR);

        auto ut3 = OP_API_UT(aclnnSort, INPUT(self, stable, dim, descending), OUTPUT(values, nullptr));
        aclnnStatus getWorkspaceResult3 = ut3.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult3, ACLNN_ERR_PARAM_NULLPTR);

        auto ut4 = OP_API_UT(aclnnSort, INPUT(nullptr, stable, dim, descending), OUTPUT(nullptr, indices));
        aclnnStatus getWorkspaceResult4 = ut4.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult4, ACLNN_ERR_PARAM_NULLPTR);

        auto ut5 = OP_API_UT(aclnnSort, INPUT(nullptr, stable, dim, descending), OUTPUT(values, nullptr));
        aclnnStatus getWorkspaceResult5 = ut5.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult5, ACLNN_ERR_PARAM_NULLPTR);

        auto ut6 = OP_API_UT(aclnnSort, INPUT(self, stable, dim, descending), OUTPUT(nullptr, nullptr));
        aclnnStatus getWorkspaceResult6 = ut6.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult6, ACLNN_ERR_PARAM_NULLPTR);

        auto ut7 = OP_API_UT(aclnnSort, INPUT((aclTensor*)nullptr, stable, dim, descending),
                             OUTPUT((aclTensor*)nullptr, (aclTensor*)nullptr));
        aclnnStatus getWorkspaceResult7 = ut7.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult7, ACLNN_ERR_PARAM_NULLPTR);
    }

    void test_run_indices_longTensor(aclDataType test_dtype, aclFormat test_format, int num_of_dim)
    {
        int64_t dim = 1;
        bool stable = false;
        bool descending = false;

        auto view_dims = GetViewDims(num_of_dim);
        const vector<int64_t>& view_dims_final = const_cast<vector<int64_t>&>(view_dims);

        auto self = TensorDesc(view_dims_final, test_dtype, test_format)
                        .ValueRange(-2, 2)
                        .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
        auto values = TensorDesc(view_dims_final, test_dtype, test_format).Precision(0.00001, 0.00001);
        auto indices = TensorDesc(view_dims_final, ACL_INT64, test_format).Precision(0.00001, 0.00001);

        auto ut = OP_API_UT(aclnnSort, INPUT(self, stable, dim, descending), OUTPUT(values, indices));
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

        auto ptr = indices.ToAclType();
        // EXPECT_EQ(ptr->GetDataType(), ACL_INT64);
        // ut.TestPrecision();
    }

    void test_run_new(aclDataType test_dtype, vector<int64_t> view_dims, int dim = -1, bool stable = false,
                      bool descending = false)
    {
        const vector<int64_t>& view_dims_final = const_cast<vector<int64_t>&>(view_dims);

        auto self = TensorDesc(view_dims_final, test_dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
        auto values = TensorDesc(view_dims_final, test_dtype, ACL_FORMAT_ND);
        auto indices = TensorDesc(view_dims_final, ACL_INT64, ACL_FORMAT_ND);

        auto ut = OP_API_UT(aclnnSort, INPUT(self, stable, dim, descending), OUTPUT(values, indices));

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        // ut.TestPrecision();
    }

    void test_run_aicpu_stable(aclDataType test_dtype, vector<int64_t> view_dims, int dim = -1, bool stable = false,
                               bool descending = false)
    {
        const vector<int64_t>& view_dims_final = const_cast<vector<int64_t>&>(view_dims);

        auto self = TensorDesc(view_dims_final, test_dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
        auto values = TensorDesc(view_dims_final, test_dtype, ACL_FORMAT_ND);
        auto indices = TensorDesc(view_dims_final, ACL_INT64, ACL_FORMAT_ND);

        auto ut = OP_API_UT(aclnnSort, INPUT(self, stable, dim, descending), OUTPUT(values, indices));

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        // ut.TestPrecision();
    }
};

///////////////////////////////////////
/////         检查输出种类         /////
///////////////////////////////////////

// sort输入支持float16   aicore
TEST_F(l2_sort_test, l2_sort_test_01) { test_run(ACL_FLOAT16, ACL_FORMAT_ND, 6); }

// sort输入支持float32   1980 aicpu, 1971 aicore
TEST_F(l2_sort_test, l2_sort_test_02) { test_run(ACL_FLOAT, ACL_FORMAT_NCHW, 4); }

// sort输入支持int8   aicpu
TEST_F(l2_sort_test, l2_sort_test_03) { test_run(ACL_INT8, ACL_FORMAT_NHWC, 4); }

// sort输入支持int16  aicpu
TEST_F(l2_sort_test, l2_sort_test_04) { test_run(ACL_INT16, ACL_FORMAT_HWCN, 4); }

// sort输入支持int32  aicpu
TEST_F(l2_sort_test, l2_sort_test_05) { test_run(ACL_INT32, ACL_FORMAT_NDHWC, 5); }

// sort输入支持int64   aicpu
TEST_F(l2_sort_test, l2_sort_test_06) { test_run(ACL_INT64, ACL_FORMAT_NCDHW, 5); }

// sort输入支持uint8   aicpu
TEST_F(l2_sort_test, l2_sort_test_07) { test_run(ACL_UINT8, ACL_FORMAT_ND, 6); }

// // NPU sort输入不支持bool, bfloat16, double, complex64, complex128
// TEST_F(l2_sort_test, l2_sort_test_08)
// {
//     test_run_invalid(ACL_BOOL, ACL_FORMAT_ND, 6);
//     test_run_invalid(ACL_BF16,   ACL_FORMAT_ND, 6);
//     // test_run_invalid(ACL_DOUBLE, ACL_FORMAT_ND, 6);
//     test_run_invalid(ACL_COMPLEX64, ACL_FORMAT_ND, 6);
//     test_run_invalid(ACL_COMPLEX128, ACL_FORMAT_ND, 6);
// }

// tensor的dtype不一致
TEST_F(l2_sort_test, l2_sort_test_09)
{
    // self + values dtype不一致是允许的
    test_run_dtype_not_consistent(ACL_INT8, ACL_INT16, ACL_INT64, ACL_FORMAT_ND, 6, false, 1, false, false);
    test_run_dtype_not_consistent(ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FORMAT_ND, 6); // indices 非INT64
    test_run_dtype_not_consistent(ACL_INT8, ACL_INT8, ACL_INT32, ACL_FORMAT_ND, 6); // indices 非INT64
}

// tensor的dtype是未定义的
TEST_F(l2_sort_test, l2_sort_test_10) { test_run_invalid(ACL_DT_UNDEFINED, ACL_FORMAT_ND, 6); }

///////////////////////////////////////
/////          检查空指针          /////
///////////////////////////////////////

// 不支持空指针 aicore + aicpu
TEST_F(l2_sort_test, l2_sort_test_11)
{
    test_run_nullptr(ACL_FLOAT16, ACL_FORMAT_ND, 4);
    test_run_nullptr(ACL_INT8, ACL_FORMAT_ND, 4);
}

///////////////////////////////////////
////   检查indices为Long Tensor    ////
///////////////////////////////////////

// aicore + aicpu
TEST_F(l2_sort_test, l2_sort_test_12)
{
    test_run_indices_longTensor(ACL_FLOAT16, ACL_FORMAT_ND, 4);
    test_run_indices_longTensor(ACL_FLOAT, ACL_FORMAT_ND, 4);
    test_run_indices_longTensor(ACL_UINT8, ACL_FORMAT_ND, 4);
    test_run_indices_longTensor(ACL_INT8, ACL_FORMAT_ND, 4);
    test_run_indices_longTensor(ACL_INT16, ACL_FORMAT_ND, 4);
    test_run_indices_longTensor(ACL_INT32, ACL_FORMAT_ND, 4);
    test_run_indices_longTensor(ACL_INT64, ACL_FORMAT_ND, 4);
}

///////////////////////////////////////
/////         数据格式覆盖         /////
///////////////////////////////////////

// 支持的数据格式有 ND、NCHW、NHWC、HWCN、NDHWC、NCDHW  aicore + aicpu
TEST_F(l2_sort_test, l2_sort_test_13)
{
    test_run(ACL_FLOAT16, ACL_FORMAT_ND, 6);
    test_run(ACL_FLOAT16, ACL_FORMAT_NCHW, 4);
    test_run(ACL_FLOAT16, ACL_FORMAT_NHWC, 4);
    test_run(ACL_FLOAT16, ACL_FORMAT_HWCN, 4);
    test_run(ACL_FLOAT16, ACL_FORMAT_NDHWC, 5);
    test_run(ACL_FLOAT16, ACL_FORMAT_NCDHW, 5);

    test_run(ACL_INT8, ACL_FORMAT_ND, 6);
    test_run(ACL_INT16, ACL_FORMAT_NCHW, 4);
    test_run(ACL_INT32, ACL_FORMAT_NHWC, 4);
    test_run(ACL_INT64, ACL_FORMAT_HWCN, 4);
    test_run(ACL_UINT8, ACL_FORMAT_NDHWC, 5);
    test_run(ACL_INT16, ACL_FORMAT_NCDHW, 5);
}

///////////////////////////////////////
/////         支持空tensor         /////
///////////////////////////////////////

// 空tensor    走不到aicore和aicpu应该
TEST_F(l2_sort_test, l2_sort_test_16)
{
    // aicore
    auto view_dims = GetViewDims(0);
    const vector<int64_t>& view_dims_final = const_cast<vector<int64_t>&>(view_dims);

    auto self = TensorDesc(view_dims_final, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Value(vector<float>{});
    auto values = TensorDesc(view_dims_final, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indices = TensorDesc(view_dims_final, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSort, INPUT(self, false, -1, false), OUTPUT(values, indices));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();

    // aicpu
    self = TensorDesc(view_dims_final, ACL_INT16, ACL_FORMAT_ND).ValueRange(-2, 2).Value(vector<float>{});
    values = TensorDesc(view_dims_final, ACL_INT16, ACL_FORMAT_ND);
    indices = TensorDesc(view_dims_final, ACL_INT64, ACL_FORMAT_ND);

    ut = OP_API_UT(aclnnSort, INPUT(self, false, -1, false), OUTPUT(values, indices));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

///////////////////////////////////////
/////       支持非连续tensor       /////
///////////////////////////////////////

// 非连续tensor    aicore + aicpu
TEST_F(l2_sort_test, l2_sort_test_17)
{
    vector<int64_t> view_dims = {5, 4};
    const vector<int64_t>& view_dims_final = const_cast<vector<int64_t>&>(view_dims);
    auto self = TensorDesc(view_dims_final, ACL_FLOAT16, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(300, 350);
    auto values = TensorDesc(view_dims_final, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indices = TensorDesc(view_dims_final, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSort, INPUT(self, true, -1, false), OUTPUT(values, indices));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();

    self = TensorDesc(view_dims_final, ACL_INT8, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(300, 350);
    values = TensorDesc(view_dims_final, ACL_INT8, ACL_FORMAT_ND);
    indices = TensorDesc(view_dims_final, ACL_INT64, ACL_FORMAT_ND);

    ut = OP_API_UT(aclnnSort, INPUT(self, false, -1, false), OUTPUT(values, indices));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

///////////////////////////////////////
/////     dim, descending参数      /////
///////////////////////////////////////

// dim为正数/负数/0, descending为True/False时正常运行   aicore + aicpu
TEST_F(l2_sort_test, l2_sort_test_19)
{
    test_run(ACL_FLOAT16, ACL_FORMAT_NCDHW, 5, false, -3, false);
    test_run(ACL_FLOAT16, ACL_FORMAT_NCDHW, 5, false, 0, false);
    test_run(ACL_FLOAT16, ACL_FORMAT_NCDHW, 5, false, 2, false);
    test_run(ACL_FLOAT16, ACL_FORMAT_NCDHW, 5, false, 2, true);

    test_run(ACL_INT8, ACL_FORMAT_NCDHW, 5, false, -3, false);
    test_run(ACL_INT16, ACL_FORMAT_NCDHW, 5, false, 0, false);
    test_run(ACL_UINT8, ACL_FORMAT_NCDHW, 5, false, 2, false);
    test_run(ACL_INT64, ACL_FORMAT_NCDHW, 5, false, 2, true);
}

// dim在范围 [-N, N-1] 范围之外 aicore + aicpu
TEST_F(l2_sort_test, l2_sort_test_20)
{
    test_run(ACL_FLOAT16, ACL_FORMAT_NCHW, 4, false, -4, false); // -N
    test_run(ACL_FLOAT16, ACL_FORMAT_NCHW, 4, false, 3, false);  // N-1
    test_run_invalid(ACL_FLOAT16, ACL_FORMAT_NCHW, 4, false, -5, false);
    test_run_invalid(ACL_FLOAT16, ACL_FORMAT_NCHW, 4, false, 4, false);

    test_run(ACL_INT8, ACL_FORMAT_NCHW, 4, false, -4, false); // -N
    test_run(ACL_INT16, ACL_FORMAT_NCHW, 4, false, 3, false); // N-1
    test_run_invalid(ACL_INT64, ACL_FORMAT_NCHW, 4, false, -5, false);
    test_run_invalid(ACL_UINT8, ACL_FORMAT_NCHW, 4, false, 4, false);
}

///////////////////////////////////////
/////       sort的维度数据量       /////
///////////////////////////////////////

// sort的维度数据量为1    aicore + aicpu
TEST_F(l2_sort_test, l2_sort_test_22)
{
    test_run_new(ACL_FLOAT16, {3, 4, 4, 1}, -1, false, false);
    test_run_new(ACL_INT32, {3, 4, 4, 1}, -1, false, false);
}

// sort的维度数据量为100001  aicore + aicpu
TEST_F(l2_sort_test, l2_sort_test_23)
{
    // test_run_new(ACL_FLOAT16, {3, 4, 4, 100001}, -1, true, false);   // values正常, indices正常. st通过
    // test_run_new(ACL_INT64, {3, 4, 4, 100001}, -1, false, false);      // values正常, indices正常. st通过
}

///////////////////////////////////////
/////       sort的维度数           /////
///////////////////////////////////////

// 维度为1    aicore + aicpu
TEST_F(l2_sort_test, l2_sort_test_24)
{
    test_run_new(ACL_FLOAT16, {10}, -1, true, true);
    test_run_new(ACL_INT32, {10}, -1, false, false);
}

// 维度为10， tranpose分支    aicore + aicpu
TEST_F(l2_sort_test, l2_sort_test_25)
{
    test_run_new(ACL_FLOAT16, {2, 1, 4, 4, 3, 1, 2, 4, 5, 7}, 4, true, true);
    test_run_new(ACL_INT32, {2, 1, 4, 4, 3, 1, 2, 4, 5, 7}, 4, false, true);
}

// 维度为10， 非tranpose分支    aicore + aicpu
TEST_F(l2_sort_test, l2_sort_test_26)
{
    test_run_new(ACL_FLOAT16, {2, 1, 4, 4, 3, 1, 2, 4, 5, 7}, -1, true, true);
    test_run_new(ACL_INT32, {2, 1, 4, 4, 3, 1, 2, 4, 5, 7}, -1, false, true);
}

///////////////////////////////////////
/////      sort stable为T / F     /////
///////////////////////////////////////

// stable为True/False   aicore + aicpu
TEST_F(l2_sort_test, l2_sort_test_27)
{
    test_run_new(ACL_FLOAT16, {2, 1, 2, 1, 2, 4, 5, 7}, -1, true, true);  // stable True
    test_run_new(ACL_FLOAT16, {2, 1, 2, 1, 2, 4, 5, 7}, -1, false, true); // stable false
    // aicpu不支持stable为True
    test_run_new(ACL_INT64, {2, 1, 2, 1, 2, 4, 5, 7}, -1, false, true);          // stable false
    test_run_aicpu_stable(ACL_INT64, {2, 1, 2, 1, 2, 4, 5, 7}, -1, true, false); // stable为True时，因为走AICPU所以报错
}

// value相等时的排序
TEST_F(l2_sort_test, l2_sort_test_28)
{
    vector<int64_t> view_dims = GetViewDims(8);
    const vector<int64_t>& view_dims_final = const_cast<vector<int64_t>&>(view_dims);

    auto self = TensorDesc(view_dims_final, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 1);
    auto values = TensorDesc(view_dims_final, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indices = TensorDesc(view_dims_final, ACL_INT64, ACL_FORMAT_ND);

    // aicore
    // stable 为False
    auto ut = OP_API_UT(aclnnSort, INPUT(self, false, -1, false), OUTPUT(values, indices));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();

    // stable 为True
    auto ut2 = OP_API_UT(aclnnSort, INPUT(self, true, -1, false), OUTPUT(values, indices));
    aclnnStatus aclRet2 = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet2, ACL_SUCCESS);
    // ut2.TestPrecision();

    // aicpu
    self = TensorDesc(view_dims_final, ACL_INT16, ACL_FORMAT_ND).ValueRange(1, 1);
    values = TensorDesc(view_dims_final, ACL_INT16, ACL_FORMAT_ND);
    indices = TensorDesc(view_dims_final, ACL_INT64, ACL_FORMAT_ND);

    // stable 为False
    ut = OP_API_UT(aclnnSort, INPUT(self, false, -1, false), OUTPUT(values, indices));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();

    // stable 为True
    ut2 = OP_API_UT(aclnnSort, INPUT(self, true, -1, false), OUTPUT(values, indices));
    aclRet2 = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet2, ACL_SUCCESS);
    // ut2.TestPrecision();
}

///////////////////////////////////////
/////        shape必须一致         /////
///////////////////////////////////////

// shape必须一致
TEST_F(l2_sort_test, l2_sort_test_29)
{
    vector<int64_t> view_dims = {3, 4, 2};
    vector<int64_t> view_dims1 = {3, 2, 4};
    const vector<int64_t>& view_dims_final = const_cast<vector<int64_t>&>(view_dims);
    const vector<int64_t>& view_dims_final1 = const_cast<vector<int64_t>&>(view_dims1);

    // indices ! = self
    auto self = TensorDesc(view_dims_final, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 1);
    auto values = TensorDesc(view_dims_final, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indices = TensorDesc(view_dims_final1, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSort, INPUT(self, false, -1, false), OUTPUT(values, indices));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // values ! = self
    self = TensorDesc(view_dims_final, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 1);
    values = TensorDesc(view_dims_final1, ACL_FLOAT16, ACL_FORMAT_ND);
    indices = TensorDesc(view_dims_final, ACL_INT64, ACL_FORMAT_ND);

    auto ut1 = OP_API_UT(aclnnSort, INPUT(self, false, -1, false), OUTPUT(values, indices));
    aclnnStatus aclRet1 = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet1, ACLNN_ERR_PARAM_INVALID);
}

///////////////////////////////////////
/////   aicpu时支持stable为True  /////
///////////////////////////////////////
TEST_F(l2_sort_test, l2_sort_test_34)
{
    test_run_aicpu_stable(ACL_FLOAT16, {2, 3, 3, 1}, -1, true); // 因为sort为1，走aicpu所以报错
    test_run_aicpu_stable(ACL_INT64, {2, 4, 5, 7}, -1, true);   // dtype走aicpu，所以报错
}

///////////////////////////////////////
/////     支持torch.tensor(a)     /////
///////////////////////////////////////

// torch.tensor(a)能正常计算
TEST_F(l2_sort_test, l2_sort_test_35)
{
    vector<int64_t> view_dims = {};
    const vector<int64_t>& view_dims_final = const_cast<vector<int64_t>&>(view_dims);

    // indices ! = self
    auto self = TensorDesc(view_dims, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(10, 10);
    auto values = TensorDesc(view_dims, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indices = TensorDesc(view_dims, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSort, INPUT(self, false, -1, false), OUTPUT(values, indices));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

TEST_F(l2_sort_test, ascend310B_l2_sort_test_36)
{
    vector<int64_t> view_dims = {20};
    const vector<int64_t>& view_dims_final = const_cast<vector<int64_t>&>(view_dims);

    // indices ! = self
    auto self = TensorDesc(view_dims_final, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto values = TensorDesc(view_dims_final, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indices = TensorDesc(view_dims_final, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSort, INPUT(self, true, -1, false), OUTPUT(values, indices));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
