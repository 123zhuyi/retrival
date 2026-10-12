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
#include "../../../op_host/op_api/aclnn_triangular_solve.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
using namespace std;

class l2_triangular_solve_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "Triangular Solve Test Setup" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "Triangular Solve Test TearDown" << endl;
    }
};

TEST_F(l2_triangular_solve_test, case_normal)
{
    auto A_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto b_desc =
        TensorDesc({1, 1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    bool upper = true;
    bool transpose = false;
    bool unitriangular = false;

    auto X_desc = TensorDesc(b_desc).Precision(0.0001, 0.0001);
    auto M_desc = TensorDesc(A_desc).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnTriangularSolve, INPUT(b_desc, A_desc, upper, transpose, unitriangular), OUTPUT(X_desc, M_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_triangular_solve_test, case_nullptr)

{
    auto A_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto b_desc =
        TensorDesc({1, 1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    bool upper = true;
    bool transpose = false;
    bool unitriangular = false;

    auto X_desc = TensorDesc(b_desc).Precision(0.0001, 0.0001);
    auto M_desc = TensorDesc(A_desc).Precision(0.0001, 0.0001);

    auto ut1 = OP_API_UT(
        aclnnTriangularSolve, INPUT(nullptr, A_desc, upper, transpose, unitriangular), OUTPUT(X_desc, M_desc));
    uint64_t workspaceSize1 = 0;
    aclnnStatus aclRet1 = ut1.TestGetWorkspaceSize(&workspaceSize1);
    EXPECT_EQ(aclRet1, ACLNN_ERR_INNER_NULLPTR);

    auto ut2 = OP_API_UT(
        aclnnTriangularSolve, INPUT(b_desc, nullptr, upper, transpose, unitriangular), OUTPUT(X_desc, M_desc));
    uint64_t workspaceSize2 = 0;
    aclnnStatus aclRet2 = ut2.TestGetWorkspaceSize(&workspaceSize2);
    EXPECT_EQ(aclRet2, ACLNN_ERR_INNER_NULLPTR);

    auto ut3 = OP_API_UT(
        aclnnTriangularSolve, INPUT(b_desc, A_desc, upper, transpose, unitriangular), OUTPUT(nullptr, M_desc));
    uint64_t workspaceSize3 = 0;
    aclnnStatus aclRet3 = ut3.TestGetWorkspaceSize(&workspaceSize3);
    EXPECT_EQ(aclRet3, ACLNN_ERR_INNER_NULLPTR);

    auto ut4 = OP_API_UT(
        aclnnTriangularSolve, INPUT(b_desc, A_desc, upper, transpose, unitriangular), OUTPUT(X_desc, nullptr));
    uint64_t workspaceSize4 = 0;
    aclnnStatus aclRet4 = ut4.TestGetWorkspaceSize(&workspaceSize4);
    EXPECT_EQ(aclRet4, ACLNN_SUCCESS);
}

TEST_F(l2_triangular_solve_test, case_dtype_valid)
{
    vector<aclDataType> ValidList = {ACL_FLOAT, ACL_DOUBLE, ACL_COMPLEX64, ACL_COMPLEX128, ACL_FLOAT16};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto A_desc =
            TensorDesc({1, 1, 3, 3}, ValidList[i], ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto b_desc = TensorDesc({1, 1, 3, 4}, ValidList[i], ACL_FORMAT_ND)
                          .Value(vector<float>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});

        bool upper = true;
        bool transpose = false;
        bool unitriangular = false;

        auto X_desc = TensorDesc(b_desc).Precision(0.0001, 0.0001);
        auto M_desc = TensorDesc(A_desc).Precision(0.0001, 0.0001);

        auto ut = OP_API_UT(
            aclnnTriangularSolve, INPUT(b_desc, A_desc, upper, transpose, unitriangular), OUTPUT(X_desc, M_desc));
        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        if (ValidList[i] != ACL_FLOAT16) {
            EXPECT_EQ(aclRet, ACLNN_SUCCESS);
        } else {
            EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
        }
    }
}

TEST_F(l2_triangular_solve_test, case_dtype_diff)
{
    auto A_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto b_desc =
        TensorDesc({1, 1, 3, 4}, ACL_DOUBLE, ACL_FORMAT_ND).Value(vector<double>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    bool upper = true;
    bool transpose = false;
    bool unitriangular = false;

    auto X_desc = TensorDesc(b_desc).Precision(0.0001, 0.0001);
    auto M_desc = TensorDesc(A_desc).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnTriangularSolve, INPUT(b_desc, A_desc, upper, transpose, unitriangular), OUTPUT(X_desc, M_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_triangular_solve_test, case_dim_less_2)
{
    auto A_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto b_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    bool upper = true;
    bool transpose = false;
    bool unitriangular = false;

    auto X_desc = TensorDesc(b_desc).Precision(0.0001, 0.0001);
    auto M_desc = TensorDesc(A_desc).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnTriangularSolve, INPUT(b_desc, A_desc, upper, transpose, unitriangular), OUTPUT(X_desc, M_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_triangular_solve_test, case_dim_more_8)
{
    auto A_desc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                      .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto b_desc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                      .Value(vector<float>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    bool upper = true;
    bool transpose = false;
    bool unitriangular = false;

    auto X_desc = TensorDesc(b_desc).Precision(0.0001, 0.0001);
    auto M_desc = TensorDesc(A_desc).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnTriangularSolve, INPUT(b_desc, A_desc, upper, transpose, unitriangular), OUTPUT(X_desc, M_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_triangular_solve_test, case_a_square)
{
    auto A_desc = TensorDesc({1, 1, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto b_desc =
        TensorDesc({1, 1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    bool upper = true;
    bool transpose = false;
    bool unitriangular = false;

    auto X_desc = TensorDesc(b_desc).Precision(0.0001, 0.0001);
    auto M_desc = TensorDesc(A_desc).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnTriangularSolve, INPUT(b_desc, A_desc, upper, transpose, unitriangular), OUTPUT(X_desc, M_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_triangular_solve_test, case_matrix_shape)
{
    auto A_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto b_desc = TensorDesc({1, 1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 2, 2, 2, 2, 2, 2});
    bool upper = true;
    bool transpose = false;
    bool unitriangular = false;

    auto X_desc = TensorDesc(b_desc).Precision(0.0001, 0.0001);
    auto M_desc = TensorDesc(A_desc).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnTriangularSolve, INPUT(b_desc, A_desc, upper, transpose, unitriangular), OUTPUT(X_desc, M_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_triangular_solve_test, case_shape_boardcast_fail)
{
    auto A_desc = TensorDesc({1, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto b_desc =
        TensorDesc({1, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    bool upper = true;
    bool transpose = false;
    bool unitriangular = false;

    auto X_desc = TensorDesc(b_desc).Precision(0.0001, 0.0001);
    auto M_desc = TensorDesc(A_desc).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnTriangularSolve, INPUT(b_desc, A_desc, upper, transpose, unitriangular), OUTPUT(X_desc, M_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_triangular_solve_test, case_shape_boardcast_succ)
{
    auto A_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto b_desc =
        TensorDesc({1, 1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    bool upper = true;
    bool transpose = false;
    bool unitriangular = false;

    auto X_desc = TensorDesc({1, 1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto M_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnTriangularSolve, INPUT(b_desc, A_desc, upper, transpose, unitriangular), OUTPUT(X_desc, M_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_triangular_solve_test, case_shape_boardcast_out_fail)
{
    auto A_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto b_desc =
        TensorDesc({1, 1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    bool upper = true;
    bool transpose = false;
    bool unitriangular = false;

    auto X_desc = TensorDesc(b_desc).Precision(0.0001, 0.0001);
    auto M_desc = TensorDesc(A_desc).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnTriangularSolve, INPUT(b_desc, A_desc, upper, transpose, unitriangular), OUTPUT(X_desc, M_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_triangular_solve_test, case_empty)
{
    auto A_desc = TensorDesc({1, 0, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto b_desc =
        TensorDesc({1, 1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    bool upper = true;
    bool transpose = false;
    bool unitriangular = false;

    auto X_desc = TensorDesc({1, 0, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto M_desc = TensorDesc({1, 0, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnTriangularSolve, INPUT(b_desc, A_desc, upper, transpose, unitriangular), OUTPUT(X_desc, M_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_triangular_solve_test, case_transpose_true)
{
    auto A_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto b_desc =
        TensorDesc({1, 1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    bool upper = true;
    bool transpose = true;
    bool unitriangular = false;

    auto X_desc = TensorDesc(b_desc).Precision(0.0001, 0.0001);
    auto M_desc = TensorDesc(A_desc).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnTriangularSolve, INPUT(b_desc, A_desc, upper, transpose, unitriangular), OUTPUT(X_desc, M_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_triangular_solve_test, case_unitriangular_true)
{
    auto A_desc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto b_desc =
        TensorDesc({1, 1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    bool upper = true;
    bool transpose = false;
    bool unitriangular = true;

    auto X_desc = TensorDesc(b_desc).Precision(0.0001, 0.0001);
    auto M_desc = TensorDesc(A_desc).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnTriangularSolve, INPUT(b_desc, A_desc, upper, transpose, unitriangular), OUTPUT(X_desc, M_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_triangular_solve_test, case_unitriangular_faile)
{
    auto A_desc = TensorDesc({1, 1, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto b_desc =
        TensorDesc({1, 1, 3, 4}, ACL_DOUBLE, ACL_FORMAT_ND).Value(vector<float>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    bool upper = true;
    bool transpose = false;
    bool unitriangular = true;

    auto X_desc = TensorDesc(b_desc).Precision(0.0001, 0.0001);
    auto M_desc = TensorDesc(A_desc).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnTriangularSolve, INPUT(b_desc, A_desc, upper, transpose, unitriangular), OUTPUT(X_desc, M_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
