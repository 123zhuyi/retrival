/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class ProdEnvMatACalcRijInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ProdEnvMatACalcRijInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ProdEnvMatACalcRijInfershape TearDown" << std::endl; }
};

// natoms is a compile-time constant, so shapes resolve concretely:
//   nsample = coord dim0 = 1; nloc = natoms[0] = 2; nnei = sum(sel_a) = 2.
//   rij = [nsample, nloc*nnei*3] = [1, 12]; the other five = [nsample, nloc*nnei] = [1, 4].
TEST_F(ProdEnvMatACalcRijInfershape, prod_env_mat_a_calc_rij_infershape_static)
{
    std::vector<int32_t> natomsValues = {2, 2, 2};
    gert::InfershapeContextPara infershapeContextPara(
        "ProdEnvMatACalcRij",
        {
            {{{1, 6}, {1, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, natomsValues.data()},
            {{{1, 9}, {1, 9}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2053}, {2053}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rcut_a", Ops::Math::AnyValue::CreateFrom<float>(1.0f)},
            {"rcut_r", Ops::Math::AnyValue::CreateFrom<float>(2.0f)},
            {"rcut_r_smth", Ops::Math::AnyValue::CreateFrom<float>(1.0f)},
            {"sel_a", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2})},
            {"sel_r", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 12}, {1, 4}, {1, 4}, {1, 4}, {1, 4}, {1, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// coord shape is unknown, so every output falls back to a dynamic 2D shape.
TEST_F(ProdEnvMatACalcRijInfershape, prod_env_mat_a_calc_rij_infershape_dynamic_coord)
{
    std::vector<int32_t> natomsValues = {2, 2, 2};
    gert::InfershapeContextPara infershapeContextPara(
        "ProdEnvMatACalcRij",
        {
            {{{-1, 6}, {-1, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, natomsValues.data()},
            {{{1, 9}, {1, 9}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2053}, {2053}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rcut_a", Ops::Math::AnyValue::CreateFrom<float>(1.0f)},
            {"rcut_r", Ops::Math::AnyValue::CreateFrom<float>(2.0f)},
            {"rcut_r_smth", Ops::Math::AnyValue::CreateFrom<float>(1.0f)},
            {"sel_a", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2})},
            {"sel_r", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// natoms is not a compile-time constant, so the concrete size cannot be inferred: fall back to dynamic 2D shapes.
TEST_F(ProdEnvMatACalcRijInfershape, prod_env_mat_a_calc_rij_infershape_natoms_not_const)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ProdEnvMatACalcRij",
        {
            {{{1, 6}, {1, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 9}, {1, 9}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2053}, {2053}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rcut_a", Ops::Math::AnyValue::CreateFrom<float>(1.0f)},
            {"rcut_r", Ops::Math::AnyValue::CreateFrom<float>(2.0f)},
            {"rcut_r_smth", Ops::Math::AnyValue::CreateFrom<float>(1.0f)},
            {"sel_a", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2})},
            {"sel_r", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
