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
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class PopulationCountInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "PopulationCountInfershapeTest SetUp" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "PopulationCountInfershapeTest TearDown" << std::endl;
    }
};

TEST_F(PopulationCountInfershapeTest, infershape_1d_int16) {
    gert::StorageShape shape = {{128}, {128}};
    gert::InfershapeContextPara infershapeContextPara(
        "PopulationCount",
        {{shape, ge::DT_INT16, ge::FORMAT_ND}},
        {{shape, ge::DT_UINT8, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PopulationCountInfershapeTest, infershape_1d_uint16) {
    gert::StorageShape shape = {{256}, {256}};
    gert::InfershapeContextPara infershapeContextPara(
        "PopulationCount",
        {{shape, ge::DT_UINT16, ge::FORMAT_ND}},
        {{shape, ge::DT_UINT8, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{256}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PopulationCountInfershapeTest, infershape_2d) {
    gert::StorageShape shape = {{32, 64}, {32, 64}};
    gert::InfershapeContextPara infershapeContextPara(
        "PopulationCount",
        {{shape, ge::DT_INT16, ge::FORMAT_ND}},
        {{shape, ge::DT_UINT8, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PopulationCountInfershapeTest, infershape_3d) {
    gert::StorageShape shape = {{4, 8, 16}, {4, 8, 16}};
    gert::InfershapeContextPara infershapeContextPara(
        "PopulationCount",
        {{shape, ge::DT_UINT16, ge::FORMAT_ND}},
        {{shape, ge::DT_UINT8, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 8, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PopulationCountInfershapeTest, infershape_4d) {
    gert::StorageShape shape = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    gert::InfershapeContextPara infershapeContextPara(
        "PopulationCount",
        {{shape, ge::DT_INT16, ge::FORMAT_ND}},
        {{shape, ge::DT_UINT8, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PopulationCountInfershapeTest, infershape_scalar) {
    gert::StorageShape shape = {{}, {}};
    gert::InfershapeContextPara infershapeContextPara(
        "PopulationCount",
        {{shape, ge::DT_INT16, ge::FORMAT_ND}},
        {{shape, ge::DT_UINT8, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PopulationCountInfershapeTest, infershape_large) {
    gert::StorageShape shape = {{1024, 1024}, {1024, 1024}};
    gert::InfershapeContextPara infershapeContextPara(
        "PopulationCount",
        {{shape, ge::DT_UINT16, ge::FORMAT_ND}},
        {{shape, ge::DT_UINT8, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024, 1024}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
