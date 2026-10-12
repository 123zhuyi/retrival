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

class TileTest : public testing::Test {
  protected:
    static void SetUpTestCase() {
        std::cout << "TileTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "TileTest TearDown" << std::endl;
    }
};

TEST_F(TileTest, tile_infershape_test_0) {
    std::vector<int32_t> values = {8, 9};
    gert::InfershapeContextPara infershapeContextPara("Tile",
                                                      {
                                                        {{{2, 1}, {2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, values.data()}
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 9}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TileTest, tile_infershape_test_1) {
    std::vector<int32_t> values = {9};
    gert::InfershapeContextPara infershapeContextPara("Tile",
                                                      {
                                                        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, values.data()}
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{9}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TileTest, tile_infershape_test_2) {
    std::vector<int32_t> values = {9};
    gert::InfershapeContextPara infershapeContextPara("Tile",
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, values.data()}
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{9}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TileTest, tile_infershape_test_3) {
    std::vector<int32_t> values = {};
    gert::InfershapeContextPara infershapeContextPara("Tile",
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true, values.data()}
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TileTest, tile_infershape_test_4) {
    std::vector<int32_t> values = {0};
    gert::InfershapeContextPara infershapeContextPara("Tile",
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, values.data()}
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: tile with INT64 dtype multiples, expect infershape to succeed and produce correct output shape
TEST_F(TileTest, tile_infershape_int64_multiples)
{
    std::vector<int64_t> values = {8, 9};
    gert::InfershapeContextPara infershapeContextPara("Tile",
                                                      {{{{2, 1}, {2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, values.data()}},
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 9}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: tile where input has fewer dimensions than multiples, expect infershape to pad input with 1s and
// produce correct output
TEST_F(TileTest, tile_infershape_input_fewer_dims_than_multiples)
{
    std::vector<int32_t> values = {2, 3};
    gert::InfershapeContextPara infershapeContextPara(
        "Tile",
        {{{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, values.data()}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    // input {2} padded to {1, 2}, multiples {2, 3}, output = {1*2, 2*3} = {2, 6}
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: tile with invalid dtype (DT_FLOAT) for multiples tensor, expect infershape to fail
TEST_F(TileTest, tile_infershape_invalid_dtype)
{
    std::vector<float> values = {2.0f, 3.0f};
    gert::InfershapeContextPara infershapeContextPara("Tile",
                                                      {{{{2, 1}, {2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND, true, values.data()}},
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// Test scenario: tile with INT64 dtype multiples and fewer input dims, expect infershape to succeed with padded input
// shape producing correct output
TEST_F(TileTest, tile_infershape_int64_fewer_dims)
{
    std::vector<int64_t> values = {2, 3};
    gert::InfershapeContextPara infershapeContextPara(
        "Tile",
        {{{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, values.data()}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    // input {2} padded to {1, 2}, multiples {2, 3}, output = {1*2, 2*3} = {2, 6}
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: tile with negative dimension in input shape, expect infershape to fail as negative dims are illegal
TEST_F(TileTest, tile_infershape_negative_dim)
{
    std::vector<int32_t> values = {2, 1};
    gert::InfershapeContextPara infershapeContextPara("Tile",
                                                      {{{{-1, 3}, {-1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, values.data()}},
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// Test scenario: tile with multiples len exceeding MAXDIMNUM (8), expect infershape to fail
TEST_F(TileTest, tile_infershape_multiples_exceed_max_dim)
{
    std::vector<int32_t> values = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    gert::InfershapeContextPara infershapeContextPara(
        "Tile",
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{9}, {9}}, ge::DT_INT32, ge::FORMAT_ND, true, values.data()}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
