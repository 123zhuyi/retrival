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

class BatchToSpaceNDInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BatchToSpaceNDInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BatchToSpaceNDInfershapeTest TearDown" << std::endl; }
};

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_basic_4d)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{4, 1, 1, 1}, {4, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 2, 2, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_unknown_rank)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_unknown_shape_4d)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{-1, -1, -1, -1}, {-1, -1, -1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1, -1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_unknown_shape_spatial_remain)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{4, -1, -1, -1}, {4, -1, -1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, -1, -1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_unknown_shape_with_crops)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {1, 1, 1, 1};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{8, -1, 6, 6}, {8, -1, 6, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, -1, 10, 6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_unknown_batch_dim)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{-1, 1, 4, 4}, {-1, 1, 4, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 2, 8, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_unknown_spatial_dim)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{4, 1, -1, -1}, {4, 1, -1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 2, -1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_unknown_rank_5d)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{-1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1, -1, -1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_unknown_shape_partial)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{16, 3, -1, 10}, {16, 3, -1, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 6, -1, 10}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_unknown_shape_with_asymmetric_crops)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {2, 1, 0, 3};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{8, -1, -1, -1}, {8, -1, -1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, -1, -1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_4d_with_crop)
{
    std::vector<int32_t> blockShapeValues = {4, 4};
    std::vector<int32_t> cropsValues = {1, 1, 2, 1};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{16, 1, 8, 8}, {16, 1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 2, 29, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_5d)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{8, 1, 2, 2, 2}, {8, 1, 2, 2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 2, 4, 2, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_large_batch)
{
    std::vector<int32_t> blockShapeValues = {4, 4};
    std::vector<int32_t> cropsValues = {1, 2, 2, 3};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{32, 3, 10, 10}, {32, 3, 10, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 9, 35, 10}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_block_shape_1)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{4, 1, 4, 4}, {4, 1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 2, 8, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_asymmetric_crops)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {1, 0, 0, 1};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{8, 1, 6, 6}, {8, 1, 6, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 1, 11, 6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_different_dtypes)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{4, 1, 2, 2}, {4, 1, 2, 2}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 2, 4, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_output_dim_zero_height)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {1, 1, 0, 0};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{4, 1, 1, 4}, {4, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 0, 2, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_output_dim_zero_width)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {0, 0, 3, 5};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{4, 1, 4, 1}, {4, 1, 4, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 2, 0, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_output_dim_zero_both_spatial)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {1, 1, 1, 1};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{4, 1, 1, 1}, {4, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 0, 0, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_output_dim_zero_all_spatial)
{
    std::vector<int32_t> blockShapeValues = {2, 2, 1};
    std::vector<int32_t> cropsValues = {1, 1, 2, 0, 0, 1};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{4, 1, 1, 1}, {4, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 0, 0, 0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_output_dim_zero_with_large_crops)
{
    std::vector<int32_t> blockShapeValues = {4, 4};
    std::vector<int32_t> cropsValues = {2, 1, 3, 3};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{16, 1, 2, 2}, {16, 1, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 2, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_output_dim_zero_5d)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {1, 1, 0, 0};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{8, 1, 1, 2, 2}, {8, 1, 1, 2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 0, 2, 2, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_output_dim_zero_asymmetric_crops)
{
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> cropsValues = {2, 0, 0, 2};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{4, 1, 2, 2}, {4, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 0, 2, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_block_shape_zero_first_dim)
{
    std::vector<int32_t> blockShapeValues = {0, 2};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{4, 1, 2, 2}, {4, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_block_shape_zero_both_dims)
{
    std::vector<int32_t> blockShapeValues = {0, 0};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{4, 1, 2, 2}, {4, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// Block_shape with float dtype (unsupported for integer const data); expect GRAPH_FAILED
TEST_F(BatchToSpaceNDInfershapeTest, batch_to_space_nd_init_fail_block_shape_dtype)
{
    std::vector<float> blockShapeValues = {2.0f, 2.0f};
    std::vector<int32_t> cropsValues = {0, 0, 0, 0};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpaceND",
        {
            {{{4, 1, 2, 2}, {4, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, cropsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
