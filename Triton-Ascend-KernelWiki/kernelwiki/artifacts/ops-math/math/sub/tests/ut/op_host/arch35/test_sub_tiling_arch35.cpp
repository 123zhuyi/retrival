/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * /file test_sub_tiling_arch35.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/arch35/sub_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class SubTilingTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "SubTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SubTilingTest TearDown" << std::endl;
    }
};

TEST_F(SubTilingTest, sub_test_0)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
    },
    {
        {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SubTilingTest, sub_test_1)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{1, 9, 6, 1, 7, 10}, {1, 9, 6, 1, 7, 10}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{13, 1, 6, 2, 7, 1}, {13, 1, 6, 2, 7, 1}}, ge::DT_BF16, ge::FORMAT_ND},
    },
    {
        {{{13, 9, 6, 2, 7, 10}, {13, 9, 6, 2, 7, 10}}, ge::DT_BF16, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SubTilingTest, sub_test_2)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{8, 11, 12, 14, 6}, {8, 11, 12, 14, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{8, 11, 12, 14, 6}, {8, 11, 12, 14, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {{{8, 11, 12, 14, 6}, {8, 11, 12, 14, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SubTilingTest, sub_test_3)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{20, 5, 1, 9, 17, 10, 3}, {20, 5, 1, 9, 17, 10, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{1, 1, 3, 9, 1, 10, 1}, {1, 1, 3, 9, 1, 10, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {{{20, 5, 3, 9, 17, 10, 3}, {20, 5, 3, 9, 17, 10, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SubTilingTest, sub_test_4)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{15, 5}, {15, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1, 5}, {1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{15, 5}, {15, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SubTilingTest, sub_test_5)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{1, 3, 1, 7, 1, 5, 1, 1}, {1, 3, 1, 7, 1, 5, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{3, 3, 3, 7, 8, 5, 10, 1}, {3, 3, 3, 7, 8, 5, 10, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{3, 3, 3, 7, 8, 5, 10, 1}, {3, 3, 3, 7, 8, 5, 10, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SubTilingTest, sub_test_6)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{1, 1, 1}, {1, 1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{15, 2, 1}, {15, 2, 1}}, ge::DT_INT32, ge::FORMAT_ND},
    },
    {
        {{{15, 2, 1}, {15, 2, 1}}, ge::DT_INT32, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SubTilingTest, sub_test_7)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{16, 5, 4, 6, 2, 2, 17}, {16, 5, 4, 6, 2, 2, 17}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{16, 5, 1, 6, 1, 2, 1}, {16, 5, 1, 6, 1, 2, 1}}, ge::DT_INT32, ge::FORMAT_ND},
    },
    {
        {{{16, 5, 4, 6, 2, 2, 17}, {16, 5, 4, 6, 2, 2, 17}}, ge::DT_INT32, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SubTilingTest, sub_test_8)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{4, 14, 6, 1}, {4, 14, 6, 1}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_INT64, ge::FORMAT_ND},
    },
    {
        {{{4, 14, 6, 4}, {4, 14, 6, 4}}, ge::DT_INT64, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SubTilingTest, sub_test_9)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{7, 1, 6, 1, 14, 1, 2}, {7, 1, 6, 1, 14, 1, 2}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{1, 20, 6, 4, 1, 4, 2}, {1, 20, 6, 4, 1, 4, 2}}, ge::DT_INT64, ge::FORMAT_ND},
    },
    {
        {{{7, 20, 6, 4, 14, 4, 2}, {7, 20, 6, 4, 14, 4, 2}}, ge::DT_INT64, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SubTilingTest, sub_test_10)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_UINT8, ge::FORMAT_ND},
        {{{9, 3, 9, 16}, {9, 3, 9, 16}}, ge::DT_UINT8, ge::FORMAT_ND},
    },
    {
        {{{9, 3, 9, 16}, {9, 3, 9, 16}}, ge::DT_UINT8, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SubTilingTest, sub_test_11)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{1, 1, 4, 1, 3, 3, 12, 1}, {1, 1, 4, 1, 3, 3, 12, 1}}, ge::DT_UINT8, ge::FORMAT_ND},
        {{{16, 10, 1, 3, 3, 1, 1, 6}, {16, 10, 1, 3, 3, 1, 1, 6}}, ge::DT_UINT8, ge::FORMAT_ND},
    },
    {
        {{{16, 10, 4, 3, 3, 3, 12, 6}, {16, 10, 4, 3, 3, 3, 12, 6}}, ge::DT_UINT8, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SubTilingTest, sub_test_12)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{8, 16, 17, 10}, {8, 16, 17, 10}}, ge::DT_INT8, ge::FORMAT_ND},
    },
    {
        {{{8, 16, 17, 10}, {8, 16, 17, 10}}, ge::DT_INT8, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SubTilingTest, sub_test_13)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{1, 20, 2, 1, 5, 5, 1, 11}, {1, 20, 2, 1, 5, 5, 1, 11}}, ge::DT_INT8, ge::FORMAT_ND},
        {{{7, 20, 1, 5, 5, 5, 4, 11}, {7, 20, 1, 5, 5, 5, 4, 11}}, ge::DT_INT8, ge::FORMAT_ND},
    },
    {
        {{{7, 20, 2, 5, 5, 5, 4, 11}, {7, 20, 2, 5, 5, 5, 4, 11}}, ge::DT_INT8, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SubTilingTest, sub_test_14)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{16, 7, 14}, {16, 7, 14}}, ge::DT_BOOL, ge::FORMAT_ND},
        {{{16, 7, 14}, {16, 7, 14}}, ge::DT_BOOL, ge::FORMAT_ND},
    },
    {
        {{{16, 7, 14}, {16, 7, 14}}, ge::DT_BOOL, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SubTilingTest, sub_test_15)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{12, 9, 3, 18, 1, 1, 4, 10}, {12, 9, 3, 18, 1, 1, 4, 10}}, ge::DT_BOOL, ge::FORMAT_ND},
        {{{12, 1, 3, 1, 1, 1, 4, 1}, {12, 1, 3, 1, 1, 1, 4, 1}}, ge::DT_BOOL, ge::FORMAT_ND},
    },
    {
        {{{12, 9, 3, 18, 1, 1, 4, 10}, {12, 9, 3, 18, 1, 1, 4, 10}}, ge::DT_BOOL, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SubTilingTest, sub_test_16)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{16, 7, 14}, {16, 7, 14}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
        {{{16, 7, 14}, {16, 7, 14}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
    },
    {
        {{{16, 7, 14}, {16, 7, 14}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SubTilingTest, sub_test_17)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{12, 9, 3, 18, 1, 1, 4, 10}, {12, 9, 3, 18, 1, 1, 4, 10}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
        {{{12, 1, 3, 1, 1, 1, 4, 1}, {12, 1, 3, 1, 1, 1, 4, 1}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
    },
    {
        {{{12, 9, 3, 18, 1, 1, 4, 10}, {12, 9, 3, 18, 1, 1, 4, 10}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SubTilingTest, sub_test_18)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{16, 7, 14}, {16, 7, 14}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
        {{{16, 7, 14}, {16, 7, 14}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
    },
    {
        {{{16, 7, 14}, {16, 7, 14}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SubTilingTest, sub_test_19)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{12, 9, 3, 18, 1, 1, 4, 10}, {12, 9, 3, 18, 1, 1, 4, 10}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
        {{{12, 1, 3, 1, 1, 1, 4, 1}, {12, 1, 3, 1, 1, 1, 4, 1}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
    },
    {
        {{{12, 9, 3, 18, 1, 1, 4, 10}, {12, 9, 3, 18, 1, 1, 4, 10}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SubTilingTest, sub_test_failed_0)
{
    optiling::SubCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Sub",
    {
        {{{17, 15, 8}, {17, 15, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{17, 1, 1}, {17, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
    },
    {
        {{{17, 15, 8}, {17, 15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
    }, &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}
