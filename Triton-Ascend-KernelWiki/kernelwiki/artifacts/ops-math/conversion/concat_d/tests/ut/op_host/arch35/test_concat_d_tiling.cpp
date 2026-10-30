/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../../op_host/arch35/concat_d_tiling_arch35.h"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class ConcatDTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ConcatDTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ConcatDTiling TearDown" << std::endl;
    }
};

TEST_F(ConcatDTiling, ConcatD_tiling_UT_one_axis_diff_shape_align)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "ConcatD",
        {
            {{{31, 3, 2560}, {31, 3, 2560}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{31, 3, 1280}, {31, 3, 1280}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{31, 3, 9980}, {31, 3, 9980}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{31, 3, 13820}, {31, 3, 13820}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1}, &compileInfo);
    uint64_t expectTilingKey = 2224;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConcatDTiling, ConcatD_tiling_UT_pure_copy)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "ConcatD",
        {
            {{{310, 3, 2560}, {310, 3, 2560}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{310, 3, 1280}, {310, 3, 1280}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{310, 3, 9980}, {310, 3, 9980}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{310, 3, 13820}, {310, 3, 13820}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1}, &compileInfo);
    uint64_t expectTilingKey = 20002;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConcatDTiling, ConcatD_tiling_UT_one_axis_diff_shape_align_int64)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "ConcatD",
        {
            {{{16384, 512}, {16384, 512}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{16384, 4}, {16384, 4}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{16384, 516}, {16384, 516}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        },
        {2}, {1}, &compileInfo);
    uint64_t expectTilingKey = 12128;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConcatDTiling, ConcatD_tiling_UT_zero_axis_same_shape_align_float)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "ConcatD",
        {
            {{{131080}, {131080}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{131080}, {131080}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{131080}, {131080}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{393240}, {393240}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1}, &compileInfo);
    uint64_t expectTilingKey = 2114;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConcatDTiling, ConcatD_tiling_UT_zero_axis_diff_shape_align_float)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "ConcatD",
        {
            {{{8256}, {8256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8224}, {8224}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4112}, {4112}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{20602}, {20602}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1}, &compileInfo);
    uint64_t expectTilingKey = 2124;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConcatDTiling, ConcatD_tiling_UT_one_axis_concat_simt_same_shape_no_align_float)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "ConcatD",
        {
            {{{12289}, {12289}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{12289}, {12289}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{12289}, {12289}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{36867}, {36867}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1}, &compileInfo);
    uint64_t expectTilingKey = 2214;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConcatDTiling, ConcatD_tiling_UT_zero_axis_diff_shape_no_align_float)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "ConcatD",
        {
            {{{8224}, {8224}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8223}, {8223}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3223}, {3223}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{19670}, {19670}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1}, &compileInfo);
    uint64_t expectTilingKey = 2224;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConcatDTiling, ConcatD_tiling_UT_one_axis_diff_shape_no_align_float)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "ConcatD",
        {
            {{{8223, 31}, {8223, 31}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8223, 33}, {8223, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8223, 31}, {8223, 31}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{24669, 95}, {24669, 95}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1}, &compileInfo);
    uint64_t expectTilingKey = 12224;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConcatDTiling, ConcatD_tiling_UT_one_axis_diff_shape_no_align_int8)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "ConcatD",
        {
            {{{8223, 31}, {8223, 31}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{8223, 32}, {8223, 32}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{8223, 31}, {8223, 31}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{24669, 94}, {24669, 94}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1}, &compileInfo);
    uint64_t expectTilingKey = 12221;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConcatDTiling, ConcatD_tiling_UT_one_axis_diff_shape_no_align_int64)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "ConcatD",
        {
            {{{8223, 31}, {8223, 31}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8223, 32}, {8223, 32}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8223, 31}, {8223, 31}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{24669, 94}, {24669, 94}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1}, &compileInfo);
    uint64_t expectTilingKey = 12228;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConcatDTiling, ConcatD_tiling_UT_one_axis_same_shape_no_align_float)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "ConcatD",
        {
            {{{8223, 31}, {8223, 31}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8223, 31}, {8223, 31}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8223, 31}, {8223, 31}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{24669, 93}, {24669, 93}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1}, &compileInfo);
    uint64_t expectTilingKey = 12314;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConcatDTiling, ConcatD_tiling_UT_one_axis_same_shape_no_align_int8)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "ConcatD",
        {
            {{{8223, 31}, {8223, 31}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{8223, 31}, {8223, 31}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{8223, 31}, {8223, 31}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{24669, 93}, {24669, 93}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1}, &compileInfo);
    uint64_t expectTilingKey = 12311;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConcatDTiling, ConcatD_tiling_UT_one_axis_same_shape_no_align_int64)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "ConcatD",
        {
            {{{8223, 31}, {8223, 31}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8223, 31}, {8223, 31}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8223, 31}, {8223, 31}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{24669, 93}, {24669, 93}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1}, &compileInfo);
    uint64_t expectTilingKey = 12214;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConcatDTiling, ConcatD_tiling_UT_error_dtype)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "ConcatD",
        {
            {{{8223, 31}, {8223, 31}}, ge::DT_COMPLEX128, ge::FORMAT_ND},
            {{{8223, 32}, {8223, 32}}, ge::DT_COMPLEX128, ge::FORMAT_ND},
            {{{8223, 31}, {8223, 31}}, ge::DT_COMPLEX128, ge::FORMAT_ND},
        },
        {
            {{{24669, 94}, {24669, 94}}, ge::DT_COMPLEX128, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ConcatDTiling, ConcatD_tiling_UT_error_dim)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "ConcatD",
        {
            {{{8223, 31}, {8223, 31}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
            {{{8223, 32}, {8223, 32}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
            {{{8223, 31}, {8223, 31}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
        },
        {
            {{{24669, 94}, {24669, 94}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(5)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ConcatDTiling, ConcatD_tiling_UT_for_more_core)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 73;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "ConcatD",
        {
            {{{8223, 31}, {8223, 31}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{8223, 32}, {8223, 32}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{8223, 31}, {8223, 31}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{24669, 94}, {24669, 94}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1}, &compileInfo);
    uint64_t expectTilingKey = 12221;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConcatDTiling, ConcatD_tiling_UT_one_axis_concat_simt_no_align_float)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara::TensorDescription tensorInput1 = {{{8224}, {8224}}, ge::DT_FLOAT, ge::FORMAT_ND};
    gert::TilingContextPara::TensorDescription tensorInput2 = {{{8225}, {8225}}, ge::DT_FLOAT, ge::FORMAT_ND};
    std::vector<gert::TilingContextPara::TensorDescription> tensorDesc = {tensorInput1, tensorInput2};
    tensorDesc.insert(tensorDesc.end(), 32, tensorInput1);
    gert::TilingContextPara tilingContextPara(
        "ConcatD", tensorDesc,
        {
            {{{271393}, {271393}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(34)},
        },
        {tensorDesc.size()}, {1}, &compileInfo);
    uint64_t expectTilingKey = 30004;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConcatDTiling, ConcatD_tiling_UT_one_axis_concat_simt_diff_shape_no_align_int64)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara::TensorDescription tensorInput1 = {{{8, 31}, {8, 31}}, ge::DT_INT64, ge::FORMAT_ND};
    gert::TilingContextPara::TensorDescription tensorInput2 = {{{8, 32}, {8, 32}}, ge::DT_INT64, ge::FORMAT_ND};
    std::vector<gert::TilingContextPara::TensorDescription> tensorDesc = {tensorInput1, tensorInput2};
    tensorDesc.insert(tensorDesc.end(), 32, tensorInput1);
    gert::TilingContextPara tilingContextPara(
        "ConcatD", tensorDesc,
        {
            {{{8, 1022}, {8, 1022}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(34)},
        },
        {tensorDesc.size()}, {1}, &compileInfo);
    uint64_t expectTilingKey = 30008;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ConcatDTiling, ConcatD_tiling_UT_one_axis_concat_simt_diff_shape_no_align_int8)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara::TensorDescription tensorInput1 = {{{8, 31}, {8, 31}}, ge::DT_INT32, ge::FORMAT_ND};
    gert::TilingContextPara::TensorDescription tensorInput2 = {{{8, 30}, {8, 30}}, ge::DT_INT32, ge::FORMAT_ND};
    std::vector<gert::TilingContextPara::TensorDescription> tensorDesc = {tensorInput1, tensorInput2};
    tensorDesc.insert(tensorDesc.end(), 32, tensorInput1);
    gert::TilingContextPara tilingContextPara(
        "ConcatD", tensorDesc,
        {
            {{{8, 1022}, {8, 1022}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(34)},
        },
        {tensorDesc.size()}, {1}, &compileInfo);
    uint64_t expectTilingKey = 30004;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}
