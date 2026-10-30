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
 * \file test_pad_v3_grad_replication.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "conversion/pad_v3_grad_replication/op_host/arch32/pad_v3_grad_replication_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class PadV3GradReplicationTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "PadV3GradReplicationTiling  SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "PadV3GradReplicationTiling  TearDown" << std::endl;
    }
};

TEST_F(PadV3GradReplicationTiling, pad_v3_grad_replication_tiling_test_success)
{
    optiling::PadV3GradReplicationCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{{{1, 1, 30, 30}, {1, 1, 30, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 28, 28}, {1, 1, 28, 28}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 2;
    std::string expectTilingData =
        "24019106386761045 1125899908240720 18014398526259200 4294967297 128849018910 900 900 4294968196 4294967297 "
        "4294967297 120259084316 784 3367254360848 4294967297 4294967297 1 30 30 26 26 3607772528666 30 28 26 1680 "
        "43690 0 1 2730 0 1 0 0 1 4095 0 26 4095 0 1 43690 0 1 2730 0 1 0 0 1 4095 0 1 ";
    std::vector<size_t> expectWorkspaces = {268864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(PadV3GradReplicationTiling, pad_v3_grad_replication_tiling_test_failed)
{
    optiling::PadV3GradReplicationCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "PadV3GradReplication",
        {{{{1, 1, 30, 30}, {1, 1, 30, 30}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 28, 28}, {1, 1, 28, 28}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}