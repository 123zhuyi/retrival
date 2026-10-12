/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "ut_op_util.h"
#include "index/non_zero_with_value/op_host/arch35/non_zero_with_value_tiling_arch35.h"
#include "test_cube_util.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
using namespace ut_util;

class NonZeroWithValueTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "NonZeroWithValueTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "NonZeroWithValueTilingTest TearDown" << std::endl; }
};

static void InitPlatForm(fe::PlatFormInfos& platformInfo, map<string, string>& socInfos,
                         map<string, string>& aicoreSpec, map<string, string>& intrinsics)
{
    string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);
    platformInfo.Init();
}

static ge::graphStatus RunTiling(gert::StorageShape& xShape, bool transpose, int64_t& tilingKeyOut)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    int64_t numel = 1;
    for (size_t i = 0; i < xShape.GetStorageShape().GetDimNum(); i++) {
        numel *= xShape.GetStorageShape().GetDim(i);
    }
    gert::StorageShape valueShape = {{numel}, {numel}};
    gert::StorageShape indexShape = {{2 * numel}, {2 * numel}};
    gert::StorageShape countShape = {{1}, {1}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(transpose)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)));

    optiling::NonZeroWithValueCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;

    std::string opType("NonZeroWithValue");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto tilingFunc = opImpl->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 3)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&valueShape, &indexShape, &countShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto ret = tilingFunc(tilingContext);
    tilingKeyOut = static_cast<int64_t>(tilingContext->GetTilingKey());
    return ret;
}

// 非空 2D → general 路径,tilingKey = 1020
TEST_F(NonZeroWithValueTilingTest, general_small_2d)
{
    gert::StorageShape xShape = {{4, 8}, {4, 8}};
    int64_t tilingKey = -1;
    EXPECT_EQ(RunTiling(xShape, true, tilingKey), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingKey, 1020);
}

// 大形状多核 → general 路径,tilingKey = 1020
TEST_F(NonZeroWithValueTilingTest, general_large_2d)
{
    gert::StorageShape xShape = {{1024, 1024}, {1024, 1024}};
    int64_t tilingKey = -1;
    EXPECT_EQ(RunTiling(xShape, true, tilingKey), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingKey, 1020);
}

// 空张量 numel==0 → null 路径,tilingKey = 1000
TEST_F(NonZeroWithValueTilingTest, null_empty_2d)
{
    gert::StorageShape xShape = {{0, 8}, {0, 8}};
    int64_t tilingKey = -1;
    EXPECT_EQ(RunTiling(xShape, true, tilingKey), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingKey, 1000);
}

// transpose=false 不支持 → 失败
TEST_F(NonZeroWithValueTilingTest, transpose_false_fail)
{
    gert::StorageShape xShape = {{4, 8}, {4, 8}};
    int64_t tilingKey = -1;
    EXPECT_EQ(RunTiling(xShape, false, tilingKey), ge::GRAPH_FAILED);
}
