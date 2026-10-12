/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_group_norm_silu_quant_tiling.cpp
 * \brief GroupNormSiluQuant arch35 (Ascend950) regbase tiling UT.
 *
 * Covers the arch35 dispatch that the A2 (arch22) UT does not touch:
 *   - same-type fp16/bf16 x per-tensor / per-channel quantScale all produce a valid regbase
 *     tiling key in {1100,1110,1120,1130,1140,1150};
 *   - empty tensor short-circuits to TILINGKEY_EMPTY_TENSOR (1000);
 *   - the input-legality guards reject: gamma/beta dtype != x, gamma/beta element count != channel,
 *     quantScale element count != 1 and != channel, num_groups <= 0.
 * Platform is fed as Ascend950 with a 256KB UB / 64 cores, matching the real A5 regbase target.
 */

#include <iostream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_util.h"
#include "platform/platform_infos_def.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../../op_host/arch35/group_norm_silu_quant_tiling_arch35.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class GroupNormSiluQuantTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GroupNormSiluQuantTilingArch35 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GroupNormSiluQuantTilingArch35 TearDown" << std::endl; }
};

namespace {
// A5 (Ascend950) regbase target: 256KB UB, 64 AIV cores.
constexpr uint64_t UB_SIZE_256K = 262144;
constexpr int64_t TILINGKEY_EMPTY_TENSOR = 1000;

static bool IsValidRegbaseKey(int64_t key)
{
    return key == 1100 || key == 1110 || key == 1120 || key == 1130 || key == 1140 || key == 1150;
}

static gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape s;
    for (auto d : dims) {
        s.MutableStorageShape().AppendDim(d);
        s.MutableOriginShape().AppendDim(d);
    }
    return s;
}

// Runs the GroupNormSiluQuant arch35 tiling once with fully explicit input shapes/dtypes and returns
// its graphStatus. On GRAPH_SUCCESS, tilingKey is filled with the produced tiling key.
// gamma/beta/quantScale shapes and dtypes are passed explicitly so the negative legality tests can
// feed mismatched values.
static ge::graphStatus RunGnsqArch35TilingShapes(const std::vector<int64_t>& xDims,
                                                 const std::vector<int64_t>& gammaDims,
                                                 const std::vector<int64_t>& betaDims,
                                                 const std::vector<int64_t>& quantScaleDims, int64_t numGroups,
                                                 ge::DataType xDt, ge::DataType gammaDt, ge::DataType betaDt,
                                                 bool activateSilu, uint64_t ubSize, int64_t& tilingKey)
{
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": )" +
                                      std::to_string(ubSize) + R"(, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::GroupNormSiluQuantRegbaseCompileInfo compile_info;

    std::string op_type("GroupNormSiluQuant");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    if (opImpl == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto tiling_func = opImpl->tiling;
    auto tiling_parse_func = opImpl->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init();
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    if (tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // mean/rstd are (N, numGroups) flattened.
    int64_t nDim = xDims.empty() ? 0 : xDims[0];
    std::vector<int64_t> meanDims = {nDim * numGroups};

    gert::StorageShape x_shape = MakeShape(xDims);
    gert::StorageShape gamma_shape = MakeShape(gammaDims);
    gert::StorageShape beta_shape = MakeShape(betaDims);
    gert::StorageShape quantScale_shape = MakeShape(quantScaleDims);
    gert::StorageShape y_shape = MakeShape(xDims);
    gert::StorageShape mean_shape = MakeShape(meanDims);
    gert::StorageShape rstd_shape = MakeShape(meanDims);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    if (param == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 3)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x_shape, &gamma_shape, &beta_shape, &quantScale_shape})
                      .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, xDt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, gammaDt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, betaDt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, xDt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, xDt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(numGroups)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                                  {"activate_silu", Ops::NN::AnyValue::CreateFrom<bool>(activateSilu)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    if (tiling_context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto status = tiling_func(tiling_context);
    if (status == ge::GRAPH_SUCCESS) {
        tilingKey = tiling_context->GetTilingKey();
    }
    return status;
}

// Convenience wrapper: well-formed gamma/beta = {channel}, quantScale per mode, same dtype as x.
// perChannel=true -> quantScale = {channel}; false -> quantScale = {1}.
static ge::graphStatus RunGnsqArch35(const std::vector<int64_t>& xDims, int64_t numGroups, ge::DataType dt,
                                     bool perChannel, bool activateSilu, int64_t& tilingKey)
{
    int64_t channel = xDims.size() > 1 ? xDims[1] : 0;
    std::vector<int64_t> quantScaleDims = perChannel ? std::vector<int64_t>{channel} : std::vector<int64_t>{1};
    return RunGnsqArch35TilingShapes(xDims, {channel}, {channel}, quantScaleDims, numGroups, dt, dt, dt, activateSilu,
                                     UB_SIZE_256K, tilingKey);
}
} // namespace

// ---------- 正向: 各 dtype x quant 模式 x silu 都产出合法 regbase key ----------
TEST_F(GroupNormSiluQuantTilingArch35, arch35_fp16_perchannel_silu)
{
    int64_t key = -1;
    auto status = RunGnsqArch35({4, 320, 16, 16}, 32, ge::DT_FLOAT16, true, true, key);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_TRUE(IsValidRegbaseKey(key)) << "unexpected tilingKey " << key;
}

TEST_F(GroupNormSiluQuantTilingArch35, arch35_bf16_perchannel_silu)
{
    int64_t key = -1;
    auto status = RunGnsqArch35({4, 320, 16, 16}, 32, ge::DT_BF16, true, true, key);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_TRUE(IsValidRegbaseKey(key)) << "unexpected tilingKey " << key;
}

TEST_F(GroupNormSiluQuantTilingArch35, arch35_fp16_pertensor_silu)
{
    int64_t key = -1;
    auto status = RunGnsqArch35({4, 320, 16, 16}, 32, ge::DT_FLOAT16, false, true, key);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_TRUE(IsValidRegbaseKey(key)) << "unexpected tilingKey " << key;
}

TEST_F(GroupNormSiluQuantTilingArch35, arch35_bf16_pertensor_nosilu)
{
    int64_t key = -1;
    auto status = RunGnsqArch35({4, 320, 16, 16}, 32, ge::DT_BF16, false, false, key);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_TRUE(IsValidRegbaseKey(key)) << "unexpected tilingKey " << key;
}

// 大 reduce(单组大)与 tiny 多组两类极端形状, 走 split_reduce(1140)/many_tiny_groups(1150) 分支; 断言仍是合法 key。
TEST_F(GroupNormSiluQuantTilingArch35, arch35_fp16_large_reduce)
{
    int64_t key = -1;
    auto status = RunGnsqArch35({1, 48, 32768}, 2, ge::DT_FLOAT16, false, true, key);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_TRUE(IsValidRegbaseKey(key)) << "unexpected tilingKey " << key;
}

TEST_F(GroupNormSiluQuantTilingArch35, arch35_bf16_many_tiny_groups)
{
    int64_t key = -1;
    auto status = RunGnsqArch35({2, 5120, 4}, 1280, ge::DT_BF16, false, true, key);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_TRUE(IsValidRegbaseKey(key)) << "unexpected tilingKey " << key;
}

// ---------- 空 tensor 短路 ----------
TEST_F(GroupNormSiluQuantTilingArch35, arch35_empty_tensor)
{
    int64_t key = -1;
    auto status = RunGnsqArch35({0, 32, 16}, 4, ge::DT_FLOAT16, false, true, key);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, TILINGKEY_EMPTY_TENSOR);
}

// ---------- 负向: 输入合法性守卫 ----------
TEST_F(GroupNormSiluQuantTilingArch35, arch35_reject_gamma_dtype_mismatch)
{
    // x=fp16, gamma=bf16 -> The dtype of gamma must be the same as x
    int64_t key = -1;
    auto status = RunGnsqArch35TilingShapes({4, 320, 16, 16}, {320}, {320}, {320}, 32, ge::DT_FLOAT16, ge::DT_BF16,
                                            ge::DT_FLOAT16, true, UB_SIZE_256K, key);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(GroupNormSiluQuantTilingArch35, arch35_reject_beta_dtype_mismatch)
{
    int64_t key = -1;
    auto status = RunGnsqArch35TilingShapes({4, 320, 16, 16}, {320}, {320}, {320}, 32, ge::DT_FLOAT16, ge::DT_FLOAT16,
                                            ge::DT_BF16, true, UB_SIZE_256K, key);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(GroupNormSiluQuantTilingArch35, arch35_reject_gamma_size_mismatch)
{
    // gamma 元素数 != channel(320)
    int64_t key = -1;
    auto status = RunGnsqArch35TilingShapes({4, 320, 16, 16}, {256}, {320}, {320}, 32, ge::DT_FLOAT16, ge::DT_FLOAT16,
                                            ge::DT_FLOAT16, true, UB_SIZE_256K, key);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(GroupNormSiluQuantTilingArch35, arch35_reject_quantscale_bad_size)
{
    // quantScale 元素数既不是 1 也不是 channel(320) -> 拒绝
    int64_t key = -1;
    auto status = RunGnsqArch35TilingShapes({4, 320, 16, 16}, {320}, {320}, {3}, 32, ge::DT_FLOAT16, ge::DT_FLOAT16,
                                            ge::DT_FLOAT16, true, UB_SIZE_256K, key);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(GroupNormSiluQuantTilingArch35, arch35_reject_num_groups_nonpositive)
{
    int64_t key = -1;
    auto status = RunGnsqArch35({4, 320, 16, 16}, 0, ge::DT_FLOAT16, true, true, key);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(GroupNormSiluQuantTilingArch35, arch35_reject_x_dims_below_min)
{
    // x 维度 < 2 -> 拒绝(与 A2 一致)
    int64_t key = -1;
    auto status = RunGnsqArch35({320}, 32, ge::DT_FLOAT16, true, true, key);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(GroupNormSiluQuantTilingArch35, arch35_reject_x_dims_above_max)
{
    // x 维度 > 8 -> 拒绝(与 A2 一致,支持范围 2-8 维)
    int64_t key = -1;
    auto status = RunGnsqArch35({2, 32, 1, 1, 1, 1, 1, 1, 1}, 4, ge::DT_FLOAT16, true, true, key);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}
