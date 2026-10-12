/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "op_log.h"
#include "kernel_run_context_facker.h"
#undef private
#include "runtime2_util.h"
#include "experiment_ops.h"
#include "array_ops.h"
#include "op_tiling/op_tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "common/utils/ut_op_util.h"
#include "common_unittest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "auto_tiling_rt2.h"

class QuantGroupedMatmulDequantTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "QuantGroupedMatmulDequantTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "QuantGroupedMatmulDequantTiling TearDown" << std::endl;
    }
};

TEST_F(QuantGroupedMatmulDequantTiling, test_tiling_float16_1) {
    std::string op_type("QuantGroupedMatmulDequant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({
                                        "hardware_info": {
                                            "BT_SIZE": 0,
                                            "load3d_constraints": "1",
                                            "Intrinsic_fix_pipe_l0c2out": false,
                                            "Intrinsic_data_move_l12ub": true,
                                            "Intrinsic_data_move_l0c2ub": true,
                                            "Intrinsic_data_move_out2l1_nd2nz": false,
                                            "UB_SIZE": 262144,
                                            "L2_SIZE": 16777216,
                                            "L1_SIZE": 1048576,
                                            "L0A_SIZE": 65536,
                                            "L0B_SIZE": 65536,
                                            "L0C_SIZE": 262144,
                                            "CORE_NUM": 8
                                        }
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct QuantGroupedMatmulDequantCompileInfo {
        uint32_t coreNum;
        uint64_t ubSize;
        uint64_t l1Size;
        uint64_t l2Size;
        uint64_t l0CSize;
        uint64_t l0ASize;
        uint64_t l0BSize;
        platform_ascendc::SocVersion socVersion;
        std::string socVersionStr = "";
    } compile_info;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    int64_t G = 16;
    int64_t M = 256;
    int64_t N = 1024;
    int64_t K = 1024;
    gert::StorageShape xShapes = {{M,K}, {M,K}};
    gert::StorageShape weightShapes = {{G,K/32,N/16,16,32}, {G,K/32,N/16,16,32}};
    gert::StorageShape weightScaleShapes = {{G,N}, {G,N}};
    gert::StorageShape groupListShapes = {{G}, {G}};
    gert::StorageShape xScaleShapes = {{M}, {M}};
    gert::StorageShape smoothScaleShapes = {{K}, {K}};
    gert::StorageShape yShapes = {{M,N}, {M,N}};
    string quant_mode("pertoken");
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(8, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&xShapes, &weightShapes, &weightScaleShapes, &groupListShapes,
                                    nullptr, &xScaleShapes, nullptr, &smoothScaleShapes})
                      .OutputShapes({&yShapes})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"x_quant_mode", ge::AnyValue::CreateFrom<std::string>(quant_mode)},
                                {"transpose_weight", ge::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 10000003);
}

TEST_F(QuantGroupedMatmulDequantTiling, test_tiling_float16_2) {
    std::string op_type("QuantGroupedMatmulDequant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({
                                        "hardware_info": {
                                            "BT_SIZE": 0,
                                            "load3d_constraints": "1",
                                            "Intrinsic_fix_pipe_l0c2out": false,
                                            "Intrinsic_data_move_l12ub": true,
                                            "Intrinsic_data_move_l0c2ub": true,
                                            "Intrinsic_data_move_out2l1_nd2nz": false,
                                            "UB_SIZE": 262144,
                                            "L2_SIZE": 16777216,
                                            "L1_SIZE": 1048576,
                                            "L0A_SIZE": 65536,
                                            "L0B_SIZE": 65536,
                                            "L0C_SIZE": 262144,
                                            "CORE_NUM": 8
                                        }
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct QuantGroupedMatmulDequantCompileInfo {
        uint32_t coreNum;
        uint64_t ubSize;
        uint64_t l1Size;
        uint64_t l2Size;
        uint64_t l0CSize;
        uint64_t l0ASize;
        uint64_t l0BSize;
        platform_ascendc::SocVersion socVersion;
        std::string socVersionStr = "";
    } compile_info;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    int64_t G = 8;
    int64_t M = 128;
    int64_t N = 5120;
    int64_t K = 1344;
    gert::StorageShape xShapes = {{M,K}, {M,K}};
    gert::StorageShape weightShapes = {{G,K/32,N/16,16,32}, {G,K/32,N/16,16,32}};
    gert::StorageShape weightScaleShapes = {{G,N}, {G,N}};
    gert::StorageShape groupListShapes = {{G}, {G}};
    gert::StorageShape xScaleShapes = {{1}, {1}};
    gert::StorageShape smoothScaleShapes = {{K}, {K}};
    gert::StorageShape yShapes = {{M,N}, {M,N}};
    string quant_mode("pertensor");
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(8, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&xShapes, &weightShapes, &weightScaleShapes, &groupListShapes,
                                    nullptr, &xScaleShapes, nullptr, &smoothScaleShapes})
                      .OutputShapes({&yShapes})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"x_quant_mode", ge::AnyValue::CreateFrom<std::string>(quant_mode)},
                                {"transpose_weight", ge::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 10000004);
}