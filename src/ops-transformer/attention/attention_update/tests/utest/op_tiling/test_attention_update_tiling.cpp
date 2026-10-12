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

#include "util/shape_util.h"
#include "kernel_run_context_facker.h"
#include "experiment_ops.h"
#include "array_ops.h"
#include "nn_other.h"
#include "op_tiling/op_tiling_util.h"
#include "common/utils/ut_op_util.h"
#include "common_unittest.h"
#include "attention_update/decode_update_tiling.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace optiling;

class AttentionUpdateTilingTest : public testing::Test {
 public:

 protected:
  static void SetUpTestCase() {
    std::cout << "RotaryPosEmbInferTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RotaryPosEmbInferTiling TearDown" << std::endl;
  }
};

TEST_F(AttentionUpdateTilingTest, test_tiling_1) {
    std::string op_type("AttentionUpdate");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    string compile_info_string = R"({
      "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 40}
                        })";
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DecodeUpdateCompileInfo compile_info;

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
    auto param = gert::TilingData::CreateCap(16 * 4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(16 * 4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape lseShape = {{128}, {128}};
    gert::StorageShape goShape = {{128, 256}, {128, 256}};
    gert::StorageShape outShape = {{128, 256}, {128, 256}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&lseShape, &goShape})
                      .OutputShapes({&outShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"update_type", ge::AnyValue::CreateFrom<int64_t>(0)},
                          {"sp", ge::AnyValue::CreateFrom<int64_t>(1)}
                      })
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
}

TEST_F(AttentionUpdateTilingTest, test_tiling_2) {
    string compile_info_string_2 = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";


    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> version;
    std::map<std::string,std::string> soc_version_infos = {{"Short_SoC_version","Ascend950"}};
    GetPlatFormInfos(compile_info_string_2.c_str(), soc_infos, aicore_spec, intrinsics);
    version["Short_SoC_version"] = "Ascend950";
    version["SoC_version"] = "Ascend910_9589";

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DecodeUpdateCompileInfo compile_info;
    compile_info.is_ascendc = true;
    compile_info.coreNum = 64;
    compile_info.ubSize = 253952;

    std::string op_type("AttentionUpdate");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
   

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape lseShape = {{128}, {128}};
    gert::StorageShape goShape = {{128, 8}, {128, 8}};
    gert::StorageShape outShape = {{128, 8}, {128, 8}};
    gert::StorageShape lseMShape = {{128}, {128}};

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&lseShape, &goShape})
                      .OutputShapes({&outShape, &lseMShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"update_type", ge::AnyValue::CreateFrom<int64_t>(1)},
                          {"sp", ge::AnyValue::CreateFrom<int64_t>(1)}
                      })
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
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 20000);
}

TEST_F(AttentionUpdateTilingTest, test_tiling_3) {
  string compile_info_string_2 = R"({
      "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false,
                        "Intrinsic_data_move_l12ub": true,
                        "Intrinsic_data_move_l0c2ub": true,
                        "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 64, "socVersion": "Ascend950"}
                        })";


  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  std::map<std::string,std::string> soc_version_infos = {{"Short_SoC_version","Ascend950"}};
  GetPlatFormInfos(compile_info_string_2.c_str(), soc_infos, aicore_spec, intrinsics);
  version["Short_SoC_version"] = "Ascend950";
  version["SoC_version"] = "Ascend910_9589";

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  optiling::DecodeUpdateCompileInfo compile_info;
  compile_info.is_ascendc = true;
  compile_info.coreNum = 64;
  compile_info.ubSize = 253952;

  std::string op_type("AttentionUpdate");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;


  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  ASSERT_NE(param, nullptr);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
  gert::StorageShape lseShape = {{128}, {128}};
  gert::StorageShape goShape = {{128, 8}, {128, 8}};
  gert::StorageShape outShape = {{128, 8}, {128, 8}};
  gert::StorageShape lseMShape = {{128}, {128}};

  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(2, 2)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&lseShape, &goShape})
                    .OutputShapes({&outShape, &lseMShape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                        {"update_type", ge::AnyValue::CreateFrom<int64_t>(1)},
                        {"sp", ge::AnyValue::CreateFrom<int64_t>(1)}
                    })
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
  auto tiling_key = tiling_context->GetTilingKey();
  ASSERT_EQ(tiling_key, 20000);
}

TEST_F(AttentionUpdateTilingTest, test_tiling_4) {
  string compile_info_string_2 = R"({
      "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false,
                        "Intrinsic_data_move_l12ub": true,
                        "Intrinsic_data_move_l0c2ub": true,
                        "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 64, "socVersion": "Ascend950"}
                        })";


  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  std::map<std::string,std::string> soc_version_infos = {{"Short_SoC_version","Ascend950"}};
  GetPlatFormInfos(compile_info_string_2.c_str(), soc_infos, aicore_spec, intrinsics);
  version["Short_SoC_version"] = "Ascend950";
  version["SoC_version"] = "Ascend910_9589";

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  optiling::DecodeUpdateCompileInfo compile_info;
  compile_info.is_ascendc = true;
  compile_info.coreNum = 64;
  compile_info.ubSize = 253952;

  std::string op_type("AttentionUpdate");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;


  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  ASSERT_NE(param, nullptr);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
  gert::StorageShape lseShape = {{128}, {128}};
  gert::StorageShape goShape = {{128, 8}, {128, 8}};
  gert::StorageShape outShape = {{128, 8}, {128, 8}};
  gert::StorageShape lseMShape = {{128}, {128}};

  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(4, 2)
                    .IrInstanceNum({2, 2})
                    .InputShapes({&lseShape, &lseShape, &goShape, &goShape})
                    .OutputShapes({&outShape, &lseMShape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                        {"update_type", ge::AnyValue::CreateFrom<int64_t>(1)},
                        {"sp", ge::AnyValue::CreateFrom<int64_t>(2)}
                    })
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
  auto tiling_key = tiling_context->GetTilingKey();
  ASSERT_EQ(tiling_key, 20000);
}

TEST_F(AttentionUpdateTilingTest, test_tiling_5) {
  string compile_info_string_2 = R"({
      "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false,
                        "Intrinsic_data_move_l12ub": true,
                        "Intrinsic_data_move_l0c2ub": true,
                        "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 64, "socVersion": "Ascend950"}
                        })";


  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  std::map<std::string,std::string> soc_version_infos = {{"Short_SoC_version","Ascend950"}};
  GetPlatFormInfos(compile_info_string_2.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  optiling::DecodeUpdateCompileInfo compile_info;
  compile_info.is_ascendc = true;
  compile_info.coreNum = 64;
  compile_info.ubSize = 253952;

  std::string op_type("AttentionUpdate");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;


  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  ASSERT_NE(param, nullptr);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
  gert::StorageShape lseShape = {{128}, {128}};
  gert::StorageShape goShape = {{128, 8}, {128, 8}};
  gert::StorageShape outShape = {{128, 8}, {128, 8}};
  gert::StorageShape lseMShape = {{128}, {128}};

  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(4, 2)
                    .IrInstanceNum({2, 2})
                    .InputShapes({&lseShape, &lseShape, &goShape, &goShape})
                    .OutputShapes({&outShape, &lseMShape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                        {"update_type", ge::AnyValue::CreateFrom<int64_t>(1)},
                        {"sp", ge::AnyValue::CreateFrom<int64_t>(2)}
                    })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
  tiling_func(tiling_context);

}

TEST_F(AttentionUpdateTilingTest, test_tiling_6) {
  string compile_info_string_2 = R"({
      "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false,
                        "Intrinsic_data_move_l12ub": true,
                        "Intrinsic_data_move_l0c2ub": true,
                        "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 64, "socVersion": "Ascend950"}
                        })";


  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  std::map<std::string,std::string> soc_version_infos = {{"Short_SoC_version","Ascend950"}};
  GetPlatFormInfos(compile_info_string_2.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  optiling::DecodeUpdateCompileInfo compile_info;
  compile_info.is_ascendc = true;
  compile_info.coreNum = 64;
  compile_info.ubSize = 253952;

  std::string op_type("AttentionUpdate");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;


  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  ASSERT_NE(param, nullptr);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
  gert::StorageShape lseShape = {{128}, {128}};
  gert::StorageShape goShape = {{128, 1024}, {128, 1024}};
  gert::StorageShape outShape = {{128, 1024}, {128, 1024}};
  gert::StorageShape lseMShape = {{128}, {128}};

  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(4, 2)
                    .IrInstanceNum({2, 2})
                    .InputShapes({&lseShape, &lseShape, &goShape, &goShape})
                    .OutputShapes({&outShape, &lseMShape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                        {"update_type", ge::AnyValue::CreateFrom<int64_t>(1)},
                        {"sp", ge::AnyValue::CreateFrom<int64_t>(2)}
                    })
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
  tiling_func(tiling_context);
}

TEST_F(AttentionUpdateTilingTest, test_tiling_7) {
  string compile_info_string_2 = R"({
      "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false,
                        "Intrinsic_data_move_l12ub": true,
                        "Intrinsic_data_move_l0c2ub": true,
                        "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 64, "socVersion": "Ascend950"}
                        })";


  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  std::map<std::string,std::string> soc_version_infos = {{"Short_SoC_version","Ascend950"}};
  GetPlatFormInfos(compile_info_string_2.c_str(), soc_infos, aicore_spec, intrinsics);
  version["Short_SoC_version"] = "Ascend950";
  version["SoC_version"] = "Ascend910_9589";

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  optiling::DecodeUpdateCompileInfo compile_info;
  compile_info.is_ascendc = true;
  compile_info.coreNum = 64;
  compile_info.ubSize = 253952;

  std::string op_type("AttentionUpdate");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;


  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  ASSERT_NE(param, nullptr);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
  gert::StorageShape lseShape = {{12800000}, {12800000}};
  gert::StorageShape goShape = {{12800000, 8}, {12800000, 8}};
  gert::StorageShape outShape = {{12800000, 8}, {1280000, 8}};
  gert::StorageShape lseMShape = {{12800000}, {12800000}};

  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(4, 2)
                    .IrInstanceNum({2, 2})
                    .InputShapes({&lseShape, &lseShape, &goShape, &goShape})
                    .OutputShapes({&outShape, &lseMShape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                        {"update_type", ge::AnyValue::CreateFrom<int64_t>(1)},
                        {"sp", ge::AnyValue::CreateFrom<int64_t>(2)}
                    })
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
  auto tiling_key = tiling_context->GetTilingKey();
  ASSERT_EQ(tiling_key, 20000);
}
