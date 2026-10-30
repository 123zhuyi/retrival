// /**
//  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
//  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
//  * CANN Open Software License Agreement Version 2.0 (the "License").
//  * Please refer to the License for details. You may not use this file except in compliance with the License.
//  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
//  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//  * See LICENSE in the root of the software repository for the full text of the License.
//  */

// #include <iostream>
// #include <vector>

// #include <gtest/gtest.h>
// #include "op_log.h"

// #include "runtime2_util.h"
// #include "kernel_run_context_facker.h"
// #include "experiment_ops.h"
// #include "array_ops.h"
// #include "nn_other.h"
// #include "op_tiling/op_tiling_util.h"
// #include "common/utils/ut_op_util.h"
// #include "common_unittest.h"
// #include "../../../op_host/is_inf_tiling_arch35.h"
// #include "../../../op_host/is_inf_tiling_def.h"
// #include "exe_graph/runtime/storage_format.h"
// #include "exe_graph/runtime/storage_shape.h"
// #include "test_cube_util.h"

// using namespace ut_util;
// using namespace std;
// using namespace ge;

// class IsInfTiling : public testing::Test {
//  protected:
//   static void SetUpTestCase() {
//     std::cout << "IsInfTiling Ascend C SetUp" << std::endl;
//   }

//   static void TearDownTestCase() {
//     std::cout << "IsInfTiling Ascend C TearDown" << std::endl;
//   }
// };

// static string TilingData2Str(const gert::TilingData *tiling_data) {
//   auto data = tiling_data->GetData();
//   string result;
//   for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
//     result += std::to_string((reinterpret_cast<const int64_t *>(tiling_data->GetData())[i / sizeof(int64_t)]));
//     result += " ";
//   }

//   return result;
// }

// static void ExecuteTestCase(ge::DataType xDtype, ge::DataType yDtype, 
//                             gert::StorageShape xShape, gert::StorageShape yShape, int tilingKeyValue,
//                             string expectTilingData, ge::graphStatus status = ge::GRAPH_SUCCESS) {
//   string compile_info_string = R"({
//       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
//                         "Intrinsic_fix_pipe_l0c2out": false, 
//                         "Intrinsic_data_move_l12ub": true, 
//                         "Intrinsic_data_move_l0c2ub": true, 
//                         "Intrinsic_data_move_out2l1_nd2nz": false,
//                         "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
//                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
//                         "CORE_NUM": 64, "socVersion": "Ascend950"}
//                         })";
//   map<string, string> soc_infos;
//   map<string, string> aicore_spec;
//   map<string, string> intrinsics;
//   map<string, string> socversions = {{"Short_SoC_version", "ASCEND950"}};
//   GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, socversions);

//   // platform info
//   fe::PlatFormInfos platform_info;
//   platform_info.Init();
    
//   // compile info
//   optiling::ElewiseCompileInfo compile_info;

//   std::string op_type("IsInf");
//   ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//   auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

//   // tilingFunc simulate
//   auto param = gert::TilingData::CreateCap(1024 * 1024);
//   ASSERT_NE(param, nullptr);
//   auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024 * 1024);
//   auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get()); 
//   auto holder = gert::TilingContextFaker()
//                     .NodeIoNum(1, 1)
//                     .IrInstanceNum({1})
//                     .InputShapes({&xShape})
//                     .OutputShapes({&yShape})
//                     .CompileInfo(&compile_info)
//                     .PlatformInfo(reinterpret_cast<char *>(&platform_info))
//                     .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .TilingData(param.get())
//                     .Workspace(ws_size)
//                     .Build();
                    
//   gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//   ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
//   tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);

//   // workspaces nullptr return failed
//   EXPECT_EQ(tiling_func(tiling_context), status);
//   if (status == ge::GRAPH_FAILED) {
//     return;
//   }
//   auto tiling_key = tiling_context->GetTilingKey();
//   EXPECT_EQ(tiling_key, tilingKeyValue);
//   // auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
//   // EXPECT_EQ(tiling_data_result, expectTilingData);
// }

// TEST_F(IsInfTiling, test_tiling_fp16_001) {
//   ge::DataType xDtype = ge::DT_FLOAT16;
//   ge::DataType yDtype = ge::DT_BOOL;
//   gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}}; 
//   gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}}; 
//   int tilingKeyValue = 101;
//   string expectTilingData = "1 8192 4096 2 10496 1 1 4096 4096 10496 ";
//   ExecuteTestCase(xDtype, yDtype, xShape, yShape, tilingKeyValue, expectTilingData, ge::GRAPH_SUCCESS);
// }

// TEST_F(IsInfTiling, test_tiling_bf16_002) {
//   ge::DataType xDtype = ge::DT_BF16;
//   ge::DataType yDtype = ge::DT_BOOL;
//   gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}}; 
//   gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}}; 
//   int tilingKeyValue = 102;
//   string expectTilingData = "1 8192 4096 2 10496 1 1 4096 4096 10496 ";
//   ExecuteTestCase(xDtype, yDtype, xShape, yShape, tilingKeyValue, expectTilingData, ge::GRAPH_SUCCESS);
// }

// TEST_F(IsInfTiling, test_tiling_fp32_003) {
//   ge::DataType xDtype = ge::DT_FLOAT;
//   ge::DataType yDtype = ge::DT_BOOL;
//   gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}}; 
//   gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}}; 
//   int tilingKeyValue = 103;
//   string expectTilingData = "1 8192 4096 2 10496 1 1 4096 4096 10496 ";
//   ExecuteTestCase(xDtype, yDtype, xShape, yShape, tilingKeyValue, expectTilingData, ge::GRAPH_SUCCESS);
// }

// TEST_F(IsInfTiling, test_tiling_invalid_input_dtype_004) {
//   ge::DataType xDtype = ge::DT_INT32;
//   ge::DataType yDtype = ge::DT_BOOL;
//   gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}}; 
//   gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}}; 
//   int tilingKeyValue = -1;
//   string expectTilingData = " ";
//   ExecuteTestCase(xDtype, yDtype, xShape, yShape, tilingKeyValue, expectTilingData, ge::GRAPH_FAILED);
// }

// TEST_F(IsInfTiling, test_tiling_invalid_output_dtype_005) {
//   ge::DataType xDtype = ge::DT_FLOAT;
//   ge::DataType yDtype = ge::DT_INT32;
//   gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}}; 
//   gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}}; 
//   int tilingKeyValue = -1;
//   string expectTilingData = " ";
//   ExecuteTestCase(xDtype, yDtype, xShape, yShape, tilingKeyValue, expectTilingData, ge::GRAPH_FAILED);
// }

// TEST_F(IsInfTiling, test_tiling_invalid_shape_006) {
//   ge::DataType xDtype = ge::DT_FLOAT;
//   ge::DataType yDtype = ge::DT_BOOL;
//   gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}}; 
//   gert::StorageShape yShape = {{1, 64, 2, 6}, {1, 64, 2, 6}}; 
//   int tilingKeyValue = -1;
//   string expectTilingData = " ";
//   ExecuteTestCase(xDtype, yDtype, xShape, yShape, tilingKeyValue, expectTilingData, ge::GRAPH_FAILED);
// }

// TEST_F(IsInfTiling, test_tiling_empty_tensor_007) {
//   ge::DataType xDtype = ge::DT_FLOAT;
//   ge::DataType yDtype = ge::DT_BOOL;
//   gert::StorageShape xShape = {{1, 64, 0, 64}, {1, 64, 0, 64}}; 
//   gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}}; 
//   int tilingKeyValue = -1;
//   string expectTilingData = " ";
//   ExecuteTestCase(xDtype, yDtype, xShape, yShape, tilingKeyValue, expectTilingData, ge::GRAPH_FAILED);
// }

// TEST_F(IsInfTiling, test_tiling_empty_tensor_008) {
//   ge::DataType xDtype = ge::DT_FLOAT;
//   ge::DataType yDtype = ge::DT_BOOL;
//   gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}}; 
//   gert::StorageShape yShape = {{1, 64, 0, 64}, {1, 64, 0, 64}}; 
//   int tilingKeyValue = -1;
//   string expectTilingData = " ";
//   ExecuteTestCase(xDtype, yDtype, xShape, yShape, tilingKeyValue, expectTilingData, ge::GRAPH_FAILED);
// }