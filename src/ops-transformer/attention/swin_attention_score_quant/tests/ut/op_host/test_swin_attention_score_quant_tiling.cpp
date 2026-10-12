/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include<gtest/gtest.h>

#include<iostream>
#include<map>
#include<sstream>
#include<string>
#include<vector>

#include"op_log.h"

#include"array_ops.h"
#include"common/utils/ut_op_util.h"
#include"common_unittest.h"
#include"exe_graph/runtime/storage_format.h"
#include"exe_graph/runtime/storage_shape.h"
#include"exe_graph/runtime/tiling_parse_context.h"
#include"fusion_ops.h"
#include"kernel_run_context_facker.h"
#include"op_tiling/op_tiling_util.h"
#include"test_cube_util.h"
#include"../../../op_host/swin_attention_score_quant_tiling.h"
#include"runtime2_util.h"
#include"platform/platform_info.h"
#include"register/op_def_registry.h"
#include"tiling/tiling_api.h"

using namespace std;

using namespace ge;
using namespace ut_util;
class SwinAttentionScoreQuantTiling:public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout<<"SwinAttentionScoreQuantTiling Setup"<<std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout<<"SwinAttentionScoreQuantTiling TearDown"<<std::endl;
    }
};

struct SwinAttentionScoreQuantCompileInfo {};

TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_with_mask_001)
{
    uint32_t b = 2048;
    uint32_t n = 3;
    uint32_t s = 49;
    uint32_t h = 32;
    gert::StorageShape query_shape = {{b,n,s,h},{b,n,s,h}};
    gert::StorageShape key_shape={{b,n,s,h},{b,n,s,h}};
    gert::StorageShape value_shape={{b,n,s,h},{b,n,s,h}};
    gert::StorageShape scale_quant_shape={{1,s},{1,s}};
    gert::StorageShape scale_dequant1_shape={{1,s},{1,s}};
    gert::StorageShape scale_dequant2_shape={{1,h},{1,h}};
    gert::StorageShape bias_quant_shape={{1,s},{1,s}};
    gert::StorageShape bias_dequant1_shape={{1,s},{1,s}};
    gert::StorageShape bias_dequant2_shape={{1,h},{1,h}};
    gert::StorageShape padding_mask1_shape={{1,n,s,s},{1,n,s,s}};
    gert::StorageShape padding_mask2_shape={{1,n,s,s},{1,n,s,s}};
    gert::StorageShape attention_score_shape={{b,n,s,h},{b,n,s,h}};
    bool query_transpose = false;
    bool key_transpose = false;
    bool value_transpose = false;
    int32_t softmax_axes = -1;	

    std::vector<gert::StorageShape> output_shapes(1,{{b,n,s,h},{b,n,s,h}});
    std::vector<void *> output_shapes_ref(1);
    for(size_t i=0; i<output_shapes.size(); ++i)
    {
        output_shapes_ref[i]=&output_shapes[i];
    }

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 8}
                          })";
    map<string, string> soc_version_infos;
    soc_version_infos.insert(make_pair("Short_SoC_version", "Ascend310P"));
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    
    SwinAttentionScoreQuantCompileInfo compile_info;

    std::string op_type("SwinAttentionScoreQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
        .KernelIONum(2, 1)
        .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
        .Outputs({&compile_info})
        .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
        .NodeIoNum(11, 1)
        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
        .InputShapes(
            {&query_shape, &key_shape, &value_shape, &scale_quant_shape, &scale_dequant1_shape, &scale_dequant2_shape,
            &bias_quant_shape, &bias_dequant1_shape, &bias_dequant2_shape, &padding_mask1_shape, &padding_mask2_shape}
        )
        // .OutputShapes(output_shapes_ref)
        .OutputShapes({&attention_score_shape})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char *>(&platform_info))
        .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(5, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(8, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(9, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(10, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({
            {"query_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"key_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"value_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"softmax_axes", ge::AnyValue::CreateFrom<int64_t>(-1)}
        })
        .TilingData(param.get())
        .Workspace(ws_size)
        .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_without_mask_002)
{
    uint32_t b = 2048;
    uint32_t n = 3;
    uint32_t s = 49;
    uint32_t h = 32;
    gert::StorageShape query_shape = {{b,n,s,h},{b,n,s,h}};
    gert::StorageShape key_shape={{b,n,s,h},{b,n,s,h}};
    gert::StorageShape value_shape={{b,n,s,h},{b,n,s,h}};
    gert::StorageShape scale_quant_shape={{1,s},{1,s}};
    gert::StorageShape scale_dequant1_shape={{1,s},{1,s}};
    gert::StorageShape scale_dequant2_shape={{1,h},{1,h}};
    gert::StorageShape bias_quant_shape={{1,s},{1,s}};
    gert::StorageShape bias_dequant1_shape={{1,s},{1,s}};
    gert::StorageShape bias_dequant2_shape={{1,h},{1,h}};
    gert::StorageShape padding_mask1_shape={{1,n,s,s},{1,n,s,s}};
    gert::StorageShape padding_mask2_shape={{1,n,s,s},{1,n,s,s}};
    gert::StorageShape attention_score_shape={{b,n,s,h},{b,n,s,h}};
    bool query_transpose = false;
    bool key_transpose = false;
    bool value_transpose = false;
    int32_t softmax_axes = -1;	

    std::vector<gert::StorageShape> output_shapes(1,{{b,n,s,h},{b,n,s,h}});
    std::vector<void *> output_shapes_ref(1);
    for(size_t i=0; i<output_shapes.size(); ++i)
    {
        output_shapes_ref[i]=&output_shapes[i];
    }

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 8}
                          })";
    map<string, string> soc_version_infos;
    soc_version_infos.insert(make_pair("Short_SoC_version", "Ascend310P"));
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    
    SwinAttentionScoreQuantCompileInfo compile_info;

    std::string op_type("SwinAttentionScoreQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
        .KernelIONum(2, 1)
        .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
        .Outputs({&compile_info})
        .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
        .NodeIoNum(11, 1)
        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
        .InputShapes(
            {&query_shape, &key_shape, &value_shape, &scale_quant_shape, &scale_dequant1_shape, &scale_dequant2_shape,
            &bias_quant_shape, &bias_dequant1_shape, &bias_dequant2_shape, nullptr, &padding_mask2_shape}
        )
        // .OutputShapes(output_shapes_ref)
        .OutputShapes({&attention_score_shape})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char *>(&platform_info))
        .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(5, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(8, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(9, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(10, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({
            {"query_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"key_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"value_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"softmax_axes", ge::AnyValue::CreateFrom<int64_t>(-1)}
        })
        .TilingData(param.get())
        .Workspace(ws_size)
        .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_null_input_003)
{
    uint32_t b = 2048;
    uint32_t n = 3;
    uint32_t s = 49;
    uint32_t h = 32;
    gert::StorageShape query_shape = {{b,n,s,h},{b,n,s,h}};
    gert::StorageShape key_shape={{b,n,s,h},{b,n,s,h}};
    gert::StorageShape value_shape={{b,n,s,h},{b,n,s,h}};
    gert::StorageShape scale_quant_shape={{1,s},{1,s}};
    gert::StorageShape scale_dequant1_shape={{1,s},{1,s}};
    gert::StorageShape scale_dequant2_shape={{1,h},{1,h}};
    gert::StorageShape bias_quant_shape={{1,s},{1,s}};
    gert::StorageShape bias_dequant1_shape={{1,s},{1,s}};
    gert::StorageShape bias_dequant2_shape={{1,h},{1,h}};
    gert::StorageShape padding_mask1_shape={{1,n,s,s},{1,n,s,s}};
    gert::StorageShape padding_mask2_shape={{1,n,s,s},{1,n,s,s}};
    gert::StorageShape attention_score_shape={{b,n,s,h},{b,n,s,h}};
    bool query_transpose = false;
    bool key_transpose = false;
    bool value_transpose = false;
    int32_t softmax_axes = -1;	

    std::vector<gert::StorageShape> output_shapes(1,{{b,n,s,h},{b,n,s,h}});
    std::vector<void *> output_shapes_ref(1);
    for(size_t i=0; i<output_shapes.size(); ++i)
    {
        output_shapes_ref[i]=&output_shapes[i];
    }

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 8}
                          })";
    map<string, string> soc_version_infos;
    soc_version_infos.insert(make_pair("Short_SoC_version", "Ascend310P"));
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    
    SwinAttentionScoreQuantCompileInfo compile_info;

    std::string op_type("SwinAttentionScoreQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
        .KernelIONum(2, 1)
        .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
        .Outputs({&compile_info})
        .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
        .NodeIoNum(11, 1)
        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
        .InputShapes(
            {&query_shape, &key_shape, &value_shape, &scale_quant_shape, &scale_dequant1_shape, &scale_dequant2_shape,
            nullptr, nullptr, nullptr, &padding_mask1_shape, &padding_mask2_shape}
        )
        // .OutputShapes(output_shapes_ref)
        .OutputShapes({&attention_score_shape})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char *>(&platform_info))
        .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(5, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(8, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(9, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(10, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({
            {"query_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"key_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"value_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"softmax_axes", ge::AnyValue::CreateFrom<int64_t>(-1)}
        })
        .TilingData(param.get())
        .Workspace(ws_size)
        .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_wrong_s_004)
{
    uint32_t b = 2048;
    uint32_t n = 3;
    uint32_t s = 1025;
    uint32_t h = 32;
    gert::StorageShape query_shape = {{b,n,s,h},{b,n,s,h}};
    gert::StorageShape key_shape={{b,n,s,h},{b,n,s,h}};
    gert::StorageShape value_shape={{b,n,s,h},{b,n,s,h}};
    gert::StorageShape scale_quant_shape={{1,s},{1,s}};
    gert::StorageShape scale_dequant1_shape={{1,s},{1,s}};
    gert::StorageShape scale_dequant2_shape={{1,h},{1,h}};
    gert::StorageShape bias_quant_shape={{1,s},{1,s}};
    gert::StorageShape bias_dequant1_shape={{1,s},{1,s}};
    gert::StorageShape bias_dequant2_shape={{1,h},{1,h}};
    gert::StorageShape padding_mask1_shape={{1,n,s,s},{1,n,s,s}};
    gert::StorageShape padding_mask2_shape={{1,n,s,s},{1,n,s,s}};
    gert::StorageShape attention_score_shape={{b,n,s,h},{b,n,s,h}};
    bool query_transpose = false;
    bool key_transpose = false;
    bool value_transpose = false;
    int32_t softmax_axes = -1;	

    std::vector<gert::StorageShape> output_shapes(1,{{b,n,s,h},{b,n,s,h}});
    std::vector<void *> output_shapes_ref(1);
    for(size_t i=0; i<output_shapes.size(); ++i)
    {
        output_shapes_ref[i]=&output_shapes[i];
    }

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 8}
                          })";
    map<string, string> soc_version_infos;
    soc_version_infos.insert(make_pair("Short_SoC_version", "Ascend310P"));
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    
    SwinAttentionScoreQuantCompileInfo compile_info;

    std::string op_type("SwinAttentionScoreQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
        .KernelIONum(2, 1)
        .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
        .Outputs({&compile_info})
        .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
        .NodeIoNum(11, 1)
        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
        .InputShapes(
            {&query_shape, &key_shape, &value_shape, &scale_quant_shape, &scale_dequant1_shape, &scale_dequant2_shape,
            &bias_quant_shape, &bias_dequant1_shape, &bias_dequant2_shape, &padding_mask1_shape, &padding_mask2_shape}
        )
        // .OutputShapes(output_shapes_ref)
        .OutputShapes({&attention_score_shape})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char *>(&platform_info))
        .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(5, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(8, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(9, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(10, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({
            {"query_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"key_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"value_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"softmax_axes", ge::AnyValue::CreateFrom<int64_t>(-1)}
        })
        .TilingData(param.get())
        .Workspace(ws_size)
        .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_wrong_h_005)
{
    uint32_t b = 2048;
    uint32_t n = 3;
    uint32_t s = 49;
    uint32_t h = 33;
    gert::StorageShape query_shape = {{b,n,s,h},{b,n,s,h}};
    gert::StorageShape key_shape={{b,n,s,h},{b,n,s,h}};
    gert::StorageShape value_shape={{b,n,s,h},{b,n,s,h}};
    gert::StorageShape scale_quant_shape={{1,s},{1,s}};
    gert::StorageShape scale_dequant1_shape={{1,s},{1,s}};
    gert::StorageShape scale_dequant2_shape={{1,h},{1,h}};
    gert::StorageShape bias_quant_shape={{1,s},{1,s}};
    gert::StorageShape bias_dequant1_shape={{1,s},{1,s}};
    gert::StorageShape bias_dequant2_shape={{1,h},{1,h}};
    gert::StorageShape padding_mask1_shape={{1,n,s,s},{1,n,s,s}};
    gert::StorageShape padding_mask2_shape={{1,n,s,s},{1,n,s,s}};
    gert::StorageShape attention_score_shape={{b,n,s,h},{b,n,s,h}};
    bool query_transpose = false;
    bool key_transpose = false;
    bool value_transpose = false;
    int32_t softmax_axes = -1;	

    std::vector<gert::StorageShape> output_shapes(1,{{b,n,s,h},{b,n,s,h}});
    std::vector<void *> output_shapes_ref(1);
    for(size_t i=0; i<output_shapes.size(); ++i)
    {
        output_shapes_ref[i]=&output_shapes[i];
    }

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 8}
                          })";
    map<string, string> soc_version_infos;
    soc_version_infos.insert(make_pair("Short_SoC_version", "Ascend310P"));
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    
    SwinAttentionScoreQuantCompileInfo compile_info;

    std::string op_type("SwinAttentionScoreQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
        .KernelIONum(2, 1)
        .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
        .Outputs({&compile_info})
        .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
        .NodeIoNum(11, 1)
        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
        .InputShapes(
            {&query_shape, &key_shape, &value_shape, &scale_quant_shape, &scale_dequant1_shape, &scale_dequant2_shape,
            &bias_quant_shape, &bias_dequant1_shape, &bias_dequant2_shape, &padding_mask1_shape, &padding_mask2_shape}
        )
        // .OutputShapes(output_shapes_ref)
        .OutputShapes({&attention_score_shape})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char *>(&platform_info))
        .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(5, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(8, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(9, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(10, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({
            {"query_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"key_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"value_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"softmax_axes", ge::AnyValue::CreateFrom<int64_t>(-1)}
        })
        .TilingData(param.get())
        .Workspace(ws_size)
        .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_wrong_key_006)
{
    uint32_t b = 2048;
    uint32_t n = 3;
    uint32_t s = 49;
    uint32_t h = 32;
    gert::StorageShape query_shape = {{b,n,s,h},{b,n,s,h}};
    gert::StorageShape key_shape={{b,n,s+1,h},{b,n,s+1,h}};
    gert::StorageShape value_shape={{b,n,s,h},{b,n,s,h}};
    gert::StorageShape scale_quant_shape={{1,s},{1,s}};
    gert::StorageShape scale_dequant1_shape={{1,s},{1,s}};
    gert::StorageShape scale_dequant2_shape={{1,h},{1,h}};
    gert::StorageShape bias_quant_shape={{1,s},{1,s}};
    gert::StorageShape bias_dequant1_shape={{1,s},{1,s}};
    gert::StorageShape bias_dequant2_shape={{1,h},{1,h}};
    gert::StorageShape padding_mask1_shape={{1,n,s,s},{1,n,s,s}};
    gert::StorageShape padding_mask2_shape={{1,n,s,s},{1,n,s,s}};
    gert::StorageShape attention_score_shape={{b,n,s,h},{b,n,s,h}};
    bool query_transpose = false;
    bool key_transpose = false;
    bool value_transpose = false;
    int32_t softmax_axes = -1;	

    std::vector<gert::StorageShape> output_shapes(1,{{b,n,s,h},{b,n,s,h}});
    std::vector<void *> output_shapes_ref(1);
    for(size_t i=0; i<output_shapes.size(); ++i)
    {
        output_shapes_ref[i]=&output_shapes[i];
    }

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 8}
                          })";
    map<string, string> soc_version_infos;
    soc_version_infos.insert(make_pair("Short_SoC_version", "Ascend310P"));
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    
    SwinAttentionScoreQuantCompileInfo compile_info;

    std::string op_type("SwinAttentionScoreQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
        .KernelIONum(2, 1)
        .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
        .Outputs({&compile_info})
        .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
        .NodeIoNum(11, 1)
        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
        .InputShapes(
            {&query_shape, &key_shape, &value_shape, &scale_quant_shape, &scale_dequant1_shape, &scale_dequant2_shape,
            &bias_quant_shape, &bias_dequant1_shape, &bias_dequant2_shape, &padding_mask1_shape, &padding_mask2_shape}
        )
        // .OutputShapes(output_shapes_ref)
        .OutputShapes({&attention_score_shape})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char *>(&platform_info))
        .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(5, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(8, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(9, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(10, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({
            {"query_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"key_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"value_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"softmax_axes", ge::AnyValue::CreateFrom<int64_t>(-1)}
        })
        .TilingData(param.get())
        .Workspace(ws_size)
        .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_wrong_value_007)
{
    uint32_t b = 2048;
    uint32_t n = 3;
    uint32_t s = 49;
    uint32_t h = 32;
    gert::StorageShape query_shape = {{b,n,s,h},{b,n,s,h}};
    gert::StorageShape key_shape={{b,n,s,h},{b,n,s,h}};
    gert::StorageShape value_shape={{b,n+1,s,h},{b,n+1,s,h}};
    gert::StorageShape scale_quant_shape={{1,s},{1,s}};
    gert::StorageShape scale_dequant1_shape={{1,s},{1,s}};
    gert::StorageShape scale_dequant2_shape={{1,h},{1,h}};
    gert::StorageShape bias_quant_shape={{1,s},{1,s}};
    gert::StorageShape bias_dequant1_shape={{1,s},{1,s}};
    gert::StorageShape bias_dequant2_shape={{1,h},{1,h}};
    gert::StorageShape padding_mask1_shape={{1,n,s,s},{1,n,s,s}};
    gert::StorageShape padding_mask2_shape={{1,n,s,s},{1,n,s,s}};
    gert::StorageShape attention_score_shape={{b,n,s,h},{b,n,s,h}};
    bool query_transpose = false;
    bool key_transpose = false;
    bool value_transpose = false;
    int32_t softmax_axes = -1;	

    std::vector<gert::StorageShape> output_shapes(1,{{b,n,s,h},{b,n,s,h}});
    std::vector<void *> output_shapes_ref(1);
    for(size_t i=0; i<output_shapes.size(); ++i)
    {
        output_shapes_ref[i]=&output_shapes[i];
    }

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 8}
                          })";
    map<string, string> soc_version_infos;
    soc_version_infos.insert(make_pair("Short_SoC_version", "Ascend310P"));
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    
    SwinAttentionScoreQuantCompileInfo compile_info;

    std::string op_type("SwinAttentionScoreQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
        .KernelIONum(2, 1)
        .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
        .Outputs({&compile_info})
        .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
        .NodeIoNum(11, 1)
        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
        .InputShapes(
            {&query_shape, &key_shape, &value_shape, &scale_quant_shape, &scale_dequant1_shape, &scale_dequant2_shape,
            &bias_quant_shape, &bias_dequant1_shape, &bias_dequant2_shape, &padding_mask1_shape, &padding_mask2_shape}
        )
        // .OutputShapes(output_shapes_ref)
        .OutputShapes({&attention_score_shape})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char *>(&platform_info))
        .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(5, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(8, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(9, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(10, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({
            {"query_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"key_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"value_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"softmax_axes", ge::AnyValue::CreateFrom<int64_t>(-1)}
        })
        .TilingData(param.get())
        .Workspace(ws_size)
        .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_wrong_mask_008)
{
    uint32_t b = 2048;
    uint32_t n = 3;
    uint32_t s = 49;
    uint32_t h = 32;
    gert::StorageShape query_shape = {{b,n,s,h},{b,n,s,h}};
    gert::StorageShape key_shape={{b,n,s,h},{b,n,s,h}};
    gert::StorageShape value_shape={{b,n+1,s,h},{b,n+1,s,h}};
    gert::StorageShape scale_quant_shape={{1,s},{1,s}};
    gert::StorageShape scale_dequant1_shape={{1,s},{1,s}};
    gert::StorageShape scale_dequant2_shape={{1,h},{1,h}};
    gert::StorageShape bias_quant_shape={{1,s},{1,s}};
    gert::StorageShape bias_dequant1_shape={{1,s},{1,s}};
    gert::StorageShape bias_dequant2_shape={{1,h},{1,h}};
    gert::StorageShape padding_mask1_shape={{1,n,s,s+1},{1,n,s,s+1}};
    gert::StorageShape padding_mask2_shape={{1,n,s,s},{1,n,s,s}};
    gert::StorageShape attention_score_shape={{b,n,s,h},{b,n,s,h}};
    bool query_transpose = false;
    bool key_transpose = false;
    bool value_transpose = false;
    int32_t softmax_axes = -1;	

    std::vector<gert::StorageShape> output_shapes(1,{{b,n,s,h},{b,n,s,h}});
    std::vector<void *> output_shapes_ref(1);
    for(size_t i=0; i<output_shapes.size(); ++i)
    {
        output_shapes_ref[i]=&output_shapes[i];
    }

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 8}
                          })";
    map<string, string> soc_version_infos;
    soc_version_infos.insert(make_pair("Short_SoC_version", "Ascend310P"));
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    
    SwinAttentionScoreQuantCompileInfo compile_info;

    std::string op_type("SwinAttentionScoreQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
        .KernelIONum(2, 1)
        .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
        .Outputs({&compile_info})
        .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
        .NodeIoNum(11, 1)
        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
        .InputShapes(
            {&query_shape, &key_shape, &value_shape, &scale_quant_shape, &scale_dequant1_shape, &scale_dequant2_shape,
            &bias_quant_shape, &bias_dequant1_shape, &bias_dequant2_shape, &padding_mask1_shape, &padding_mask2_shape}
        )
        // .OutputShapes(output_shapes_ref)
        .OutputShapes({&attention_score_shape})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char *>(&platform_info))
        .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(5, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(8, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(9, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(10, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({
            {"query_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"key_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"value_transpose", ge::AnyValue::CreateFrom<bool>(false)},
            {"softmax_axes", ge::AnyValue::CreateFrom<int64_t>(-1)}
        })
        .TilingData(param.get())
        .Workspace(ws_size)
        .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}