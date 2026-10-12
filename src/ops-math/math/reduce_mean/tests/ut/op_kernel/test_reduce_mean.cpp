/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_reduce_mean.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../data_utils.h"
#include "reduce_mean.cpp"
#include "../tiling_test_help.h"
#include "../atvoss_reduce_op_test_help.h"

using namespace std;

class reduce_mean_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "reduce_mean_test SetUp \n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "reduce_mean_test TearDown\n" << endl;
    }
};

static bool CompareData()
{
    std::string cmd = "cd ./reduce_mean_data/ && python3 verify.py y.bin golden.bin ";
    return system(cmd.c_str()) == 0;
}

static void InitEnv()
{
    system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/reduce_mean/reduce_mean_data ./");
    system("chmod -R 755 ./reduce_mean_data/");
    system("cd ./reduce_mean_data/ && rm -rf ./*bin");
    system("cd ./reduce_mean_data/ && python3 gen_data.py");
}

template <std::size_t Index>
void InvokeReduceMeanKernel(const TilingTestHelp::TilingInfo& tilingInfo, GM_ADDR x, GM_ADDR axes, GM_ADDR y,
                            GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (Index < CommonReduceParams.size()) { // 没有额外模板参数的用CommonReduceParams，有额外模板参数用对应算子的Params
        // 没有额外模板参数作为tilingkey用COMMON_REDUCE_PARAM
        // 有额外模板参数作为tilingkey用CUSTOM_REDUCE_PARAM
        if (tilingInfo.tilingKey == GET_TPL_TILING_KEY(COMMON_REDUCE_PARAM(Index))) {
            ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
            std::cout << "tilingInfo.tilingKey:" << tilingInfo.tilingKey << std::endl;
            std::cout << __func__ << " Params:" << ToString(CommonReduceParams[Index]) << std::endl;
            auto func = reduce_mean<COMMON_REDUCE_PARAM(Index)>;
            AscendC::SetKernelMode(KernelMode::AIV_MODE);
            ICPU_RUN_KF(func, tilingInfo.blockNum, x, axes, y, workspace, tiling);
        } else {
            InvokeReduceMeanKernel<Index + 1>(tilingInfo, x, axes, y, workspace, tiling);
        }
    }
}

//template <std::size_t Index, typename DtypeX>
//void InvokeReduceMeanOp(const TilingTestHelp::TilingInfo& tilingInfo, GM_ADDR x, GM_ADDR axes, GM_ADDR y,
//                            GM_ADDR workspace, GM_ADDR tiling)
//{
//    if constexpr (Index < CommonReduceParams.size()) { // 没有额外模板参数的用CommonReduceParams，有额外模板参数用对应算子的Params
//        if (tilingInfo.tilingKey == GET_TPL_TILING_KEY(COMMON_REDUCE_PARAM(Index))) {
//            ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
//            auto func = reduce_mean_op<COMMON_REDUCE_PARAM(Index), DtypeX>;
//            ICPU_RUN_KF(func, tilingInfo.blockNum, x, axes, y, workspace, tiling);
//        } else {
//            InvokeReduceMeanOp<Index + 1, DtypeX>(tilingInfo, x, axes, y, workspace, tiling);
//        }
//    }
//}

TEST_F(reduce_mean_test, test_case_0)
{
    uint32_t numBlocks = 4;
    uint32_t dim0 = 4;
    uint32_t dim1 = 64;
    std::vector<int64_t> input{4, 64};
    std::vector<int64_t> output{4};
    std::vector<int64_t> intputAxes{1};
    auto inDType = DT_FLOAT;
    TilingTestHelp::ParamManager manager;
    manager.Init(2, 1);
    manager.AddInputShape(input, inDType, FORMAT_ND);
    manager.AddConstInput(intputAxes);
    manager.AddOutputShape(output, inDType, FORMAT_ND);
    TilingTestHelp::TilingInfo tilingInfo;
    // UT_SOC_VERSION pass by UT framework, such as "Ascend910D1"
    EXPECT_EQ(DoTiling("ReduceMean", manager, tilingInfo, UT_SOC_VERSION, "{}"), true);
    ASSERT_TRUE(tilingInfo.tilingDataSize > 0);

    size_t xSize = TilingTestHelp::ShapeSize(input) * sizeof(float);
    size_t ySize = TilingTestHelp::ShapeSize(output) * sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* axes = (uint8_t*)AscendC::GmAlloc(sizeof(int32_t));  // axes没赋值, 用的tiling参数
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes.front());
    InitEnv();

    char* cpath = get_current_dir_name();
    string path(cpath);
    free(cpath);
    ReadFile(path + "/reduce_mean_data/x.bin", xSize, x, xSize);
    InvokeReduceMeanKernel<0>(tilingInfo, x, axes, y, workspace, tilingInfo.tilingData.get());
    WriteFile(path + "/reduce_mean_data/y.bin", y, ySize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)axes);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);

    EXPECT_EQ(CompareData(), true);
}

//TEST_F(reduce_mean_test, test_case_1)
//{
//    uint32_t numBlocks = 4;
//    uint32_t dim0 = 4;
//    uint32_t dim1 = 64;
//    std::vector<int64_t> input{4, 64};
//    std::vector<int64_t> output{4};
//    std::vector<int64_t> intputAxes{1};
//    auto inDType = DT_FLOAT;
//    TilingTestHelp::ParamManager manager;
//    manager.Init(2, 1);
//    manager.AddInputShape(input, inDType, FORMAT_ND);
//    manager.AddConstInput(intputAxes);
//    manager.AddOutputShape(output, inDType, FORMAT_ND);
//    TilingTestHelp::TilingInfo tilingInfo;
//    // UT_SOC_VERSION pass by UT framework, such as "Ascend910D1"
//    EXPECT_EQ(DoTiling("ReduceMean", manager, tilingInfo, UT_SOC_VERSION, "{}"), true);
//    ASSERT_TRUE(tilingInfo.tilingDataSize > 0);
//
//    size_t xSize = TilingTestHelp::ShapeSize(input) * sizeof(float);
//    size_t ySize = TilingTestHelp::ShapeSize(output) * sizeof(float);
//
//    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
//    uint8_t* axes = (uint8_t*)AscendC::GmAlloc(sizeof(int32_t));  // axes没赋值, 用的tiling参数
//    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
//    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes.front());
//    InitEnv();
//
//    char* cpath = get_current_dir_name();
//    string path(cpath);
//    free(cpath);
//    ReadFile(path + "/reduce_mean_data/x.bin", xSize, x, xSize);
//    InvokeReduceMeanOp<0, float>(tilingInfo, x, axes, y, workspace, tilingInfo.tilingData.get());
//    WriteFile(path + "/reduce_mean_data/y.bin", y, ySize);
//
//    AscendC::GmFree((void*)x);
//    AscendC::GmFree((void*)axes);
//    AscendC::GmFree((void*)y);
//    AscendC::GmFree((void*)workspace);
//
//    EXPECT_EQ(CompareData(), true);
//}