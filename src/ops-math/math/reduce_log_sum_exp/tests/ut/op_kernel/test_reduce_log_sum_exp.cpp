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
 * \file test_reduce_log_sum_exp.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../data_utils.h"
#include "reduce_log_sum_exp.cpp"

using namespace std;

class reduce_log_sum_exp_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "reduce_log_sum_exp_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "reduce_log_sum_exp_test TearDown\n" << endl;
    }
};

static bool CompareData()
{
    std::string cmd = "cd ./reduce_log_sum_exp_data/ && python3 verify.py y.bin golden.bin ";
    return system(cmd.c_str()) == 0;
}

static void InitEnv(int32_t dimA, int32_t dimR, int32_t axes)
{
    system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/reduce_log_sum_exp/reduce_log_sum_exp_data ./");
    system("chmod -R 755 ./reduce_log_sum_exp_data/");
    system("cd ./reduce_log_sum_exp_data/ && rm -rf ./*bin");
    char buffer[1024] = {0};
    sprintf_s(buffer, "cd ./reduce_log_sum_exp_data/ && python3 gen_data.py %d %d %d", dimA, dimR, axes);
    system(buffer);
}

TEST_F(reduce_log_sum_exp_test, test_case_0)
{
    uint64_t tilingKey = 2571;
    uint32_t numBlocks = 4;
    uint32_t dimA = 4;
    uint32_t dimR = 64;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t xSize = static_cast<size_t>(dimA) * static_cast<size_t>(dimR) * sizeof(float);
    size_t ySize = dimA * sizeof(float);

    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* axes = (uint8_t*)AscendC::GmAlloc(sizeof(int32_t));  // axes没赋值, 用的tiling参数
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ReduceOpTilingData));

    ReduceOpTilingData* tilingData = reinterpret_cast<ReduceOpTilingData*>(tiling);
    tilingData->factorACntPerCore = 1;
    tilingData->factorATotalCnt = 4;
    tilingData->ubFactorA = 1;
    tilingData->factorRCntPerCore = 1;
    tilingData->factorRTotalCnt = 1;
    tilingData->ubFactorR = 1;
    tilingData->groupR = 1;
    tilingData->outSize = 4;
    tilingData->basicBlock = 51200;
    tilingData->coreNum = 64;
    tilingData->meanVar = 0.015625;
    tilingData->shape[0] = 4;
    tilingData->shape[1] = 64;
    tilingData->stride[0] = 64;
    tilingData->stride[1] = 1;
    tilingData->dstStride[0] = 1;
    tilingData->dstStride[1] = 1;

    InitEnv(dimA, dimR, 1);
    char* cpath = get_current_dir_name();
    string path(cpath);
    free(cpath);
    ReadFile(path + "/reduce_log_sum_exp_data/x.bin", xSize, x, xSize);

    ICPU_SET_TILING_KEY(tilingKey);
    auto reduce_log_sum_exp_func = [](GM_ADDR x, GM_ADDR axes, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        reduce_log_sum_exp<11, 10, 0>(x, axes, y, workspace, tiling);
    };
    ICPU_RUN_KF(reduce_log_sum_exp_func, numBlocks, x, axes, y, workspace, tiling);
    WriteFile(path + "/reduce_log_sum_exp_data/y.bin", y, ySize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)axes);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    EXPECT_EQ(CompareData(), true);
}