/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_exp_segsum_grad.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "exp_segsum_grad_tiling.h"

extern "C" __global__ __aicore__ void exp_segsum_grad(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class exp_segsum_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "exp_segsum_grad_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "exp_segsum_grad_test TearDown\n" << std::endl;
    }
};

TEST_F(exp_segsum_grad_test, test_case_float_1)
{
    // system(
    //     "cp -rf "
    //     "../../../../math/exp_segsum_grad/tests/ut/op_kernel/exp_segsum_grad_data ./");
    // system("chmod -R 755 ./exp_segsum_grad_data/");
    // system("cd ./exp_segsum_grad_data/ && python3 gen_data.py '(1, 1, 2, 4, 4)' 'float32'");

    size_t inputByteSize = 2 * 4 * 4 * sizeof(float);
    size_t outputByteSize = 2 * 4 * sizeof(float);
    size_t tiling_data_size = sizeof(ExpSegsumGradTilingData);
    size_t workspaceSize = 32 * 1024 * 1024;
    uint32_t numBlocks = 2;

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    // std::string fileName1 = "./exp_segsum_grad_data/float32_input1_exp_segsum_grad.bin";
    // std::string fileName2 = "./exp_segsum_grad_data/float32_input2_exp_segsum_grad.bin";

    // ReadFile(fileName1, inputByteSize, x1, inputByteSize);
    // ReadFile(fileName2, inputByteSize, x2, inputByteSize);

    ExpSegsumGradTilingData* tilingDatafromBin = reinterpret_cast<ExpSegsumGradTilingData*>(tiling);

    tilingDatafromBin->needCoreNum = 2;
    tilingDatafromBin->batches = 2;
    tilingDatafromBin->tailDimLength = 2;
    tilingDatafromBin->slideSize = 32;

    tilingDatafromBin->batchStart[0] = 0;
    tilingDatafromBin->batchStart[1] = 1;
    tilingDatafromBin->batchEnd[0] = 1;
    tilingDatafromBin->batchEnd[1] = 2;


    ICPU_SET_TILING_KEY(1);

    ICPU_RUN_KF(exp_segsum_grad, numBlocks, x1, x2, y, workspace, (uint8_t*)(tilingDatafromBin));
    // fileName1 = "./exp_segsum_grad_data/float32_output_exp_segsum_grad.bin";
    // WriteFile(fileName1, y, outputByteSize);

    AscendC::GmFree((void*)(x1));
    AscendC::GmFree((void*)(x2));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    // system("cd ./exp_segsum_grad_data/ && python3 compare_data.py 'float32'");
}