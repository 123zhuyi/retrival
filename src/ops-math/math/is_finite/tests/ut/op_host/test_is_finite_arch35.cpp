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
 * \file test_is_finite_arch35.cpp
 * \brief
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "is_finite_regbase_tiling.h"
#include "../../../../../math/is_finite/op_kernel/is_finite_apt.cpp"
#include <cstdint>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "../data_utils.h"
#include <iostream>
#include <string>
#endif

using namespace std;

class is_finite_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "is_finite_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "is_finite_test TearDown\n" << endl;
    }
};

TEST_F(is_finite_test, ascend910D1_test_case_half_001)
{
    size_t xByteSize = 64 * sizeof(half);
    size_t yByteSize = 64 * sizeof(bool);
    size_t tiling_data_size = sizeof(IsFiniteRegbaseTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    IsFiniteRegbaseTilingData* tilingDatafromBin = reinterpret_cast<IsFiniteRegbaseTilingData*>(tiling);

    tilingDatafromBin->baseTiling.scheMode = 1;
    tilingDatafromBin->baseTiling.dim0 = 64;
    tilingDatafromBin->baseTiling.blockFormer = 64;
    tilingDatafromBin->baseTiling.blockNum = 1;
    tilingDatafromBin->baseTiling.ubFormer = 6400;
    tilingDatafromBin->baseTiling.ubLoopOfFormerBlock = 1;
    tilingDatafromBin->baseTiling.ubLoopOfTailBlock = 1;
    tilingDatafromBin->baseTiling.ubTailOfFormerBlock = 64;
    tilingDatafromBin->baseTiling.ubTailOfTailBlock = 64;
    tilingDatafromBin->baseTiling.elemNum = 6400;

    ICPU_SET_TILING_KEY(3UL);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/is_finite/is_finite_data ./");
    system("chmod -R 755 ./is_finite_data/");
    system("cd ./is_finite_data/ && rm -rf ./*bin && python3 gen_data.py '(64)' 'float16'");
    ReadFile(path + "/is_finite_data/float16_input_t_is_finite.bin", xByteSize, x, xByteSize);

    auto is_finite_kernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::is_finite<0, 1>(x, y, workspace, tiling);
    };
    ICPU_RUN_KF(is_finite_kernel, numBlocks, x, y, workspace, tiling);

    WriteFile(path + "/is_finite_data/float16_output_t_is_finite.bin", y, yByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);

    int res = system("cd ./is_finite_data/ && python3 compare_data.py 'float16'");
    system("rm -rf is_finite_data");
    ASSERT_EQ(res, 0);
}

TEST_F(is_finite_test, ascend910D1_test_case_bfloat_002)
{
    size_t xByteSize = 64 * sizeof(half);
    size_t yByteSize = 64 * sizeof(bool);
    size_t tiling_data_size = sizeof(IsFiniteRegbaseTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    IsFiniteRegbaseTilingData* tilingDatafromBin = reinterpret_cast<IsFiniteRegbaseTilingData*>(tiling);

    tilingDatafromBin->baseTiling.scheMode = 1;
    tilingDatafromBin->baseTiling.dim0 = 64;
    tilingDatafromBin->baseTiling.blockFormer = 64;
    tilingDatafromBin->baseTiling.blockNum = 1;
    tilingDatafromBin->baseTiling.ubFormer = 6400;
    tilingDatafromBin->baseTiling.ubLoopOfFormerBlock = 1;
    tilingDatafromBin->baseTiling.ubLoopOfTailBlock = 1;
    tilingDatafromBin->baseTiling.ubTailOfFormerBlock = 64;
    tilingDatafromBin->baseTiling.ubTailOfTailBlock = 64;
    tilingDatafromBin->baseTiling.elemNum = 6400;

    ICPU_SET_TILING_KEY(5UL);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/is_finite/is_finite_data ./");
    system("chmod -R 755 ./is_finite_data/");
    system("cd ./is_finite_data/ && rm -rf ./*bin && python3 gen_data.py '(64)' 'bfloat16'");
    ReadFile(path + "/is_finite_data/bfloat16_input_t_is_finite.bin", xByteSize, x, xByteSize);

    auto is_finite_kernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::is_finite<0, 2>(x, y, workspace, tiling);
    };
    ICPU_RUN_KF(is_finite_kernel, numBlocks, x, y, workspace, tiling);
    WriteFile(path + "/is_finite_data/bfloat16_output_t_is_finite.bin", y, yByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);

    int res = system("cd ./is_finite_data/ && python3 compare_data.py 'bfloat16'");
    system("rm -rf is_finite_data");
    ASSERT_EQ(res, 0);
}

TEST_F(is_finite_test, ascend910D1_test_case_float_003)
{
    size_t xByteSize = 64 * sizeof(float);
    size_t yByteSize = 64 * sizeof(bool);
    size_t tiling_data_size = sizeof(IsFiniteRegbaseTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    IsFiniteRegbaseTilingData* tilingDatafromBin = reinterpret_cast<IsFiniteRegbaseTilingData*>(tiling);

    tilingDatafromBin->baseTiling.scheMode = 1;
    tilingDatafromBin->baseTiling.dim0 = 64;
    tilingDatafromBin->baseTiling.blockFormer = 64;
    tilingDatafromBin->baseTiling.blockNum = 1;
    tilingDatafromBin->baseTiling.ubFormer = 6400;
    tilingDatafromBin->baseTiling.ubLoopOfFormerBlock = 1;
    tilingDatafromBin->baseTiling.ubLoopOfTailBlock = 1;
    tilingDatafromBin->baseTiling.ubTailOfFormerBlock = 64;
    tilingDatafromBin->baseTiling.ubTailOfTailBlock = 64;
    tilingDatafromBin->baseTiling.elemNum = 6400;

    ICPU_SET_TILING_KEY(7UL);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/is_finite/is_finite_data ./");
    system("chmod -R 755 ./is_finite_data/");
    system("cd ./is_finite_data/ && rm -rf ./*bin && python3 gen_data.py '(64)' 'float32'");
    ReadFile(path + "/is_finite_data/float32_input_t_is_finite.bin", xByteSize, x, xByteSize);

    auto is_finite_kernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::is_finite<0, 3>(x, y, workspace, tiling);
    };
    ICPU_RUN_KF(is_finite_kernel, numBlocks, x, y, workspace, tiling);

    WriteFile(path + "/is_finite_data/float32_output_t_is_finite.bin", y, yByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);

    int res = system("cd ./is_finite_data/ && python3 compare_data.py 'float32'");
    system("rm -rf is_finite_data");
    ASSERT_EQ(res, 0);
}