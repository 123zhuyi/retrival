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
 * \file test_is_inf.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_host/is_inf_tiling_def.h"
#include "data_utils.h"
#include "tiling_case_executor.h"

using namespace std;

extern "C" __global__ __aicore__ void is_inf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class IsInfTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "is_inf_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./is_inf_data/");
    }
    static void TearDownTestCase() {
        std::cout << "is_inf_test TearDown" << std::endl;
    }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string IsInfTest::rootPath = "../../../../";
const std::string IsInfTest::dataPath = rootPath + "math/is_inf/tests/ut/op_kernel/is_inf_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b) {
    return (a + b - 1) / b * b;
}

TEST_F(IsInfTest, test_case_float16_1) {
    optiling::IsInfCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("IsInf",
                                              {{{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                              {{{{128, 64}, {128, 64}}, ge::DT_BOOL, ge::FORMAT_ND},},
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./is_inf_data/ && python3 gen_data.py '(128, 64)' 'float16'");
    uint32_t dataCount = 128 * 64;
    size_t inputByteSize = dataCount * sizeof(half);
    std::string fileName = "./is_inf_data/float16_input.bin";;
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, x, inputByteSize);
    size_t outputByteSize = dataCount * sizeof(bool);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(is_inf, tilingInfo.blockNum, x, y, workspace, tiling);

    fileName = "./is_inf_data/float16_output_golden.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./is_inf_data/ && python3 compare_data.py 'float16'");
}

TEST_F(IsInfTest, test_case_float32_1) {
    optiling::IsInfCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("IsInf",
                                              {{{{256, 33}, {256, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {{{{256, 33}, {256, 33}}, ge::DT_BOOL, ge::FORMAT_ND},},
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./is_inf_data/ && python3 gen_data.py '(256, 33)' 'float32'");
    uint32_t dataCount = 256 * 33;
    size_t inputByteSize = dataCount * sizeof(float);
    std::string fileName = "./is_inf_data/float32_input.bin";;
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, x, inputByteSize);
    size_t outputByteSize = dataCount * sizeof(bool);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(is_inf, tilingInfo.blockNum, x, y, workspace, tiling);

    fileName = "./is_inf_data/float32_output_golden.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./is_inf_data/ && python3 compare_data.py 'float32'");
}

TEST_F(IsInfTest, test_case_bfloat16_1) {
    optiling::IsInfCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("IsInf",
                                              {{{{256, 33}, {256, 33}}, ge::DT_BF16, ge::FORMAT_ND},},
                                              {{{{256, 33}, {256, 33}}, ge::DT_BOOL, ge::FORMAT_ND},},
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./is_inf_data/ && python3 gen_data.py '(256, 33)' 'bfloat16'");
    uint32_t dataCount = 256 * 33;
    size_t inputByteSize = dataCount * sizeof(bfloat16_t);
    std::string fileName = "./is_inf_data/bfloat16_input.bin";;
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, x, inputByteSize);
    size_t outputByteSize = dataCount * sizeof(bool);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(is_inf, tilingInfo.blockNum, x, y, workspace, tiling);

    fileName = "./is_inf_data/bfloat16_output_golden.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./is_inf_data/ && python3 compare_data.py 'bfloat16'");
}