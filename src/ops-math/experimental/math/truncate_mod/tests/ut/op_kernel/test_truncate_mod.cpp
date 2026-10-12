/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "../../../op_kernel/truncate_mod.cpp"

using namespace std;

class TruncateModTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "truncate_mod_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./truncate_mod_data/");
    }
    static void TearDownTestCase() { std::cout << "truncate_mod_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string TruncateModTest::rootPath = "../../../../experimental/";
const std::string TruncateModTest::dataPath = rootPath + "math/truncate_mod/tests/ut/op_kernel/truncate_mod_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(TruncateModTest, test_case_int32_1)
{
    uint32_t blockDim = 1;
    system("cd ./truncate_mod_data/ && python3 gen_data.py '(1024)' 'int32'");
    uint32_t dataCount = 1024;
    size_t inputByteSize = dataCount * sizeof(int32_t);

    std::string x1_fileName = "./truncate_mod_data/int32_input_truncate_mod_x1.bin";
    std::string x2_fileName = "./truncate_mod_data/int32_input_truncate_mod_x2.bin";

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));

    ReadFile(x1_fileName, inputByteSize, x1, inputByteSize);
    ReadFile(x2_fileName, inputByteSize, x2, inputByteSize);

    size_t outputByteSize = dataCount * sizeof(int32_t);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(TruncateModTilingData));

    TruncateModTilingData* tilingData = reinterpret_cast<TruncateModTilingData*>(tiling);
    tilingData->coreNum = 1;
    tilingData->bufferNum = 1;
    tilingData->tailElems = 0;
    tilingData->epochs = 1;
    tilingData->epochsForLastCore = 1;
    tilingData->coreLength = dataCount;
    tilingData->tileLength = dataCount;
    tilingData->tailTileLength = 0;
    tilingData->tailTileLengthForLastCore = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = truncate_mod<TRUNCATEMOD_TPL_SCH_MODE_3>;
    ICPU_RUN_KF(func, blockDim, x1, x2, y, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./truncate_mod_data/int32_output_truncate_mod.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x1));
    AscendC::GmFree((void*)(x2));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./truncate_mod_data/ && python3 compare_data.py 'int32'");
}
