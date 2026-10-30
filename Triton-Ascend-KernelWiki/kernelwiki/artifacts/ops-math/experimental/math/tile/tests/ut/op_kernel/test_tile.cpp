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
#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "../../../op_kernel/tile.cpp"

using namespace std;

class TileTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TileTest SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./tile_data/");
    }
    static void TearDownTestCase() { std::cout << "TileTest TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string TileTest::rootPath = "../../../../experimental/";
const std::string TileTest::dataPath = rootPath + "math/tile/tests/ut/op_kernel/tile_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

static void BuildTilingData(
    TileTilingData* tilingData, const std::vector<int32_t>& shape, const std::vector<int32_t>& multiples,
    int32_t elemBytes, int32_t blockDim, int32_t ubSize)
{
    memset(tilingData, 0, sizeof(TileTilingData));
    int32_t ndim = static_cast<int32_t>(shape.size());
    tilingData->numDims = ndim;
    tilingData->elemBytes = elemBytes;
    tilingData->blockDim = blockDim;
    tilingData->ubSize = ubSize;
    tilingData->totalInputElems = 1;
    tilingData->totalOutputElems = 1;
    for (int32_t idx = 0; idx < ndim; idx++) {
        tilingData->inputShape[idx] = shape[idx];
        tilingData->multiples[idx] = multiples[idx];
        tilingData->outputShape[idx] = shape[idx] * multiples[idx];
        tilingData->totalInputElems *= shape[idx];
        tilingData->totalOutputElems *= tilingData->outputShape[idx];
    }
    tilingData->inputStrides[ndim - 1] = 1;
    tilingData->outputStrides[ndim - 1] = 1;
    for (int32_t idx = ndim - 2; idx >= 0; idx--) {
        tilingData->inputStrides[idx] = tilingData->inputStrides[idx + 1] * tilingData->inputShape[idx + 1];
        tilingData->outputStrides[idx] = tilingData->outputStrides[idx + 1] * tilingData->outputShape[idx + 1];
    }
}

TEST_F(TileTest, test_case_float32_2d)
{
    std::vector<int32_t> shape = {2, 3};
    std::vector<int32_t> mult = {3, 2};

    uint8_t* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(sizeof(TileTilingData), 32)));
    TileTilingData* tilingData = reinterpret_cast<TileTilingData*>(tiling);
    BuildTilingData(tilingData, shape, mult, 4, 1, 87381);

    system("cd ./tile_data/ && python3 gen_data.py '2,3' '3,2' float32");
    uint32_t inputCount = static_cast<uint32_t>(tilingData->totalInputElems);
    uint32_t outputCount = static_cast<uint32_t>(tilingData->totalOutputElems);
    size_t inputByteSize = inputCount * sizeof(float);
    size_t outputByteSize = outputCount * sizeof(float);

    uint8_t* x = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(inputByteSize + 32, 32)));
    ReadFile("./tile_data/float32_input_t_tile.bin", inputByteSize, x, inputByteSize);
    uint8_t* y = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(outputByteSize, 32)));
    uint8_t* multGm = static_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(32 * 1024 * 1024));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = tile<TILE_TPL_SCH_MODE_DEFAULT>;
    ICPU_RUN_KF(func, tilingData->blockDim, x, multGm, y, workspace, tiling);

    WriteFile("./tile_data/float32_output_t_tile.bin", y, outputByteSize);

    AscendC::GmFree(static_cast<void*>(x));
    AscendC::GmFree(static_cast<void*>(y));
    AscendC::GmFree(static_cast<void*>(multGm));
    AscendC::GmFree(static_cast<void*>(workspace));
    AscendC::GmFree(static_cast<void*>(tiling));

    system("cd ./tile_data/ && python3 compare_data.py float32_golden_t_tile.bin float32_output_t_tile.bin float32");
}

TEST_F(TileTest, test_case_int32_1d)
{
    std::vector<int32_t> shape = {128};
    std::vector<int32_t> mult = {8};

    uint8_t* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(sizeof(TileTilingData), 32)));
    TileTilingData* tilingData = reinterpret_cast<TileTilingData*>(tiling);
    BuildTilingData(tilingData, shape, mult, 4, 1, 87381);

    system("cd ./tile_data/ && python3 gen_data.py '128' '8' int32");
    uint32_t inputCount = static_cast<uint32_t>(tilingData->totalInputElems);
    uint32_t outputCount = static_cast<uint32_t>(tilingData->totalOutputElems);
    size_t inputByteSize = inputCount * sizeof(int32_t);
    size_t outputByteSize = outputCount * sizeof(int32_t);

    uint8_t* x = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(inputByteSize + 32, 32)));
    ReadFile("./tile_data/int32_input_t_tile.bin", inputByteSize, x, inputByteSize);
    uint8_t* y = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(outputByteSize, 32)));
    uint8_t* multGm = static_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(32 * 1024 * 1024));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = tile<TILE_TPL_SCH_MODE_DEFAULT>;
    ICPU_RUN_KF(func, tilingData->blockDim, x, multGm, y, workspace, tiling);

    WriteFile("./tile_data/int32_output_t_tile.bin", y, outputByteSize);

    AscendC::GmFree(static_cast<void*>(x));
    AscendC::GmFree(static_cast<void*>(y));
    AscendC::GmFree(static_cast<void*>(multGm));
    AscendC::GmFree(static_cast<void*>(workspace));
    AscendC::GmFree(static_cast<void*>(tiling));

    system("cd ./tile_data/ && python3 compare_data.py int32_golden_t_tile.bin int32_output_t_tile.bin int32");
}

TEST_F(TileTest, test_case_float32_large_inner)
{
    std::vector<int32_t> shape = {1, 1, 25000};
    std::vector<int32_t> mult = {3, 3, 12};

    uint8_t* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(sizeof(TileTilingData), 32)));
    TileTilingData* tilingData = reinterpret_cast<TileTilingData*>(tiling);
    BuildTilingData(tilingData, shape, mult, 4, 1, 87381);

    system("cd ./tile_data/ && python3 gen_data.py '1,1,25000' '3,3,12' float32");
    uint32_t inputCount = static_cast<uint32_t>(tilingData->totalInputElems);
    uint32_t outputCount = static_cast<uint32_t>(tilingData->totalOutputElems);
    size_t inputByteSize = inputCount * sizeof(float);
    size_t outputByteSize = outputCount * sizeof(float);

    uint8_t* x = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(inputByteSize + 32, 32)));
    ReadFile("./tile_data/float32_input_t_tile.bin", inputByteSize, x, inputByteSize);
    uint8_t* y = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(outputByteSize, 32)));
    uint8_t* multGm = static_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(32 * 1024 * 1024));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = tile<TILE_TPL_SCH_MODE_DEFAULT>;
    ICPU_RUN_KF(func, tilingData->blockDim, x, multGm, y, workspace, tiling);

    WriteFile("./tile_data/float32_output_t_tile.bin", y, outputByteSize);

    AscendC::GmFree(static_cast<void*>(x));
    AscendC::GmFree(static_cast<void*>(y));
    AscendC::GmFree(static_cast<void*>(multGm));
    AscendC::GmFree(static_cast<void*>(workspace));
    AscendC::GmFree(static_cast<void*>(tiling));

    system("cd ./tile_data/ && python3 compare_data.py float32_golden_t_tile.bin float32_output_t_tile.bin float32");
}
