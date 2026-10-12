/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <iostream>
#include <cstdint>
#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "../../../op_kernel/arch35/get_shape_tiling_data.h"

using namespace std;

extern "C" __global__ __aicore__ void get_shape(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class GetShapeKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "GetShapeKernelTest SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "GetShapeKernelTest TearDown\n" << endl; }

    void RunGetShape(const vector<vector<int64_t>>& inputShapes, const vector<int32_t>& expectedOutput)
    {
        int64_t inputNum = static_cast<int64_t>(inputShapes.size());
        int64_t totalDimNum = 0;
        for (const auto& s : inputShapes) {
            totalDimNum += static_cast<int64_t>(s.size());
        }

        const int64_t descDataSize = 128;
        const int64_t descDataBytes = descDataSize * sizeof(int64_t);

        int64_t dim = static_cast<int64_t>(inputShapes[0].size());
        int64_t descStructSize = (dim == 0) ? 2 : 1 + dim;
        int64_t headerSize = 1 + inputNum * descStructSize;
        int64_t dataPtrOffset = headerSize * sizeof(int64_t);
        int64_t totalListBytes = dataPtrOffset + inputNum * sizeof(int64_t);

        int64_t totalDataBytes = inputNum * descDataBytes;
        int64_t totalGmBytes = totalListBytes + totalDataBytes;
        size_t outputBytes = GetShapeConst::MAX_TOTAL_DIM * sizeof(int32_t);

        uint8_t* gm = (uint8_t*)AscendC::GmAlloc(totalGmBytes);
        uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputBytes);
        uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
        size_t tilingSize = sizeof(GetShapeTilingData);
        uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

        ASSERT_NE(gm, nullptr);
        ASSERT_NE(y, nullptr);
        ASSERT_NE(workspace, nullptr);
        ASSERT_NE(tiling, nullptr);
        memset(gm, 0, totalGmBytes);
        memset(y, 0, outputBytes);
        memset(tiling, 0, tilingSize);

        uint64_t* listMem = reinterpret_cast<uint64_t*>(gm);
        listMem[0] = static_cast<uint64_t>(dataPtrOffset);
        listMem[1] = static_cast<uint64_t>(dim) | (static_cast<uint64_t>(inputNum) << 32);

        uint8_t* dataRegion = gm + totalListBytes;
        for (int64_t i = 0; i < inputNum; ++i) {
            int64_t curDim = static_cast<int64_t>(inputShapes[i].size());
            uint64_t* shapeSlot = listMem + 1 + i * descStructSize;
            shapeSlot[0] = static_cast<uint64_t>(curDim) | (static_cast<uint64_t>(inputNum) << 32);
            for (int64_t d = 0; d < curDim; ++d) {
                shapeSlot[1 + d] = static_cast<uint64_t>(inputShapes[i][d]);
            }

            uint64_t* dataPtrSlot = reinterpret_cast<uint64_t*>(gm + dataPtrOffset + i * sizeof(int64_t));
            uint64_t dataAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dataRegion + i * descDataBytes));
            *dataPtrSlot = dataAddr;

            int64_t* descData = reinterpret_cast<int64_t*>(dataRegion + i * descDataBytes);
            memset(descData, 0, descDataBytes);
            descData[3] = curDim;
            for (int64_t d = 0; d < curDim; ++d) {
                descData[4 + d] = inputShapes[i][d];
            }
        }

        GetShapeTilingData* tilingData = reinterpret_cast<GetShapeTilingData*>(tiling);
        tilingData->inputNum = static_cast<int32_t>(inputNum);

        uint64_t tilingKey = 0;
        uint32_t numBlocks = 1;

        ICPU_SET_TILING_KEY(tilingKey);
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_RUN_KF(get_shape, numBlocks, gm, y, workspace, tiling);

        int32_t* outPtr = reinterpret_cast<int32_t*>(y);
        for (size_t i = 0; i < expectedOutput.size(); ++i) {
            EXPECT_EQ(outPtr[i], expectedOutput[i]) << "Mismatch at index " << i;
        }

        AscendC::GmFree(gm);
        AscendC::GmFree(y);
        AscendC::GmFree(workspace);
        AscendC::GmFree(tiling);
    }
};

TEST_F(GetShapeKernelTest, SingleInput_3D_Tensor) { RunGetShape({{2, 3, 4}}, {2, 3, 4}); }

TEST_F(GetShapeKernelTest, SingleInput_1D_Tensor) { RunGetShape({{100}}, {100}); }

TEST_F(GetShapeKernelTest, SingleInput_8D_Tensor) { RunGetShape({{1, 2, 3, 4, 5, 6, 7, 8}}, {1, 2, 3, 4, 5, 6, 7, 8}); }

TEST_F(GetShapeKernelTest, MultiInput_2Tensors) { RunGetShape({{2, 3, 4}, {5, 6}}, {2, 3, 4, 5, 6}); }

TEST_F(GetShapeKernelTest, LargeDimValues) { RunGetShape({{1000000, 2000000, 3000000}}, {1000000, 2000000, 3000000}); }
