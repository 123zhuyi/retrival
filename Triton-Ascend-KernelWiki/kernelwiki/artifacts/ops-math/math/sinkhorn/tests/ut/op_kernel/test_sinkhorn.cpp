/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include "../../../op_host/sinkhorn_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void sinkhorn(GM_ADDR cost, GM_ADDR p, GM_ADDR workspace, GM_ADDR tiling);

class SinkhornTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "SinkhornTest SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "SinkhornTest TearDown\n" << endl;
  }
};

template<typename T>
void checkTotalP(T *pp, int shapeSize) {
  float totalP = 0;
  for(int i = 0; i < shapeSize; i++) {
    totalP += (float)(pp[i]);
  }
  EXPECT_NEAR(totalP, 1.0f, 0.01f);
}

TEST_F(SinkhornTest, sinkhorn_float_48_2) {
  size_t shapeSize = 48 * 2;
  size_t inputCostByteSize = shapeSize * sizeof(float);
  size_t outputPByteSize = shapeSize * sizeof(float);
  size_t tilingDataSize = sizeof(SinkhornTilingDataUT);

  uint8_t* cost = (uint8_t*)AscendC::GmAlloc(inputCostByteSize);
  uint8_t* p = (uint8_t*)AscendC::GmAlloc(outputPByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(2*16*1024*1024); // 280 workspace
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t numBlocks = 1;

  float *fp = (float *)cost;
  for(int i = 0; i < shapeSize; i++) {
    fp[i] = 1.0f;
  }

  SinkhornTilingDataUT* tilingData = reinterpret_cast<SinkhornTilingDataUT*>(tiling);

  tilingData->formerNum = 1;              // former 数量
  tilingData->formerRow = 48;             // former cost行数
  tilingData->formerLength = 96;          // former cost总长

  tilingData->formerTileNum = 1;          // former Tile数量
  tilingData->formerLastTileRow = 48;     // fomer last Tile行数
  tilingData->formerLastTileLength = 96;  // fomer last Tile长度

  tilingData->tailNum = 0;                // tail 数量
  tilingData->tailRow = 0;                // tail cost行数
  tilingData->tailLength = 0;             // tail cost总长

  tilingData->tailTileNum = 0;            // tail Tile数量
  tilingData->tailLastTileRow = 0;        // tail last Tile行数
  tilingData->tailLastTileLength = 0;     // tail last Tile长度

  tilingData->tileRow = 1959;             // Tile行数(非Last)
  tilingData->tileLength = 3918;          // Tile长度(非Last)

  tilingData->totalRow = 48;              // 总行数
  tilingData->totalCol = 2;               // 总列数
  tilingData->totalColAligned = 8;        // 对齐后的总列数

  tilingData->tol = 0.0001;               // 误差

  ICPU_SET_TILING_KEY(0); // float16 tilingKey = 0
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(sinkhorn, numBlocks, cost, p, workspace, (uint8_t*)(tilingData));

  checkTotalP((float *)p, shapeSize);

  AscendC::GmFree(cost);
  AscendC::GmFree(p);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
}

TEST_F(SinkhornTest, sinkhorn_float_8_2) {
  size_t shapeSize = 8 * 2;
  size_t inputCostByteSize = shapeSize * sizeof(float);
  size_t outputPByteSize = shapeSize * sizeof(float);
  size_t tilingDataSize = sizeof(SinkhornTilingDataUT);

  uint8_t* cost = (uint8_t*)AscendC::GmAlloc(inputCostByteSize);
  uint8_t* p = (uint8_t*)AscendC::GmAlloc(outputPByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(2*16*1024*1024); // 280 workspace
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t numBlocks = 1;

  float *fp = (float *)cost;
  float testCost[] = {45.0f, 48.0f, 65.0f, 68.0f, 68.0f, 10.0f, 84.0f, 22.0f, 37.0f, 71.0f, 13.0f, 59.0f, 66.0f, 40.0f, 47.0f, 82.0f};

  for (int i = 0; i < shapeSize; i++) {
    fp[i] = testCost[i];
  }

  SinkhornTilingDataUT* tilingData = reinterpret_cast<SinkhornTilingDataUT*>(tiling);

  tilingData->formerNum = 1;              // former 数量
  tilingData->formerRow = 8;              // former cost行数
  tilingData->formerLength = 16;          // former cost总长

  tilingData->formerTileNum = 3;          // former Tile数量
  tilingData->formerLastTileRow = 2;     // fomer last Tile行数
  tilingData->formerLastTileLength = 4;  // fomer last Tile长度

  tilingData->tailNum = 0;                // tail 数量
  tilingData->tailRow = 0;                // tail cost行数
  tilingData->tailLength = 0;             // tail cost总长

  tilingData->tailTileNum = 0;            // tail Tile数量
  tilingData->tailLastTileRow = 0;        // tail last Tile行数
  tilingData->tailLastTileLength = 0;     // tail last Tile长度

  tilingData->tileRow = 3;             // Tile行数(非Last)
  tilingData->tileLength = 6;          // Tile长度(非Last)

  tilingData->totalRow = 8;              // 总行数
  tilingData->totalCol = 2;               // 总列数
  tilingData->totalColAligned = 8;        // 对齐后的总列数

  tilingData->tol = 0.0001;               // 误差

  ICPU_SET_TILING_KEY(0); // float16 tilingKey = 0
  ICPU_RUN_KF(sinkhorn, numBlocks, cost, p, workspace, (uint8_t*)(tilingData));
  checkTotalP((float *)p, shapeSize);

  AscendC::GmFree(cost);
  AscendC::GmFree(p);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
}

TEST_F(SinkhornTest, sinkhorn_float16_48_2) {
  size_t shapeSize = 48 * 2;
  size_t inputCostByteSize = shapeSize * sizeof(float);
  size_t outputPByteSize = shapeSize * sizeof(float);
  size_t tilingDataSize = sizeof(SinkhornTilingDataUT);

  uint8_t* cost = (uint8_t*)AscendC::GmAlloc(inputCostByteSize);
  uint8_t* p = (uint8_t*)AscendC::GmAlloc(outputPByteSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(2*16*1024*1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t numBlocks = 1;

  half *fp = (half *)cost;
  for(int i = 0; i < shapeSize; i++) {
    fp[i] = (half)1.0f;
  }

  SinkhornTilingDataUT* tilingData = reinterpret_cast<SinkhornTilingDataUT*>(tiling);

  tilingData->formerNum = 1;              // former 数量
  tilingData->formerRow = 48;             // former cost行数
  tilingData->formerLength = 96;          // former cost总长

  tilingData->formerTileNum = 1;          // former Tile数量
  tilingData->formerLastTileRow = 48;     // fomer last Tile行数
  tilingData->formerLastTileLength = 96;  // fomer last Tile长度

  tilingData->tailNum = 0;                // tail 数量
  tilingData->tailRow = 0;                // tail cost行数
  tilingData->tailLength = 0;             // tail cost总长

  tilingData->tailTileNum = 0;            // tail Tile数量
  tilingData->tailLastTileRow = 0;        // tail last Tile行数
  tilingData->tailLastTileLength = 0;     // tail last Tile长度

  tilingData->tileRow = 1959;             // Tile行数(非Last)
  tilingData->tileLength = 3918;          // Tile长度(非Last)

  tilingData->totalRow = 48;              // 总行数
  tilingData->totalCol = 2;               // 总列数
  tilingData->totalColAligned = 16;       // 对齐后的总列数

  tilingData->tol = 0.0001;               // 误差

  ICPU_SET_TILING_KEY(1); // float16 tilingKey = 1
  ICPU_RUN_KF(sinkhorn, numBlocks, cost, p, workspace, (uint8_t*)(tilingData));
  // checkTotalP((half *)p, shapeSize);

  AscendC::GmFree(cost);
  AscendC::GmFree(p);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
}