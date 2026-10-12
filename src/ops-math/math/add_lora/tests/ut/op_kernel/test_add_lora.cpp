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
#include "../../../op_host/add_lora_tiling.h"

#include <cstdint>

using namespace std;
extern "C" __global__ __aicore__ void add_lora(GM_ADDR y, GM_ADDR x, GM_ADDR weightB,
    GM_ADDR indices, GM_ADDR weightA, GM_ADDR y_out, GM_ADDR workspace, GM_ADDR tiling);
class add_lora_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "add_lora SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "add_lora TearDown\n" << endl;
  }
};

TEST_F(add_lora_test, test_add_lora_0)
{
    size_t Batch = 1;
    size_t H2 = 4096;
    size_t H1 = 16;
    size_t weight = 2;
    size_t layer =1;
    size_t R = 16;
    size_t yInputSize = Batch * H2 * sizeof(half);
    size_t xInputSize = Batch * H1 * sizeof(half);
    size_t weightAFileSize = weight *layer *H1*R*sizeof(half);
    size_t weightBFileSize =  weight *layer *R*H2*sizeof(half);
    size_t indiceFileSize = Batch* sizeof(int32_t);
    size_t tilingDataSize = sizeof(AddLoraTilingData);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yInputSize);
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xInputSize);
    uint8_t *weightA = (uint8_t *)AscendC::GmAlloc(weightAFileSize);
    uint8_t *weightB = (uint8_t *)AscendC::GmAlloc(weightBFileSize);
    uint8_t *indice = (uint8_t *)AscendC::GmAlloc(indiceFileSize);

    uint8_t *yOut = (uint8_t *)AscendC::GmAlloc(yInputSize);

    uint64_t tilingKey = 100001;
    uint32_t numBlocks = 20;
    size_t workspaceFileSize = 16781184;

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    AddLoraTilingData* tilingDatafromBin = reinterpret_cast<AddLoraTilingData*>(tiling);

    tilingDatafromBin->usedCoreNum = numBlocks;
    tilingDatafromBin->layer = 1;
    tilingDatafromBin->batch = Batch;
    tilingDatafromBin->H1 = H1;
    tilingDatafromBin->H2 = H2;
    tilingDatafromBin->R = R;
    tilingDatafromBin->wBatch = weight;
    tilingDatafromBin->layer_idx = 0;
    tilingDatafromBin->y_offset = 0;
    tilingDatafromBin->scale = 2.0;
    tilingDatafromBin->y_slice_size= H2;
    tilingDatafromBin->addLoraFlag = 1;
    tilingDatafromBin->taskNumPerCore = Batch/ (numBlocks *2);
    char * path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(add_lora, numBlocks, y, x, weightB, indice, weightA,
                yOut, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightA);
    AscendC::GmFree((void *)weightB);
    AscendC::GmFree((void *)indice);
    AscendC::GmFree((void *)yOut);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(add_lora_test, test_add_lora_1)
{
    size_t Batch = 100;
    size_t H2 = 4096;
    size_t H1 = 16;
    size_t weight = 2;
    size_t layer =1;
    size_t R = 16;
    size_t yInputSize = Batch * H2 * sizeof(half);
    size_t xInputSize = Batch * H1 * sizeof(half);
    size_t weightAFileSize = weight *layer *H1*R*sizeof(half);
    size_t weightBFileSize =  weight *layer *R*H2*sizeof(half);
    size_t indiceFileSize = Batch* sizeof(int32_t);
    size_t tilingDataSize = sizeof(AddLoraTilingData);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yInputSize);
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xInputSize);
    uint8_t *weightA = (uint8_t *)AscendC::GmAlloc(weightAFileSize);
    uint8_t *weightB = (uint8_t *)AscendC::GmAlloc(weightBFileSize);
    uint8_t *indice = (uint8_t *)AscendC::GmAlloc(indiceFileSize);

    uint8_t *yOut = (uint8_t *)AscendC::GmAlloc(yInputSize);

    uint64_t tilingKey = 100000;
    uint32_t numBlocks = 20;
    size_t workspaceFileSize = 16781184;

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    AddLoraTilingData* tilingDatafromBin = reinterpret_cast<AddLoraTilingData*>(tiling);

    tilingDatafromBin->usedCoreNum = numBlocks;
    tilingDatafromBin->layer = 1;
    tilingDatafromBin->batch = Batch;
    tilingDatafromBin->H1 = H1;
    tilingDatafromBin->H2 = H2;
    tilingDatafromBin->R = R;
    tilingDatafromBin->wBatch = weight;
    tilingDatafromBin->layer_idx = 0;
    tilingDatafromBin->y_offset = 0;
    tilingDatafromBin->scale = 2.0;
    tilingDatafromBin->y_slice_size= H2;
    tilingDatafromBin->addLoraFlag = 1;
    tilingDatafromBin->taskNumPerCore = Batch/ (numBlocks *2);
    char * path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(add_lora, numBlocks, y, x, weightB, indice, weightA,
                yOut, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightA);
    AscendC::GmFree((void *)weightB);
    AscendC::GmFree((void *)indice);
    AscendC::GmFree((void *)yOut);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(add_lora_test, test_add_lora_2)
{
    size_t Batch = 100;
    size_t H2 = 4096;
    size_t H1 = 16;
    size_t weight = 2;
    size_t layer =1;
    size_t R = 16;
    size_t yInputSize = Batch * H2 * sizeof(half);
    size_t xInputSize = Batch * H1 * sizeof(half);
    size_t weightAFileSize = weight *layer *H1*R*sizeof(half);
    size_t weightBFileSize =  weight *layer *R*H2*sizeof(half);
    size_t indiceFileSize = Batch* sizeof(int32_t);
    size_t tilingDataSize = sizeof(AddLoraTilingData);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yInputSize);
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xInputSize);
    uint8_t *weightA = (uint8_t *)AscendC::GmAlloc(weightAFileSize);
    uint8_t *weightB = (uint8_t *)AscendC::GmAlloc(weightBFileSize);
    uint8_t *indice = (uint8_t *)AscendC::GmAlloc(indiceFileSize);

    uint8_t *yOut = (uint8_t *)AscendC::GmAlloc(yInputSize);

    uint64_t tilingKey = 100000;
    uint32_t numBlocks = 20;
    size_t workspaceFileSize = 16781184;

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    AddLoraTilingData* tilingDatafromBin = reinterpret_cast<AddLoraTilingData*>(tiling);

    tilingDatafromBin->usedCoreNum = numBlocks;
    tilingDatafromBin->layer = 1;
    tilingDatafromBin->batch = Batch;
    tilingDatafromBin->H1 = H1;
    tilingDatafromBin->H2 = H2;
    tilingDatafromBin->R = R;
    tilingDatafromBin->wBatch = weight;
    tilingDatafromBin->layer_idx = 0;
    tilingDatafromBin->y_offset = 0;
    tilingDatafromBin->scale = 2.0;
    tilingDatafromBin->y_slice_size= H2;
    tilingDatafromBin->addLoraFlag = 0;
    tilingDatafromBin->taskNumPerCore = Batch/ (numBlocks *2);
    char * path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(add_lora, numBlocks, y, x, weightB, indice, weightA,
                yOut, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightA);
    AscendC::GmFree((void *)weightB);
    AscendC::GmFree((void *)indice);
    AscendC::GmFree((void *)yOut);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(add_lora_test, test_add_lora_3)
{
    size_t Batch = 1;
    size_t H2 = 4096;
    size_t H1 = 16;
    size_t weight = 2;
    size_t layer =1;
    size_t R = 16;
    size_t yInputSize = Batch * H2 * sizeof(half);
    size_t xInputSize = Batch * H1 * sizeof(half);
    size_t weightAFileSize = weight *layer *H1*R*sizeof(half);
    size_t weightBFileSize =  weight *layer *R*H2*sizeof(half);
    size_t indiceFileSize = Batch* sizeof(int32_t);
    size_t tilingDataSize = sizeof(AddLoraTilingData);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yInputSize);
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xInputSize);
    uint8_t *weightA = (uint8_t *)AscendC::GmAlloc(weightAFileSize);
    uint8_t *weightB = (uint8_t *)AscendC::GmAlloc(weightBFileSize);
    uint8_t *indice = (uint8_t *)AscendC::GmAlloc(indiceFileSize);

    uint8_t *yOut = (uint8_t *)AscendC::GmAlloc(yInputSize);

    uint64_t tilingKey = 100001;
    uint32_t numBlocks = 20;
    size_t workspaceFileSize = 16781184;

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    AddLoraTilingData* tilingDatafromBin = reinterpret_cast<AddLoraTilingData*>(tiling);

    tilingDatafromBin->usedCoreNum = numBlocks;
    tilingDatafromBin->layer = 1;
    tilingDatafromBin->batch = Batch;
    tilingDatafromBin->H1 = H1;
    tilingDatafromBin->H2 = H2;
    tilingDatafromBin->R = R;
    tilingDatafromBin->wBatch = weight;
    tilingDatafromBin->layer_idx = 0;
    tilingDatafromBin->y_offset = 0;
    tilingDatafromBin->scale = 2.0;
    tilingDatafromBin->y_slice_size= H2;
    tilingDatafromBin->addLoraFlag = 0;
    tilingDatafromBin->taskNumPerCore = Batch/ (numBlocks *2);
    char * path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(add_lora, numBlocks, y, x, weightB, indice, weightA,
                yOut, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightA);
    AscendC::GmFree((void *)weightB);
    AscendC::GmFree((void *)indice);
    AscendC::GmFree((void *)yOut);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}