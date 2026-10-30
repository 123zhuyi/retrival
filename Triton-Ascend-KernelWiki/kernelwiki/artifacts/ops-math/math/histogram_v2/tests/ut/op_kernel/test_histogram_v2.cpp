/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "../../../op_host/histogram_v2_tiling.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void histogram_v2(GM_ADDR self, GM_ADDR min, GM_ADDR max, GM_ADDR binsCount, GM_ADDR workspace, GM_ADDR tiling);
class histogram_v2_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        cout << "histogram_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "histogram_v2_test TearDown\n" << endl;
    }
};

HistogramV2TilingData* GetTilingData(uint64_t tilingKey, uint8_t *tiling, uint32_t numBlocks) {
    int64_t totalLength = 128;
    int64_t bins = 1;
    HistogramV2TilingData* tilingData = reinterpret_cast<HistogramV2TilingData*>(tiling);

    // 常量定义
    int64_t SIZE_OF_FP32 = 4;
    int64_t BYTE_BLOCK = 32;

    int64_t HISTOGRAM_V2_FP32 = 0;
    int64_t HISTOGRAM_V2_INT32 = 1;
    int64_t HISTOGRAM_V2_NOT_SUPPORT = -1;

    int64_t UB_SELF_LENGTH = 16320;
    int64_t UB_BINS_LENGTH = 16320;
    // tiling数据定义
    int64_t binsAligned;

    int64_t formerNum;
    int64_t formerLength;
    int64_t formerLengthAligned;
    int64_t tailLength;
    int64_t tailLengthAligned;

    int64_t formerTileNum;
    int64_t formerTileDataLength;
    int64_t formerTileLeftDataLength;
    int64_t formerTileLeftDataLengthAligned;

    int64_t tailTileNum;
    int64_t tailTileDataLength;
    int64_t tailTileLeftDataLength;
    int64_t tailTileLeftDataLengthAligned;
    
    //中间数据定义
    int64_t tailNum;

    //数据分核
    int64_t coreNum = numBlocks;
    int64_t alignNum = BYTE_BLOCK / SIZE_OF_FP32;
    formerNum = 1;
    tailNum = coreNum - formerNum;
    tailLength = totalLength / coreNum;
    formerLength = totalLength - (tailLength * coreNum) + tailLength;
    formerLengthAligned = ((formerLength + alignNum -1) / alignNum) * alignNum;
    tailLengthAligned = ((tailLength + alignNum -1) / alignNum) * alignNum;

    //核上数据切分
    int64_t tileLength = UB_SELF_LENGTH;
    if (tilingKey == 5) {
        tileLength = tileLength / 2;
    }
    formerTileNum = formerLength / tileLength;
    formerTileDataLength = tileLength;
    formerTileLeftDataLength = formerLength - formerTileNum * formerTileDataLength;
    formerTileLeftDataLengthAligned = ((formerTileLeftDataLength + alignNum -1) / alignNum) * alignNum;

    tailTileNum = tailLength / tileLength;
    tailTileDataLength = tileLength;
    tailTileLeftDataLength = tailLength - tailTileNum * tailTileDataLength;
    tailTileLeftDataLengthAligned = ((tailTileLeftDataLength + alignNum -1) / alignNum) * alignNum;

    //设置tilingData
    tilingData->bins = bins;
    tilingData->ubBinsLength = UB_BINS_LENGTH;
    tilingData->formerNum = formerNum;
    tilingData->formerLength = formerLength;
    tilingData->formerLengthAligned = formerLengthAligned;
    tilingData->tailLength = tailLength;
    tilingData->tailLengthAligned = tailLengthAligned;

    tilingData->formerTileNum = formerTileNum;
    tilingData->formerTileDataLength = formerTileDataLength;
    tilingData->formerTileLeftDataLength = formerTileLeftDataLength;
    tilingData->formerTileLeftDataLengthAligned = formerTileLeftDataLengthAligned;
    
    tilingData->tailTileNum = tailTileNum;
    tilingData->tailTileDataLength = tailTileDataLength;
    tilingData->tailTileLeftDataLength = tailTileLeftDataLength;
    tilingData->tailTileLeftDataLengthAligned = tailTileLeftDataLengthAligned;
    return tilingData;
}

TEST_F(histogram_v2_test, test_case_0) {
    int64_t totalLength = 128;
    int64_t bins = 1;

    // inputs
    size_t self_size = totalLength * sizeof(float);
    size_t min_size = sizeof(float);
    size_t max_size = sizeof(float);
    size_t binsCount_size = bins * sizeof(float);
    size_t tiling_data_size = sizeof(HistogramV2TilingData);

    uint8_t *self = (uint8_t*)AscendC::GmAlloc(self_size);
    uint8_t *min = (uint8_t*)AscendC::GmAlloc(min_size);
    uint8_t *max = (uint8_t*)AscendC::GmAlloc(max_size);
    uint8_t *binsCount = (uint8_t*)AscendC::GmAlloc(binsCount_size);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1; //cpu模拟使用单核
    system("cp -r ../../../../math/histogram_v2/tests/ut/op_kernel/histogram_v2_data ./");
    system("chmod -R 755 ./histogram_v2_data/");
    system("cd ./histogram_v2_data/ && rm -rf ./*bin");
    system("cd ./histogram_v2_data/ && python3 gen_data.py 128 1 -1 1");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/histogram_v2_data/input_self.bin", self_size, self, self_size);
    ReadFile(path + "/histogram_v2_data/min.bin", min_size, min, min_size);
    ReadFile(path + "/histogram_v2_data/max.bin", max_size, max, max_size);
    uint64_t tilingKey = 0;
    auto tilingData = GetTilingData(tilingKey, tiling, numBlocks);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(histogram_v2, numBlocks, self, min, max, binsCount, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(self);
    AscendC::GmFree(min);
    AscendC::GmFree(max);
    AscendC::GmFree(binsCount);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

