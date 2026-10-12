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
#include "gtest/gtest.h"
#include "../../../op_host/coalesce_sparse_tiling.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;
extern "C" __global__ __aicore__ void coalesce_sparse(
    GM_ADDR unique_len, GM_ADDR unique_indices, GM_ADDR indices, GM_ADDR values, GM_ADDR new_indices, GM_ADDR new_value,
    GM_ADDR workspace, GM_ADDR tiling);
class coalesce_sparse_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "coalesce_sparse_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "coalesce_sparse_test TearDown\n" << endl;
    }
};

TEST_F(coalesce_sparse_test, test_case_int32_fp32)
{
    size_t unique_len_size_align = 32;
    size_t unique_indices_size_align = 32;
    size_t index_size_align = 32;
    size_t value_size_align = 32;
    size_t new_index_size_align = 32;
    size_t new_value_size_align = 32;
    size_t unique_len_size = 4;
    size_t unique_indices_size = 4;
    size_t index_size = 4;
    size_t value_size = 4;
    size_t new_index_size = 8;
    size_t new_value_size = 6;
    // inputs
    size_t tiling_data_size = sizeof(CoalesceSparseTilingData);

    uint8_t* unique_len = (uint8_t*)AscendC::GmAlloc(unique_len_size_align);
    uint8_t* unique_indices = (uint8_t*)AscendC::GmAlloc(unique_indices_size_align);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(index_size_align);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(value_size_align);
    uint8_t* new_index = (uint8_t*)AscendC::GmAlloc(new_index_size_align);
    uint8_t* new_value = (uint8_t*)AscendC::GmAlloc(new_value_size_align);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;
    CoalesceSparseTilingData* tilingDatafromBin = reinterpret_cast<CoalesceSparseTilingData*>(tiling);
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->m = 2;
    tilingDatafromBin->valueSize = 1;
    tilingDatafromBin->taskNum = 256;
    tilingDatafromBin->taskTail = 4;
    tilingDatafromBin->moveOneSize = 4064;
    tilingDatafromBin->taskRepeatTimes = 0;
    tilingDatafromBin->taskRepeatTail = 256;
    tilingDatafromBin->taskTailRepeatTimes = 0;
    tilingDatafromBin->taskTailRepeatTail = 4;
    tilingDatafromBin->moveValueTimes = 1;
    tilingDatafromBin->moveValueLen = 4;
    tilingDatafromBin->taskTailRepeatTail = 0;
    tilingDatafromBin->deterministicFlag = 0;

    ICPU_SET_TILING_KEY(9);
    ICPU_RUN_KF(
        coalesce_sparse, numBlocks, unique_len, unique_indices, index, value, new_index, new_value, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(unique_len);
    AscendC::GmFree(unique_indices);
    AscendC::GmFree(index);
    AscendC::GmFree(value);

    AscendC::GmFree(new_index);
    AscendC::GmFree(new_value);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}