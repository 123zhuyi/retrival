/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_strided_slice_assign_v2.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_host/strided_slice_assign_v2_tiling.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void strided_slice_assign_v2(GM_ADDR var, GM_ADDR input_value, GM_ADDR begin,
    GM_ADDR end, GM_ADDR strides, GM_ADDR axes, GM_ADDR var_out, GM_ADDR workspace, GM_ADDR tiling);

class strided_slice_assign_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "strided_slice_assign_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "strided_slice_assign_v2_test TearDown\n" << endl;
    }
};

TEST_F(strided_slice_assign_v2_test, test_case_0)
{
    size_t varByteSize = 4 * 6 * 8 * sizeof(int16_t);
    size_t inputByteSize = 2 * 2 * 4 * sizeof(int16_t);
    size_t idxByteSize = 3 * sizeof(int32_t);

    size_t tiling_data_size = sizeof(StridedSliceAssignV2TilingData);

    uint8_t *var = (uint8_t *)AscendC::GmAlloc(varByteSize);
    uint8_t *input_value = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *begin = (uint8_t *)AscendC::GmAlloc(idxByteSize);
    uint8_t *end = (uint8_t *)AscendC::GmAlloc(idxByteSize);
    uint8_t *strides = (uint8_t *)AscendC::GmAlloc(idxByteSize);
    uint8_t *axes = (uint8_t *)AscendC::GmAlloc(idxByteSize);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 2);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 4;

    char *path_ = get_current_dir_name();
    string path(path_);

    StridedSliceAssignV2TilingData *tilingDatafromBin =
        reinterpret_cast<StridedSliceAssignV2TilingData *>(tiling);

    tilingDatafromBin->dimNum = 3;
    tilingDatafromBin->varDim[0] = 4;
    tilingDatafromBin->varDim[1] = 6;
    tilingDatafromBin->varDim[2] = 8;
    tilingDatafromBin->inputValueDim[0] = 2;
    tilingDatafromBin->inputValueDim[1] = 2;
    tilingDatafromBin->inputValueDim[2] = 4;
    tilingDatafromBin->begin[0] = 1;
    tilingDatafromBin->begin[1] = 2;
    tilingDatafromBin->begin[2] = 3;
    tilingDatafromBin->strides[0] = 2;
    tilingDatafromBin->strides[1] = 3;
    tilingDatafromBin->strides[2] = 1;
    tilingDatafromBin->varCumShape[0] = 0;
    tilingDatafromBin->varCumShape[1] = 48;
    tilingDatafromBin->varCumShape[2] = 8;
    tilingDatafromBin->inputCumShape[0] = 0;
    tilingDatafromBin->inputCumShape[1] = 8;
    tilingDatafromBin->inputCumShape[2] = 4;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(strided_slice_assign_v2, numBlocks, var, input_value, begin, end, strides, axes, var, workspace,
        (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(var);
    AscendC::GmFree(input_value);
    AscendC::GmFree(begin);
    AscendC::GmFree(end);
    AscendC::GmFree(strides);
    AscendC::GmFree(axes);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}