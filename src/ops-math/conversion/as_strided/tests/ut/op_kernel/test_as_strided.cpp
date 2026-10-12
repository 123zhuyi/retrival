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
 * \file test_as_strided.cpp
 * \brief
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "../../../op_host/as_strided_tiling_arch35.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "impl/dav_c220/rpc/rpc_log.h"
#include "../data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void as_strided(
    uint8_t *x, uint8_t *size, uint8_t *stride, uint8_t *y, uint8_t *workspace, uint8_t *tiling);

class as_strided_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "as_strided_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "as_strided_test TearDown\n" << endl;
    }
};

TEST_F(as_strided_test, test_case_0)
{
    uint64_t tilingKey = 2;
    uint32_t numBlocks = 9;
    int64_t typeSize = 2;

    size_t x_FileSize = 1652720 * typeSize;
    size_t size_FileSize = 3 * sizeof(int32_t);
    size_t stride_FileSize = 3 * sizeof(int32_t);
    size_t y_FileSize = 1024 * 64 * 8 * typeSize;
    size_t workspace_FileSize = 16781184;
    size_t tiling_FileSize = 90 * sizeof(int32_t);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_FileSize);
    uint8_t *size = (uint8_t *)AscendC::GmAlloc(size_FileSize);
    uint8_t *stride = (uint8_t *)AscendC::GmAlloc(stride_FileSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(y_FileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_FileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_FileSize);

    system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/as_strided/as_strided_data ./");
    system("chmod -R 755 ./as_strided/");
    system("cd ./as_strided_data/ && rm -rf ./*bin");
    system("cd ./as_strided_data/ && python3 gen_data.py");
    system("cd ./as_strided_data/ && python3 gen_tiling.py");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/as_strided_data/x.bin", x_FileSize, x, x_FileSize);
    ReadFile(path + "/as_strided_data/size.bin", size_FileSize, size, size_FileSize);
    ReadFile(path + "/as_strided_data/stride.bin", stride_FileSize, stride, stride_FileSize);
    ReadFile(path + "/as_strided_data/tiling.bin", tiling_FileSize, tiling, tiling_FileSize);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(as_strided, numBlocks, x, size, stride, y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)size);
    AscendC::GmFree((void *)stride);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}