/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_view_copy.cpp
 * \brief
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "view_copy_tiling.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "../data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;


extern "C" __global__ __aicore__ void view_copy(
    uint8_t *dst, uint8_t *dstSize, uint8_t *dstStride, uint8_t *dstStorageOffset, uint8_t *src, uint8_t *srcSize,
    uint8_t *srcStride, uint8_t *srcStorageOffset, uint8_t *out, uint8_t *workspace, uint8_t *tiling);

class view_copy_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "view_copy_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "view_copy_test TearDown\n" << endl;
    }
};

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for(auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}


bool ExcuteTestCase(const std::vector<int64_t> &dst, const std::vector<int64_t> &dstSize,
    const std::vector<int64_t> &dstStride,const std::vector<int64_t> &dstStorageOffset,
    const std::vector<int64_t> &src,const std::vector<int64_t> &srcSize,
    const std::vector<int64_t> &srcStride,const std::vector<int64_t> &srcStorageOffset,
    uint32_t blockDim, const std::string &dtype, int64_t tilingKey, const std::string &caseNameStr)
{
    uint32_t blockNum = 2;
    int64_t typeSize = 1;
    if (dtype == "float" || dtype == "float32" || dtype == "int32" || dtype =="uint32") {
      typeSize = 4;
    } else if (dtype == "float16" || dtype == "bfloat16" || dtype == "int16" || dtype =="uint16")
    {
        typeSize = 2;
    } else if (dtype == "int64"|| dtype =="uint64")
    {
        typeSize = 8;
    } else if (dtype == "bool" || dtype == "int8" || dtype == "hifloat8" || dtype == "float8_e5m2" || dtype == "float8_e4m3fn")
    {
        typeSize = 1;
    }

    size_t param1FileSize = GetShapeSize(dst) * typeSize;
    size_t param2FileSize = dstSize.size() * sizeof(int64_t);
    size_t param3FileSize = dstStride.size() * sizeof(int64_t);
    size_t param4FileSize = dstStorageOffset.size() * sizeof(int64_t);
    size_t param5FileSize = GetShapeSize(src) * typeSize;
    size_t param6FileSize = srcSize.size() * sizeof(int64_t);
    size_t param7FileSize = srcStride.size() * sizeof(int64_t);
    size_t param8FileSize = srcStorageOffset.size() * sizeof(int64_t);

    size_t workspaceFileSize = 16*1024*1024 + 64*sizeof(int32_t);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);

    size_t tilingSize = sizeof(ViewCopyTilingData);

    uint8_t *param1 = (uint8_t *)AscendC::GmAlloc((param1FileSize + 31)/32*32);
    uint8_t *param2 = (uint8_t *)AscendC::GmAlloc((param2FileSize + 31)/32*32);
    uint8_t *param3 = (uint8_t *)AscendC::GmAlloc((param3FileSize + 31)/32*32);
    uint8_t *param4 = (uint8_t *)AscendC::GmAlloc((param4FileSize + 31)/32*32);
    uint8_t *param5 = (uint8_t *)AscendC::GmAlloc((param5FileSize + 31)/32*32);
    uint8_t *param6 = (uint8_t *)AscendC::GmAlloc((param6FileSize + 31)/32*32);
    uint8_t *param7 = (uint8_t *)AscendC::GmAlloc((param7FileSize + 31)/32*32);
    uint8_t *param8 = (uint8_t *)AscendC::GmAlloc((param8FileSize + 31)/32*32);

    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/view_copy/view_copy_data ./");
    system("chmod -R 755 ./view_copy/");
    system("cd ./view_copy_data/ && rm -rf ./*bin");
    std::string cmd = "cd ./view_copy_data/ && python3 gen_data.py \'" + caseNameStr + "\' " + dtype;
    system(cmd.c_str());
    cmd = "cd ./view_copy_data/ && python3 gen_tiling.py \'" + caseNameStr + "\' ";
    system(cmd.c_str());

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/view_copy_data/dst.bin", param1FileSize, param1, param1FileSize);
    ReadFile(path + "/view_copy_data/dst_size.bin", param2FileSize, param2, param2FileSize);
    ReadFile(path + "/view_copy_data/dst_stride.bin", param3FileSize, param3, param3FileSize);
    ReadFile(path + "/view_copy_data/dst_storage_offset.bin", param4FileSize, param4, param4FileSize);
    ReadFile(path + "/view_copy_data/src.bin", param5FileSize, param5, param5FileSize);
    ReadFile(path + "/view_copy_data/src_size.bin", param6FileSize, param6, param6FileSize);
    ReadFile(path + "/view_copy_data/src_stride.bin", param7FileSize, param7, param7FileSize);
    ReadFile(path + "/view_copy_data/src_storage_offset.bin", param8FileSize, param8, param8FileSize);
    ReadFile(path + "/view_copy_data/tiling.bin", tilingSize, tiling, tilingSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(view_copy, blockDim, param1, param2, param3, param4, param5, param6, param7, param8, param1, workspace, tiling);
    WriteFile("./view_copy_data/cce_cpu_out.bin", param1, param1FileSize);

    AscendC::GmFree((void *)param1);
    AscendC::GmFree((void *)param2);
    AscendC::GmFree((void *)param3);
    AscendC::GmFree((void *)param4);
    AscendC::GmFree((void *)param5);
    AscendC::GmFree((void *)param6);
    AscendC::GmFree((void *)param7);
    AscendC::GmFree((void *)param8);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    return true;
}

TEST_F(view_copy_test, test_case_dim1_b32)
{
    uint64_t tilingKey = 114;
    uint32_t blockDim = 4;

    std::vector<int64_t> dst = {32};
    std::vector<int64_t> dstSize = {32};
    std::vector<int64_t> dstStride = {1};
    std::vector<int64_t> dstStorageOffset = {0};
    std::vector<int64_t> src = {32};
    std::vector<int64_t> srcSize = {32};
    std::vector<int64_t> srcStride = {1};
    std::vector<int64_t> srcStorageOffset = {0};

    std::string caseNameStr = "case1";
    std::string dtype = "float32";
    EXPECT_EQ(ExcuteTestCase(dst, dstSize, dstStride, dstStorageOffset, src, srcSize,srcStride,srcStorageOffset,
                             blockDim, dtype, tilingKey, caseNameStr), true);

}

TEST_F(view_copy_test, test_case_dim2_b16)
{
    uint64_t tilingKey = 122;
    uint32_t blockDim = 4;

    std::vector<int64_t> dst = {1, 500};
    std::vector<int64_t> dstSize = {1, 500};
    std::vector<int64_t> dstStride = {500, 1};
    std::vector<int64_t> dstStorageOffset = {0};
    std::vector<int64_t> src = {500, 4};
    std::vector<int64_t> srcSize = {1, 500};
    std::vector<int64_t> srcStride = {2000, 4};
    std::vector<int64_t> srcStorageOffset = {0};

    std::string caseNameStr = "case2";
    std::string dtype = "float16";
    EXPECT_EQ(ExcuteTestCase(dst, dstSize, dstStride, dstStorageOffset, src, srcSize,srcStride,srcStorageOffset,
                             blockDim, dtype, tilingKey, caseNameStr), true);
}

TEST_F(view_copy_test, test_case_dim2_b8)
{
    uint64_t tilingKey = 121;
    uint32_t blockDim = 4;
    std::vector<int64_t> dst = {75, 2};
    std::vector<int64_t> dstSize = {75, 2};
    std::vector<int64_t> dstStride = {2, 1};
    std::vector<int64_t> dstStorageOffset = {0};
    std::vector<int64_t> src = {75, 4};
    std::vector<int64_t> srcSize = {75, 2};
    std::vector<int64_t> srcStride = {4, 1};
    std::vector<int64_t> srcStorageOffset = {0};

    std::string caseNameStr = "case3";
    std::string dtype = "int8";
    EXPECT_EQ(ExcuteTestCase(dst, dstSize, dstStride, dstStorageOffset, src, srcSize,srcStride,srcStorageOffset,
                             blockDim, dtype, tilingKey, caseNameStr), true);

}


TEST_F(view_copy_test, test_case_dim3_b8)
{
    uint64_t tilingKey = 131;
    uint32_t blockDim = 4;

    std::vector<int64_t> dst = {2, 120, 2};
    std::vector<int64_t> dstSize = {2, 120, 2};
    std::vector<int64_t> dstStride = {240, 2, 1};
    std::vector<int64_t> dstStorageOffset = {0};
    std::vector<int64_t> src = {2, 120, 5};
    std::vector<int64_t> srcSize = {2, 120, 2};
    std::vector<int64_t> srcStride = {600, 5, 1};
    std::vector<int64_t> srcStorageOffset = {0};

    std::string caseNameStr = "case4";
    std::string dtype = "int8";
    EXPECT_EQ(ExcuteTestCase(dst, dstSize, dstStride, dstStorageOffset, src, srcSize,srcStride,srcStorageOffset,
                             blockDim, dtype, tilingKey, caseNameStr), true);
}


TEST_F(view_copy_test, test_case_dim4_b16)
{
    uint64_t tilingKey = 142;
    uint32_t blockDim = 4;
    std::vector<int64_t> dst = {3, 50, 1, 2};
    std::vector<int64_t> dstSize = {2, 50, 1, 2};
    std::vector<int64_t> dstStride = {100, 2, 2, 1};
    std::vector<int64_t> dstStorageOffset = {0};
    std::vector<int64_t> src = {3, 50, 1, 9};
    std::vector<int64_t> srcSize = {2, 50, 1, 2};
    std::vector<int64_t> srcStride = {450, 9, 9, 1};
    std::vector<int64_t> srcStorageOffset = {0};

    std::string caseNameStr = "case5";
    std::string dtype = "int16";
    EXPECT_EQ(ExcuteTestCase(dst, dstSize, dstStride, dstStorageOffset, src, srcSize,srcStride,srcStorageOffset,
                             blockDim, dtype, tilingKey, caseNameStr), true);
}


TEST_F(view_copy_test, test_case_dim5_b32)
{
    uint64_t tilingKey = 154;
    uint32_t blockDim = 4;

    std::vector<int64_t> dst = {2, 2, 2, 2, 2};
    std::vector<int64_t> dstSize = {2, 2, 2, 2, 2};
    std::vector<int64_t> dstStride = {16, 8, 4, 2, 1};
    std::vector<int64_t> dstStorageOffset = {0};
    std::vector<int64_t> src = {5, 2, 2, 3, 2};
    std::vector<int64_t> srcSize = {2, 2, 2, 2, 2};
    std::vector<int64_t> srcStride = {24, 12, 6, 2, 1};
    std::vector<int64_t> srcStorageOffset = {0};

    std::string caseNameStr = "case6";
    std::string dtype = "int32";
    EXPECT_EQ(ExcuteTestCase(dst, dstSize, dstStride, dstStorageOffset, src, srcSize,srcStride,srcStorageOffset,
                             blockDim, dtype, tilingKey, caseNameStr), true);
}

TEST_F(view_copy_test, test_case_dim8_b32)
{
    uint64_t tilingKey = 184;
    uint32_t blockDim = 4;

    std::vector<int64_t> dst = {2, 4, 3, 2, 2, 2, 2, 2};
    std::vector<int64_t> dstSize = {2, 4, 3, 2, 2, 2, 2, 2};
    std::vector<int64_t> dstStride = {384, 96, 32, 16, 8, 4, 2, 1};
    std::vector<int64_t> dstStorageOffset = {0};
    std::vector<int64_t> src = {2, 4, 3, 5, 2, 2, 3, 2};
    std::vector<int64_t> srcSize = {2, 4, 3, 2, 2, 2, 2, 2};
    std::vector<int64_t> srcStride = {1440, 360, 120, 24, 12, 6, 2, 1};
    std::vector<int64_t> srcStorageOffset = {0};

    std::string caseNameStr = "case7";
    std::string dtype = "int32";
    EXPECT_EQ(ExcuteTestCase(dst, dstSize, dstStride, dstStorageOffset, src, srcSize,srcStride,srcStorageOffset,
                             blockDim, dtype, tilingKey, caseNameStr), true);
}

TEST_F(view_copy_test, test_case_dim1_b32_simt)
{
    uint64_t tilingKey = 114;
    uint32_t blockDim = 4;

    std::vector<int64_t> dst = {32};
    std::vector<int64_t> dstSize = {32};
    std::vector<int64_t> dstStride = {1};
    std::vector<int64_t> dstStorageOffset = {0};
    std::vector<int64_t> src = {32};
    std::vector<int64_t> srcSize = {32};
    std::vector<int64_t> srcStride = {1};
    std::vector<int64_t> srcStorageOffset = {0};

    std::string caseNameStr = "case8";
    std::string dtype = "float32";
    EXPECT_EQ(ExcuteTestCase(dst, dstSize, dstStride, dstStorageOffset, src, srcSize,srcStride,srcStorageOffset,
                             blockDim, dtype, tilingKey, caseNameStr), true);

}

TEST_F(view_copy_test, test_case_dim2_b16_simt)
{
    uint64_t tilingKey = 122;
    uint32_t blockDim = 4;

    std::vector<int64_t> dst = {1, 500};
    std::vector<int64_t> dstSize = {1, 500};
    std::vector<int64_t> dstStride = {500, 1};
    std::vector<int64_t> dstStorageOffset = {0};
    std::vector<int64_t> src = {500, 4};
    std::vector<int64_t> srcSize = {1, 500};
    std::vector<int64_t> srcStride = {2000, 4};
    std::vector<int64_t> srcStorageOffset = {0};

    std::string caseNameStr = "case9";
    std::string dtype = "float16";
    EXPECT_EQ(ExcuteTestCase(dst, dstSize, dstStride, dstStorageOffset, src, srcSize,srcStride,srcStorageOffset,
                             blockDim, dtype, tilingKey, caseNameStr), true);
}

TEST_F(view_copy_test, test_case_dim2_b8_simt)
{
    uint64_t tilingKey = 121;
    uint32_t blockDim = 4;
    std::vector<int64_t> dst = {75, 2};
    std::vector<int64_t> dstSize = {75, 2};
    std::vector<int64_t> dstStride = {2, 1};
    std::vector<int64_t> dstStorageOffset = {0};
    std::vector<int64_t> src = {75, 4};
    std::vector<int64_t> srcSize = {75, 2};
    std::vector<int64_t> srcStride = {4, 1};
    std::vector<int64_t> srcStorageOffset = {0};

    std::string caseNameStr = "case10";
    std::string dtype = "int8";
    EXPECT_EQ(ExcuteTestCase(dst, dstSize, dstStride, dstStorageOffset, src, srcSize,srcStride,srcStorageOffset,
                             blockDim, dtype, tilingKey, caseNameStr), true);

}


TEST_F(view_copy_test, test_case_dim3_b8_simt)
{
    uint64_t tilingKey = 131;
    uint32_t blockDim = 4;

    std::vector<int64_t> dst = {2, 120, 2};
    std::vector<int64_t> dstSize = {2, 120, 2};
    std::vector<int64_t> dstStride = {240, 2, 1};
    std::vector<int64_t> dstStorageOffset = {0};
    std::vector<int64_t> src = {2, 120, 5};
    std::vector<int64_t> srcSize = {2, 120, 2};
    std::vector<int64_t> srcStride = {600, 5, 1};
    std::vector<int64_t> srcStorageOffset = {0};

    std::string caseNameStr = "case11";
    std::string dtype = "int8";
    EXPECT_EQ(ExcuteTestCase(dst, dstSize, dstStride, dstStorageOffset, src, srcSize,srcStride,srcStorageOffset,
                             blockDim, dtype, tilingKey, caseNameStr), true);
}


TEST_F(view_copy_test, test_case_dim4_b16_simt)
{
    uint64_t tilingKey = 142;
    uint32_t blockDim = 4;
    std::vector<int64_t> dst = {3, 50, 1, 2};
    std::vector<int64_t> dstSize = {2, 50, 1, 2};
    std::vector<int64_t> dstStride = {100, 2, 2, 1};
    std::vector<int64_t> dstStorageOffset = {0};
    std::vector<int64_t> src = {3, 50, 1, 9};
    std::vector<int64_t> srcSize = {2, 50, 1, 2};
    std::vector<int64_t> srcStride = {450, 9, 9, 1};
    std::vector<int64_t> srcStorageOffset = {0};

    std::string caseNameStr = "case12";
    std::string dtype = "int16";
    EXPECT_EQ(ExcuteTestCase(dst, dstSize, dstStride, dstStorageOffset, src, srcSize,srcStride,srcStorageOffset,
                             blockDim, dtype, tilingKey, caseNameStr), true);
}


TEST_F(view_copy_test, test_case_dim5_b32_simt)
{
    uint64_t tilingKey = 154;
    uint32_t blockDim = 4;

    std::vector<int64_t> dst = {2, 2, 2, 2, 2};
    std::vector<int64_t> dstSize = {2, 2, 2, 2, 2};
    std::vector<int64_t> dstStride = {16, 8, 4, 2, 1};
    std::vector<int64_t> dstStorageOffset = {0};
    std::vector<int64_t> src = {5, 2, 2, 3, 2};
    std::vector<int64_t> srcSize = {2, 2, 2, 2, 2};
    std::vector<int64_t> srcStride = {24, 12, 6, 2, 1};
    std::vector<int64_t> srcStorageOffset = {0};

    std::string caseNameStr = "case13";
    std::string dtype = "int32";
    EXPECT_EQ(ExcuteTestCase(dst, dstSize, dstStride, dstStorageOffset, src, srcSize,srcStride,srcStorageOffset,
                             blockDim, dtype, tilingKey, caseNameStr), true);
}

TEST_F(view_copy_test, test_case_dim8_b32_simt)
{
    uint64_t tilingKey = 184;
    uint32_t blockDim = 4;

    std::vector<int64_t> dst = {2, 4, 3, 2, 2, 2, 2, 2};
    std::vector<int64_t> dstSize = {2, 4, 3, 2, 2, 2, 2, 2};
    std::vector<int64_t> dstStride = {384, 96, 32, 16, 8, 4, 2, 1};
    std::vector<int64_t> dstStorageOffset = {0};
    std::vector<int64_t> src = {2, 4, 3, 5, 2, 2, 3, 2};
    std::vector<int64_t> srcSize = {2, 4, 3, 2, 2, 2, 2, 2};
    std::vector<int64_t> srcStride = {1440, 360, 120, 24, 12, 6, 2, 1};
    std::vector<int64_t> srcStorageOffset = {0};

    std::string caseNameStr = "case14";
    std::string dtype = "int32";
    EXPECT_EQ(ExcuteTestCase(dst, dstSize, dstStride, dstStorageOffset, src, srcSize,srcStride,srcStorageOffset,
                             blockDim, dtype, tilingKey, caseNameStr), true);
}

TEST_F(view_copy_test, test_case_dim3_b64_simt)
{
    uint64_t tilingKey = 131;
    uint32_t blockDim = 4;

    std::vector<int64_t> dst = {2, 120, 2};
    std::vector<int64_t> dstSize = {2, 120, 2};
    std::vector<int64_t> dstStride = {240, 2, 1};
    std::vector<int64_t> dstStorageOffset = {0};
    std::vector<int64_t> src = {2, 120, 5};
    std::vector<int64_t> srcSize = {2, 120, 2};
    std::vector<int64_t> srcStride = {600, 5, 1};
    std::vector<int64_t> srcStorageOffset = {0};

    std::string caseNameStr = "case11";
    std::string dtype = "int64";
    EXPECT_EQ(ExcuteTestCase(dst, dstSize, dstStride, dstStorageOffset, src, srcSize,srcStride,srcStorageOffset,
                             blockDim, dtype, tilingKey, caseNameStr), true);
}

TEST_F(view_copy_test, test_case_dim4_b8)
{
    uint64_t tilingKey = 142;
    uint32_t blockDim = 4;
    std::vector<int64_t> dst = {3, 50, 1, 2};
    std::vector<int64_t> dstSize = {2, 50, 1, 2};
    std::vector<int64_t> dstStride = {100, 2, 2, 1};
    std::vector<int64_t> dstStorageOffset = {0};
    std::vector<int64_t> src = {3, 50, 1, 9};
    std::vector<int64_t> srcSize = {2, 50, 1, 2};
    std::vector<int64_t> srcStride = {450, 9, 9, 1};
    std::vector<int64_t> srcStorageOffset = {0};

    std::string caseNameStr = "case5";
    std::string dtype = "hifloat8";
    EXPECT_EQ(ExcuteTestCase(dst, dstSize, dstStride, dstStorageOffset, src, srcSize,srcStride,srcStorageOffset,
                             blockDim, dtype, tilingKey, caseNameStr), true);
}