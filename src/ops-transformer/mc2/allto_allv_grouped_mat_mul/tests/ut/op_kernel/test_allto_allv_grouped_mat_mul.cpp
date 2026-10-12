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
#include <gtest/gtest.h>
#include "tikicpulib.h"
#include "allto_allv_grouped_mat_mul_tiling_def.h"
#include "../../../op_kernel/allto_allv_grouped_mat_mul.cpp"

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};
class AlltoAllvGroupedMatMulTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllvGroupedMatMulTest SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "AlltoAllvGroupedMatMulTest TearDown\n" << std::endl;
    }
};

// shard = 1
TEST_F(AlltoAllvGroupedMatMulTest, AlltoAllvGroupedMatMulTest0)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllvGmmTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllvGmmCommonTilingInfo commonTilingInfo{4096, 2048, 2,      7168,  7168,  4096,  4096,  4096,  1,   1,
                                                  40,   20,   194560, false, false, false, false, false, true};
    AlltoAllvGmmTilingData* tilingData = reinterpret_cast<AlltoAllvGmmTilingData*>(tiling);
    tilingData->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BSK * commonTilingInfo.H1 * sizeof(uint16_t));
    uint8_t* gmmweightGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H1 *
                                                      commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* expertTokenNumGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.epWorldSize * sizeof(uint16_t));
    uint8_t* mmxGM = nullptr;
    uint8_t* mmweightGM = nullptr;
    uint8_t* gmmyGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyGM = nullptr;
    uint8_t* allGatherOutGM = nullptr;
    uint8_t* alltoAllvOutGM = nullptr;

    ICPU_SET_TILING_KEY(0);

    auto alltoAllvGroupedMatMulWrapper = [](
                                                                 GM_ADDR gmmxGM, GM_ADDR gmmweightGM,
                                                                 GM_ADDR expertTokenNumGM, GM_ADDR mmxGM,
                                                                 GM_ADDR mmweightGM, GM_ADDR gmmyGM, GM_ADDR mmyGM,
                                                                 GM_ADDR allGatherOutGM, GM_ADDR alltoAllvOutGM,
                                                                 GM_ADDR workspaceGM, GM_ADDR tilingGM
    ) {
        allto_allv_grouped_mat_mul<0, false, false, false>(gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM,
                                                            mmweightGM, gmmyGM, mmyGM,
                                                            allGatherOutGM, alltoAllvOutGM, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(alltoAllvGroupedMatMulWrapper, 20, gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM, mmweightGM, gmmyGM, mmyGM,
                allGatherOutGM, alltoAllvOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)expertTokenNumGM);
    AscendC::GmFree((void*)gmmyGM);
}

TEST_F(AlltoAllvGroupedMatMulTest, AlltoAllvGroupedMatMulTest10)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllvGmmTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllvGmmCommonTilingInfo commonTilingInfo{4096, 2048, 2,      7168,  7168,  4096,  4096,  4096,  1,   1,
                                                  40,   20,   194560, false, false, false, false, false, true};
    AlltoAllvGmmTilingData* tilingData = reinterpret_cast<AlltoAllvGmmTilingData*>(tiling);
    tilingData->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BSK * commonTilingInfo.H1 * sizeof(uint16_t));
    uint8_t* gmmweightGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H1 *
                                                      commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* expertTokenNumGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.epWorldSize * sizeof(uint16_t));
    uint8_t* mmxGM = nullptr;
    uint8_t* mmweightGM = nullptr;
    uint8_t* gmmyGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyGM = nullptr;
    uint8_t* allGatherOutGM = nullptr;
    uint8_t* alltoAllvOutGM = nullptr;

    ICPU_SET_TILING_KEY(10);
    auto alltoAllvGroupedMatMulWrapper = [](
                                                                 GM_ADDR gmmxGM, GM_ADDR gmmweightGM,
                                                                 GM_ADDR expertTokenNumGM, GM_ADDR mmxGM,
                                                                 GM_ADDR mmweightGM, GM_ADDR gmmyGM, GM_ADDR mmyGM,
                                                                 GM_ADDR allGatherOutGM, GM_ADDR alltoAllvOutGM,
                                                                 GM_ADDR workspaceGM, GM_ADDR tilingGM
    ) {
        allto_allv_grouped_mat_mul<0, false, false, false>(gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM,
                                                            mmweightGM, gmmyGM, mmyGM,
                                                            allGatherOutGM, alltoAllvOutGM, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(alltoAllvGroupedMatMulWrapper, 20, gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM, mmweightGM, gmmyGM, mmyGM,
                allGatherOutGM, alltoAllvOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)expertTokenNumGM);
    AscendC::GmFree((void*)gmmyGM);
}

TEST_F(AlltoAllvGroupedMatMulTest, AlltoAllvGroupedMatMulTest100)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllvGmmTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllvGmmCommonTilingInfo commonTilingInfo{4096, 2048, 2,      7168,  7168,  4096,  4096,  4096,  1,   1,
                                                  40,   20,   194560, false, false, false, false, false, true};
    AlltoAllvGmmTilingData* tilingData = reinterpret_cast<AlltoAllvGmmTilingData*>(tiling);
    tilingData->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BSK * commonTilingInfo.H1 * sizeof(uint16_t));
    uint8_t* gmmweightGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H1 *
                                                      commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* expertTokenNumGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.epWorldSize * sizeof(uint16_t));
    uint8_t* mmxGM = nullptr;
    uint8_t* mmweightGM = nullptr;
    uint8_t* gmmyGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyGM = nullptr;
    uint8_t* allGatherOutGM = nullptr;
    uint8_t* alltoAllvOutGM = nullptr;

    ICPU_SET_TILING_KEY(100);
    auto alltoAllvGroupedMatMulWrapper = [](
                                                                 GM_ADDR gmmxGM, GM_ADDR gmmweightGM,
                                                                 GM_ADDR expertTokenNumGM, GM_ADDR mmxGM,
                                                                 GM_ADDR mmweightGM, GM_ADDR gmmyGM, GM_ADDR mmyGM,
                                                                 GM_ADDR allGatherOutGM, GM_ADDR alltoAllvOutGM,
                                                                 GM_ADDR workspaceGM, GM_ADDR tilingGM
    ) {
        allto_allv_grouped_mat_mul<0, false, false, false>(gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM,
                                                            mmweightGM, gmmyGM, mmyGM,
                                                            allGatherOutGM, alltoAllvOutGM, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(alltoAllvGroupedMatMulWrapper, 20, gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM, mmweightGM, gmmyGM, mmyGM,
                allGatherOutGM, alltoAllvOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)expertTokenNumGM);
    AscendC::GmFree((void*)gmmyGM);
}

TEST_F(AlltoAllvGroupedMatMulTest, AlltoAllvGroupedMatMulTest101)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllvGmmTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllvGmmCommonTilingInfo commonTilingInfo{4096, 2048, 2,      7168,  7168,  4096,  4096,  4096,  1,   1,
                                                  40,   20,   194560, false, false, false, false, false, true};
    AlltoAllvGmmTilingData* tilingData = reinterpret_cast<AlltoAllvGmmTilingData*>(tiling);
    tilingData->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BSK * commonTilingInfo.H1 * sizeof(uint16_t));
    uint8_t* gmmweightGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H1 *
                                                      commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* expertTokenNumGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.epWorldSize * sizeof(uint16_t));
    uint8_t* mmxGM = nullptr;
    uint8_t* mmweightGM = nullptr;
    uint8_t* gmmyGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyGM = nullptr;
    uint8_t* allGatherOutGM = nullptr;
    uint8_t* alltoAllvOutGM = nullptr;

    ICPU_SET_TILING_KEY(101);
    auto alltoAllvGroupedMatMulWrapper = [](
                                                                 GM_ADDR gmmxGM, GM_ADDR gmmweightGM,
                                                                 GM_ADDR expertTokenNumGM, GM_ADDR mmxGM,
                                                                 GM_ADDR mmweightGM, GM_ADDR gmmyGM, GM_ADDR mmyGM,
                                                                 GM_ADDR allGatherOutGM, GM_ADDR alltoAllvOutGM,
                                                                 GM_ADDR workspaceGM, GM_ADDR tilingGM
    ) {
        allto_allv_grouped_mat_mul<0, false, false, false>(gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM,
                                                            mmweightGM, gmmyGM, mmyGM,
                                                            allGatherOutGM, alltoAllvOutGM, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(alltoAllvGroupedMatMulWrapper, 20, gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM, mmweightGM, gmmyGM, mmyGM,
                allGatherOutGM, alltoAllvOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)expertTokenNumGM);
    AscendC::GmFree((void*)gmmyGM);
}

TEST_F(AlltoAllvGroupedMatMulTest, AlltoAllvGroupedMatMulTest110)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllvGmmTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllvGmmCommonTilingInfo commonTilingInfo{4096, 2048, 2,      7168,  7168,  4096,  4096,  4096,  1,   1,
                                                  40,   20,   194560, false, false, false, false, false, true};
    AlltoAllvGmmTilingData* tilingData = reinterpret_cast<AlltoAllvGmmTilingData*>(tiling);
    tilingData->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BSK * commonTilingInfo.H1 * sizeof(uint16_t));
    uint8_t* gmmweightGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H1 *
                                                      commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* expertTokenNumGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.epWorldSize * sizeof(uint16_t));
    uint8_t* mmxGM = nullptr;
    uint8_t* mmweightGM = nullptr;
    uint8_t* gmmyGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyGM = nullptr;
    uint8_t* allGatherOutGM = nullptr;
    uint8_t* alltoAllvOutGM = nullptr;

    ICPU_SET_TILING_KEY(110);
    auto alltoAllvGroupedMatMulWrapper = [](
                                                                 GM_ADDR gmmxGM, GM_ADDR gmmweightGM,
                                                                 GM_ADDR expertTokenNumGM, GM_ADDR mmxGM,
                                                                 GM_ADDR mmweightGM, GM_ADDR gmmyGM, GM_ADDR mmyGM,
                                                                 GM_ADDR allGatherOutGM, GM_ADDR alltoAllvOutGM,
                                                                 GM_ADDR workspaceGM, GM_ADDR tilingGM
    ) {
        allto_allv_grouped_mat_mul<0, false, false, false>(gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM,
                                                            mmweightGM, gmmyGM, mmyGM,
                                                            allGatherOutGM, alltoAllvOutGM, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(alltoAllvGroupedMatMulWrapper, 20, gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM, mmweightGM, gmmyGM, mmyGM,
                allGatherOutGM, alltoAllvOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)expertTokenNumGM);
    AscendC::GmFree((void*)gmmyGM);
}

TEST_F(AlltoAllvGroupedMatMulTest, AlltoAllvGroupedMatMulTest111)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllvGmmTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllvGmmCommonTilingInfo commonTilingInfo{4096, 2048, 2,      7168,  7168,  4096,  4096,  4096,  1,   1,
                                                  40,   20,   194560, false, false, false, false, false, true};
    AlltoAllvGmmTilingData* tilingData = reinterpret_cast<AlltoAllvGmmTilingData*>(tiling);
    tilingData->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BSK * commonTilingInfo.H1 * sizeof(uint16_t));
    uint8_t* gmmweightGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H1 *
                                                      commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* expertTokenNumGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.epWorldSize * sizeof(uint16_t));
    uint8_t* mmxGM = nullptr;
    uint8_t* mmweightGM = nullptr;
    uint8_t* gmmyGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyGM = nullptr;
    uint8_t* allGatherOutGM = nullptr;
    uint8_t* alltoAllvOutGM = nullptr;

    ICPU_SET_TILING_KEY(111);
    auto alltoAllvGroupedMatMulWrapper = [](
                                                                 GM_ADDR gmmxGM, GM_ADDR gmmweightGM,
                                                                 GM_ADDR expertTokenNumGM, GM_ADDR mmxGM,
                                                                 GM_ADDR mmweightGM, GM_ADDR gmmyGM, GM_ADDR mmyGM,
                                                                 GM_ADDR allGatherOutGM, GM_ADDR alltoAllvOutGM,
                                                                 GM_ADDR workspaceGM, GM_ADDR tilingGM
    ) {
        allto_allv_grouped_mat_mul<0, false, false, false>(gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM,
                                                            mmweightGM, gmmyGM, mmyGM,
                                                            allGatherOutGM, alltoAllvOutGM, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(alltoAllvGroupedMatMulWrapper, 20, gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM, mmweightGM, gmmyGM, mmyGM,
                allGatherOutGM, alltoAllvOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)expertTokenNumGM);
    AscendC::GmFree((void*)gmmyGM);
}

TEST_F(AlltoAllvGroupedMatMulTest, AlltoAllvGroupedMatMulTest1000)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllvGmmTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllvGmmCommonTilingInfo commonTilingInfo{4096, 2048, 2,      7168,  7168,  4096,  4096,  4096,  1,   1,
                                                  40,   20,   194560, false, false, false, false, false, true};
    AlltoAllvGmmTilingData* tilingData = reinterpret_cast<AlltoAllvGmmTilingData*>(tiling);
    tilingData->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BSK * commonTilingInfo.H1 * sizeof(uint16_t));
    uint8_t* gmmweightGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H1 *
                                                      commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* expertTokenNumGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.epWorldSize * sizeof(uint16_t));
    uint8_t* mmxGM = nullptr;
    uint8_t* mmweightGM = nullptr;
    uint8_t* gmmyGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyGM = nullptr;
    uint8_t* allGatherOutGM = nullptr;
    uint8_t* alltoAllvOutGM = nullptr;

    ICPU_SET_TILING_KEY(1000);
    auto alltoAllvGroupedMatMulWrapper = [](
                                                                 GM_ADDR gmmxGM, GM_ADDR gmmweightGM,
                                                                 GM_ADDR expertTokenNumGM, GM_ADDR mmxGM,
                                                                 GM_ADDR mmweightGM, GM_ADDR gmmyGM, GM_ADDR mmyGM,
                                                                 GM_ADDR allGatherOutGM, GM_ADDR alltoAllvOutGM,
                                                                 GM_ADDR workspaceGM, GM_ADDR tilingGM
    ) {
        allto_allv_grouped_mat_mul<0, false, false, false>(gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM,
                                                            mmweightGM, gmmyGM, mmyGM,
                                                            allGatherOutGM, alltoAllvOutGM, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(alltoAllvGroupedMatMulWrapper, 20, gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM, mmweightGM, gmmyGM, mmyGM,
                allGatherOutGM, alltoAllvOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)expertTokenNumGM);
    AscendC::GmFree((void*)gmmyGM);
}

TEST_F(AlltoAllvGroupedMatMulTest, AlltoAllvGroupedMatMulTest1010)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllvGmmTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllvGmmCommonTilingInfo commonTilingInfo{4096, 2048, 2,      7168,  7168,  4096,  4096,  4096,  1,   1,
                                                  40,   20,   194560, false, false, false, false, false, true};
    AlltoAllvGmmTilingData* tilingData = reinterpret_cast<AlltoAllvGmmTilingData*>(tiling);
    tilingData->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BSK * commonTilingInfo.H1 * sizeof(uint16_t));
    uint8_t* gmmweightGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H1 *
                                                      commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* expertTokenNumGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.epWorldSize * sizeof(uint16_t));
    uint8_t* mmxGM = nullptr;
    uint8_t* mmweightGM = nullptr;
    uint8_t* gmmyGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyGM = nullptr;
    uint8_t* allGatherOutGM = nullptr;
    uint8_t* alltoAllvOutGM = nullptr;

    ICPU_SET_TILING_KEY(1010);
    auto alltoAllvGroupedMatMulWrapper = [](
                                                                 GM_ADDR gmmxGM, GM_ADDR gmmweightGM,
                                                                 GM_ADDR expertTokenNumGM, GM_ADDR mmxGM,
                                                                 GM_ADDR mmweightGM, GM_ADDR gmmyGM, GM_ADDR mmyGM,
                                                                 GM_ADDR allGatherOutGM, GM_ADDR alltoAllvOutGM,
                                                                 GM_ADDR workspaceGM, GM_ADDR tilingGM
    ) {
        allto_allv_grouped_mat_mul<0, false, false, false>(gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM,
                                                            mmweightGM, gmmyGM, mmyGM,
                                                            allGatherOutGM, alltoAllvOutGM, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(alltoAllvGroupedMatMulWrapper, 20, gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM, mmweightGM, gmmyGM, mmyGM,
                allGatherOutGM, alltoAllvOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)expertTokenNumGM);
    AscendC::GmFree((void*)gmmyGM);
}

TEST_F(AlltoAllvGroupedMatMulTest, AlltoAllvGroupedMatMulTest1110)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllvGmmTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllvGmmCommonTilingInfo commonTilingInfo{4096, 2048, 2,      7168,  7168,  4096,  4096,  4096,  1,   1,
                                                  40,   20,   194560, false, false, false, false, false, true};
    AlltoAllvGmmTilingData* tilingData = reinterpret_cast<AlltoAllvGmmTilingData*>(tiling);
    tilingData->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BSK * commonTilingInfo.H1 * sizeof(uint16_t));
    uint8_t* gmmweightGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H1 *
                                                      commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* expertTokenNumGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.epWorldSize * sizeof(uint16_t));
    uint8_t* mmxGM = nullptr;
    uint8_t* mmweightGM = nullptr;
    uint8_t* gmmyGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyGM = nullptr;
    uint8_t* allGatherOutGM = nullptr;
    uint8_t* alltoAllvOutGM = nullptr;

    ICPU_SET_TILING_KEY(1110);
    auto alltoAllvGroupedMatMulWrapper = [](
                                                                 GM_ADDR gmmxGM, GM_ADDR gmmweightGM,
                                                                 GM_ADDR expertTokenNumGM, GM_ADDR mmxGM,
                                                                 GM_ADDR mmweightGM, GM_ADDR gmmyGM, GM_ADDR mmyGM,
                                                                 GM_ADDR allGatherOutGM, GM_ADDR alltoAllvOutGM,
                                                                 GM_ADDR workspaceGM, GM_ADDR tilingGM
    ) {
        allto_allv_grouped_mat_mul<0, false, false, false>(gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM,
                                                            mmweightGM, gmmyGM, mmyGM,
                                                            allGatherOutGM, alltoAllvOutGM, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(alltoAllvGroupedMatMulWrapper, 20, gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM, mmweightGM, gmmyGM, mmyGM,
                allGatherOutGM, alltoAllvOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)expertTokenNumGM);
    AscendC::GmFree((void*)gmmyGM);
}

TEST_F(AlltoAllvGroupedMatMulTest, AlltoAllvGroupedMatMulTest1111)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::string group{"group"};
    size_t sysWorkspaceSize = 256 * 1024 * 1024;
    size_t usrWorkspaceSize = 256 * 1024 * 1024;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(AlltoAllvGmmTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AlltoAllvGmmCommonTilingInfo commonTilingInfo{4096, 2048, 2,      7168,  7168,  4096,  4096,  4096,  1,   1,
                                                  40,   20,   194560, false, false, false, false, false, true};
    AlltoAllvGmmTilingData* tilingData = reinterpret_cast<AlltoAllvGmmTilingData*>(tiling);
    tilingData->commonTilingInfo = commonTilingInfo;

    uint8_t* gmmxGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.BSK * commonTilingInfo.H1 * sizeof(uint16_t));
    uint8_t* gmmweightGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.H1 *
                                                      commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* expertTokenNumGM =
        (uint8_t*)AscendC::GmAlloc(commonTilingInfo.E_ep * commonTilingInfo.epWorldSize * sizeof(uint16_t));
    uint8_t* mmxGM = nullptr;
    uint8_t* mmweightGM = nullptr;
    uint8_t* gmmyGM = (uint8_t*)AscendC::GmAlloc(commonTilingInfo.A * commonTilingInfo.N1 * sizeof(uint16_t));
    uint8_t* mmyGM = nullptr;
    uint8_t* allGatherOutGM = nullptr;
    uint8_t* alltoAllvOutGM = nullptr;

    ICPU_SET_TILING_KEY(1111);
    auto alltoAllvGroupedMatMulWrapper = [](
                                                                 GM_ADDR gmmxGM, GM_ADDR gmmweightGM,
                                                                 GM_ADDR expertTokenNumGM, GM_ADDR mmxGM,
                                                                 GM_ADDR mmweightGM, GM_ADDR gmmyGM, GM_ADDR mmyGM,
                                                                 GM_ADDR allGatherOutGM, GM_ADDR alltoAllvOutGM,
                                                                 GM_ADDR workspaceGM, GM_ADDR tilingGM
    ) {
        allto_allv_grouped_mat_mul<0, false, false, false>(gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM,
                                                            mmweightGM, gmmyGM, mmyGM,
                                                            allGatherOutGM, alltoAllvOutGM, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(alltoAllvGroupedMatMulWrapper, 20, gmmxGM, gmmweightGM, expertTokenNumGM, mmxGM, mmweightGM, gmmyGM, mmyGM,
                allGatherOutGM, alltoAllvOutGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)gmmxGM);
    AscendC::GmFree((void*)gmmweightGM);
    AscendC::GmFree((void*)expertTokenNumGM);
    AscendC::GmFree((void*)gmmyGM);
}