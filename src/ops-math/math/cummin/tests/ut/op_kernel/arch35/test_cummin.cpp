/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_cummin.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/cummin_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void cummin(GM_ADDR x, GM_ADDR y, GM_ADDR argmin, GM_ADDR workspace, GM_ADDR tiling);

class CumminTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "cummin_test SetUp" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "cummin_test TearDown" << std::endl;
    }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b) {
    return (a + b - 1) / b * b;
}

TEST_F(CumminTest, test_case_float16_1) {
    optiling::CumminCompileInfo compileInfo = {56, 253952, 32, 256, 256};
    gert::TilingContextPara tilingContextPara("Cummin",
                                            {
                                                {{{128, 64}, {128, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{128, 64}, {128, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{128, 64}, {128, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
                                            },
                                            &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    uint32_t dataCount = 128 * 64;
    size_t inputByteSize = dataCount * sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    uint8_t* argmin = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(0);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(optiling::CumminCompileInfo));

    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(cummin, tilingInfo.blockNum, x, y, workspace, tiling);


    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)(argmin));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

}
