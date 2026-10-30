/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// arch32 (non-RegBase: ascend310p / ascend910_93 / ascend910b) tiling UT. Each case fakes a
// non-RegBase SoC ("ascend910b") via the histogram_v2_ut::RunTilingWithSoc helper, so
// IsRegbaseSocVersion() is false and the HistogramV2MembaseTiling template is selected regardless
// of the build SoC. This is compiled unconditionally alongside the arch35 (SIMT) UT, so a single
// build (any SoC) covers both templates. It covers the MemBase-specific tiling branches; MemBase
// GetShapeAttrsInfo does not validate inputs, so input-validation failure cases live only in arch35.

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

#include "../histogram_v2_tiling_ut_util.h"
#include "../../../../op_host/histogram_v2_tiling.h"

using namespace ge;
using namespace std;

namespace {
using AV = Ops::Math::AnyValue;
using TD = gert::TilingContextPara::TensorDescription;

class HistogramV2MembaseTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HistogramV2MembaseTilingTest (arch32/MemBase) SetUp" << std::endl; }
    static void TearDownTestCase()
    {
        std::cout << "HistogramV2MembaseTilingTest (arch32/MemBase) TearDown" << std::endl;
    }
};

// Fake a non-RegBase SoC so IsRegbaseSocVersion() is false and HistogramV2MembaseTiling is selected
// regardless of the build SoC — lets the MemBase tiling be covered in the same single build as the
// SIMT (arch35) UT. Defaults to "ascend910b"; the 310P-specific branch reads NpuArch from the
// platform (HistogramV2BaseClass::GetPlatformInfo), so that case passes "ascend310p" explicitly.
static bool RunTiling(const gert::TilingContextPara& para, uint64_t& tilingKey, const std::string& soc = "ascend910b")
{
    return histogram_v2_ut::RunTilingWithSoc(para, soc, tilingKey);
}

static bool RunTiling(const gert::TilingContextPara& para, const std::string& soc = "ascend910b")
{
    uint64_t key = 0;
    return RunTiling(para, key, soc);
}

// Build a HistogramV2 tiling param with 3 inputs (x, min, max) + 1 output (y).
static gert::TilingContextPara MakePara(void* compileInfo, const std::vector<int64_t>& xDims, ge::DataType xDtype,
                                        const std::vector<int64_t>& mmDims, ge::DataType mmDtype,
                                        const std::vector<int64_t>& yDims, ge::DataType yDtype, int64_t bins)
{
    gert::StorageShape xShape;
    for (auto d : xDims) {
        xShape.MutableOriginShape().AppendDim(d);
        xShape.MutableStorageShape().AppendDim(d);
    }
    gert::StorageShape mmShape;
    for (auto d : mmDims) {
        mmShape.MutableOriginShape().AppendDim(d);
        mmShape.MutableStorageShape().AppendDim(d);
    }
    gert::StorageShape yShape;
    for (auto d : yDims) {
        yShape.MutableOriginShape().AppendDim(d);
        yShape.MutableStorageShape().AppendDim(d);
    }
    auto para = gert::TilingContextPara("HistogramV2",
                                        {
                                            TD(xShape, xDtype, ge::FORMAT_ND),
                                            TD(mmShape, mmDtype, ge::FORMAT_ND),
                                            TD(mmShape, mmDtype, ge::FORMAT_ND),
                                        },
                                        {
                                            TD(yShape, yDtype, ge::FORMAT_ND),
                                        },
                                        {
                                            gert::TilingContextPara::OpAttr("bins", AV::CreateFrom<int64_t>(bins)),
                                        },
                                        compileInfo);
    return para;
}

static optiling::HistogramV2CompileInfo MakeCompileInfo(int64_t coreNum = 64, NpuArch arch = NpuArch::DAV_2201)
{
    optiling::HistogramV2CompileInfo ci;
    ci.totalCoreNum = static_cast<int32_t>(coreNum);
    ci.ubSizePlatForm = 262144;
    ci.sysWorkspaceSize = 16 * 1024 * 1024;
    ci.npuArch = arch;
    return ci;
}

// ---------------------- MemBase valid cases ----------------------

TEST_F(HistogramV2MembaseTilingTest, tiling_fp32_ub_full)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_FLOAT, {1}, ge::DT_FLOAT, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

// dtype cases below exercise the MemBase SetTilingKeyMode dtype -> TilingKey dispatch
// (HISTOGRAM_V2_FP16 / INT32 / INT8 / UINT8 / INT16); int64 is covered separately below.
TEST_F(HistogramV2MembaseTilingTest, tiling_fp16)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_FLOAT16, {1}, ge::DT_FLOAT16, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

TEST_F(HistogramV2MembaseTilingTest, tiling_int32)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_INT32, {1}, ge::DT_INT32, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

TEST_F(HistogramV2MembaseTilingTest, tiling_int8)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_INT8, {1}, ge::DT_INT8, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

TEST_F(HistogramV2MembaseTilingTest, tiling_uint8)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_UINT8, {1}, ge::DT_UINT8, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

TEST_F(HistogramV2MembaseTilingTest, tiling_int16)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_INT16, {1}, ge::DT_INT16, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

// int64 exercises the MemBase TilingDataInCore int64 branch (tileLength / 2).
TEST_F(HistogramV2MembaseTilingTest, tiling_int64)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_INT64, {1}, ge::DT_INT64, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

// npuArch DAV_2002 exercises the MemBase 310P branch (ubSelfLength_310P, userWorkspaceSize, ScheduleMode).
TEST_F(HistogramV2MembaseTilingTest, tiling_membase_310p_branch)
{
    auto ci = MakeCompileInfo(64, NpuArch::DAV_2002);
    auto para = MakePara(&ci, {256}, ge::DT_FLOAT, {1}, ge::DT_FLOAT, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para, "ascend310p"));
}

// totalLength < coreNum -> MemBase tailLength == 0 branch (coreNum reduced to 1).
TEST_F(HistogramV2MembaseTilingTest, tiling_tail_length_zero)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {1}, ge::DT_FLOAT, {1}, ge::DT_FLOAT, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

} // namespace
