/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// arch35 (RegBase / ascend950) tiling UT. Each case fakes a RegBase SoC ("ascend950") via the
// histogram_v2_ut::RunTilingWithSoc helper, so IsRegbaseSocVersion() is true and the
// HistogramV2SimtTiling template is selected regardless of the build SoC. This is compiled
// unconditionally alongside the arch32 (MemBase) UT, so a single build (any SoC) covers both
// the SIMT and MemBase tiling templates without needing --soc=ascend950.

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

class HistogramV2SimtTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HistogramV2SimtTilingTest (arch35/SIMT) SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "HistogramV2SimtTilingTest (arch35/SIMT) TearDown" << std::endl; }
};

// Fake a RegBase SoC ("ascend950") so IsRegbaseSocVersion() is true and HistogramV2SimtTiling is
// selected regardless of the build SoC — lets the SIMT tiling be covered in a single build.
static bool RunTiling(const gert::TilingContextPara& para, uint64_t& tilingKey)
{
    return histogram_v2_ut::RunTilingWithSoc(para, "ascend950", tilingKey);
}

static bool RunTiling(const gert::TilingContextPara& para)
{
    uint64_t key = 0;
    return RunTiling(para, key);
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

// ---------------------- valid cases ----------------------

TEST_F(HistogramV2SimtTilingTest, tiling_fp32_ub_full)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_FLOAT, {1}, ge::DT_FLOAT, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

TEST_F(HistogramV2SimtTilingTest, tiling_fp16)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_FLOAT16, {1}, ge::DT_FLOAT16, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

TEST_F(HistogramV2SimtTilingTest, tiling_int32)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_INT32, {1}, ge::DT_INT32, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

TEST_F(HistogramV2SimtTilingTest, tiling_int8)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_INT8, {1}, ge::DT_INT8, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

TEST_F(HistogramV2SimtTilingTest, tiling_uint8)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_UINT8, {1}, ge::DT_UINT8, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

TEST_F(HistogramV2SimtTilingTest, tiling_int16)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_INT16, {1}, ge::DT_INT16, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

// int64 exercises the SIMT dtype val 6 path.
TEST_F(HistogramV2SimtTilingTest, tiling_int64)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_INT64, {1}, ge::DT_INT64, {100}, ge::DT_INT32, 100);
    EXPECT_TRUE(RunTiling(para));
}

// fp32 output for fp16 input (SIMT fp32-out key offset path).
TEST_F(HistogramV2SimtTilingTest, tiling_fp16_out_fp32)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_FLOAT16, {1}, ge::DT_FLOAT16, {100}, ge::DT_FLOAT, 100);
    EXPECT_TRUE(RunTiling(para));
}

// fp32 output for fp32 input.
TEST_F(HistogramV2SimtTilingTest, tiling_fp32_out_fp32)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_FLOAT, {1}, ge::DT_FLOAT, {100}, ge::DT_FLOAT, 100);
    EXPECT_TRUE(RunTiling(para));
}

// bins >= ubNumCanUse and totalLength > bins/100 -> SIMT UB_NOT_FULL branch.
TEST_F(HistogramV2SimtTilingTest, tiling_ub_not_full)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {1000}, ge::DT_FLOAT, {1}, ge::DT_FLOAT, {60000}, ge::DT_INT32, 60000);
    EXPECT_TRUE(RunTiling(para));
}

// bins >= ubNumCanUse and totalLength <= bins/100 -> SIMT UB_NOT_FULL_SIMT branch.
TEST_F(HistogramV2SimtTilingTest, tiling_ub_not_full_simt)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {100}, ge::DT_FLOAT, {1}, ge::DT_FLOAT, {60000}, ge::DT_INT32, 60000);
    EXPECT_TRUE(RunTiling(para));
}

// ---------------------- failure cases (SIMT GetShapeAttrsInfo validation) ----------------------

TEST_F(HistogramV2SimtTilingTest, tiling_bins_non_positive)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_FLOAT, {1}, ge::DT_FLOAT, {1}, ge::DT_INT32, -1);
    EXPECT_FALSE(RunTiling(para));
}

TEST_F(HistogramV2SimtTilingTest, tiling_minmax_shape_invalid)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_FLOAT, {2}, ge::DT_FLOAT, {100}, ge::DT_INT32, 100);
    EXPECT_FALSE(RunTiling(para));
}

TEST_F(HistogramV2SimtTilingTest, tiling_minmax_dtype_mismatch)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_FLOAT, {1}, ge::DT_FLOAT16, {100}, ge::DT_INT32, 100);
    EXPECT_FALSE(RunTiling(para));
}

TEST_F(HistogramV2SimtTilingTest, tiling_unsupported_dtype)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_BF16, {1}, ge::DT_BF16, {100}, ge::DT_INT32, 100);
    EXPECT_FALSE(RunTiling(para));
}

TEST_F(HistogramV2SimtTilingTest, tiling_out_size_ne_bins)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_FLOAT, {1}, ge::DT_FLOAT, {50}, ge::DT_INT32, 100);
    EXPECT_FALSE(RunTiling(para));
}

TEST_F(HistogramV2SimtTilingTest, tiling_fp32_out_int_input)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_INT32, {1}, ge::DT_INT32, {100}, ge::DT_FLOAT, 100);
    EXPECT_FALSE(RunTiling(para));
}

TEST_F(HistogramV2SimtTilingTest, tiling_out_dtype_invalid)
{
    auto ci = MakeCompileInfo();
    auto para = MakePara(&ci, {256}, ge::DT_INT32, {1}, ge::DT_INT32, {100}, ge::DT_FLOAT16, 100);
    EXPECT_FALSE(RunTiling(para));
}

TEST_F(HistogramV2SimtTilingTest, tiling_core_num_zero)
{
    auto ci = MakeCompileInfo(0);
    auto para = MakePara(&ci, {256}, ge::DT_FLOAT, {1}, ge::DT_FLOAT, {100}, ge::DT_INT32, 100);
    EXPECT_FALSE(RunTiling(para));
}

} // namespace
