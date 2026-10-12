/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "tiling_case_executor.h"
#include "../../../op_kernel/bernoulli_mask_tiling_data.h"
#include "../../../op_kernel/bernoulli_mask_tiling_key.h"

namespace {
constexpr uint32_t A2_VECTOR_CORE_NUM = 40;
constexpr uint64_t A2_UB_BYTES = 192 * 1024;
constexpr uint64_t MAX_TILE_ELEMENTS = 16 * 1024;
constexpr uint64_t DOUBLE_TILE_ELEMENTS = 14 * 1024;
constexpr uint64_t MASK_ALIGN_ELEMENTS = 256;

struct BernoulliMaskCompileInfoForTest {};

uint64_t CeilDiv(uint64_t value, uint64_t divisor) { return divisor == 0 ? 0 : (value + divisor - 1) / divisor; }

uint64_t AlignUp(uint64_t value, uint64_t alignment) { return CeilDiv(value, alignment) * alignment; }

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t dim : dims) {
        shape.MutableShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

gert::TilingContextPara MakeContext(int64_t elements, ge::DataType outputDtype,
                                    BernoulliMaskCompileInfoForTest* compileInfo, uint32_t coreNum = A2_VECTOR_CORE_NUM,
                                    uint64_t ubBytes = A2_UB_BYTES, bool maskAliasesOut = false)
{
    gert::StorageShape maskShape = MakeShape({elements});
    gert::StorageShape outputShape = MakeShape({elements});
    return gert::TilingContextPara(
        "BernoulliMask",
        {
            {maskShape, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {outputShape, outputDtype, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("output_shape",
                                            Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({elements})),
            gert::TilingContextPara::OpAttr("mask_aliases_out",
                                            Ops::Math::AnyValue::CreateFrom<int64_t>(maskAliasesOut ? 1 : 0)),
        },
        compileInfo, coreNum, ubBytes);
}

void CheckSuccess(int64_t elements, ge::DataType dtype, uint64_t expectedKey, uint64_t expectedTileElements,
                  uint32_t expectedBlockDim, uint64_t expectedElementsPerCore, bool maskAliasesOut = false)
{
    BernoulliMaskCompileInfoForTest compileInfo;
    auto context = MakeContext(elements, dtype, &compileInfo, A2_VECTOR_CORE_NUM, A2_UB_BYTES, maskAliasesOut);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    ASSERT_EQ(info.tilingDataSize, sizeof(optiling::BernoulliMaskTilingData));
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(info.workspaceSizes[0], 0U);
    EXPECT_EQ(info.tilingKey, expectedKey);
    EXPECT_EQ(info.blockNum, expectedBlockDim);

    const auto* data = reinterpret_cast<const optiling::BernoulliMaskTilingData*>(info.tilingData.get());
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->totalElements, static_cast<uint64_t>(elements));
    EXPECT_EQ(data->elementsPerCore, expectedElementsPerCore);
    EXPECT_EQ(data->tileElements, expectedTileElements);
    EXPECT_EQ(data->maskAliasesOut, maskAliasesOut ? 1U : 0U);
}

struct DtypeCase {
    const char* name;
    ge::DataType dtype;
    uint64_t tilingKey;
    uint64_t tileElements;
};

class BernoulliMaskDtypeTiling : public testing::TestWithParam<DtypeCase> {};

TEST_P(BernoulliMaskDtypeTiling, maps_supported_dtype_to_kernel_key)
{
    const auto& param = GetParam();
    CheckSuccess(129, param.dtype, param.tilingKey, param.tileElements, 1, MASK_ALIGN_ELEMENTS);
}

INSTANTIATE_TEST_SUITE_P(
    Ascend910B, BernoulliMaskDtypeTiling,
    testing::Values(DtypeCase{"fp16", ge::DT_FLOAT16, BernoulliMaskKey::FLOAT16, MAX_TILE_ELEMENTS},
                    DtypeCase{"fp32", ge::DT_FLOAT, BernoulliMaskKey::FLOAT, MAX_TILE_ELEMENTS},
                    DtypeCase{"fp64", ge::DT_DOUBLE, BernoulliMaskKey::DOUBLE, DOUBLE_TILE_ELEMENTS},
                    DtypeCase{"uint8", ge::DT_UINT8, BernoulliMaskKey::UINT8_OR_BOOL, MAX_TILE_ELEMENTS},
                    DtypeCase{"int8", ge::DT_INT8, BernoulliMaskKey::INT8, MAX_TILE_ELEMENTS},
                    DtypeCase{"int16", ge::DT_INT16, BernoulliMaskKey::INT16, MAX_TILE_ELEMENTS},
                    DtypeCase{"int32", ge::DT_INT32, BernoulliMaskKey::INT32, MAX_TILE_ELEMENTS},
                    DtypeCase{"int64", ge::DT_INT64, BernoulliMaskKey::INT64, DOUBLE_TILE_ELEMENTS},
                    DtypeCase{"bool", ge::DT_BOOL, BernoulliMaskKey::UINT8_OR_BOOL, MAX_TILE_ELEMENTS},
                    DtypeCase{"bf16", ge::DT_BF16, BernoulliMaskKey::BFLOAT16, MAX_TILE_ELEMENTS}),
    [](const testing::TestParamInfo<DtypeCase>& info) { return std::string(info.param.name); });

class BernoulliMaskBoundaryTiling : public testing::TestWithParam<int64_t> {};

TEST_P(BernoulliMaskBoundaryTiling, handles_packed_mask_and_alignment_boundaries)
{
    const int64_t elements = GetParam();
    const uint32_t blockDim = elements > static_cast<int64_t>(MAX_TILE_ELEMENTS) ? 2U : 1U;
    const uint64_t elementsPerCore = elements == 0 ? 0 :
                                                     AlignUp(CeilDiv(static_cast<uint64_t>(elements), blockDim),
                                                             MASK_ALIGN_ELEMENTS);
    CheckSuccess(elements, ge::DT_FLOAT, BernoulliMaskKey::FLOAT, MAX_TILE_ELEMENTS, blockDim, elementsPerCore);
}

INSTANTIATE_TEST_SUITE_P(BitAndTileEdges, BernoulliMaskBoundaryTiling,
                         testing::Values(0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 129, 16383, 16384, 16385));

TEST(BernoulliMaskTiling, fp64_tile_boundary_uses_two_cores)
{
    CheckSuccess(DOUBLE_TILE_ELEMENTS + 1, ge::DT_DOUBLE, BernoulliMaskKey::DOUBLE, DOUBLE_TILE_ELEMENTS, 2,
                 AlignUp(CeilDiv(DOUBLE_TILE_ELEMENTS + 1, 2), MASK_ALIGN_ELEMENTS));
}

TEST(BernoulliMaskTiling, large_shape_uses_all_a2_vector_cores)
{
    constexpr uint64_t elements = 1000000;
    constexpr uint64_t elementsPerCore = ((elements / A2_VECTOR_CORE_NUM + MASK_ALIGN_ELEMENTS - 1) /
                                          MASK_ALIGN_ELEMENTS) *
                                         MASK_ALIGN_ELEMENTS;
    CheckSuccess(elements, ge::DT_FLOAT, BernoulliMaskKey::FLOAT, MAX_TILE_ELEMENTS, A2_VECTOR_CORE_NUM,
                 elementsPerCore);
}

TEST(BernoulliMaskTiling, records_mask_output_alias_mode)
{
    constexpr uint64_t elements = 1000000;
    constexpr uint64_t elementsPerCore = ((elements / A2_VECTOR_CORE_NUM + MASK_ALIGN_ELEMENTS - 1) /
                                          MASK_ALIGN_ELEMENTS) *
                                         MASK_ALIGN_ELEMENTS;
    CheckSuccess(elements, ge::DT_FLOAT, BernoulliMaskKey::FLOAT, MAX_TILE_ELEMENTS, A2_VECTOR_CORE_NUM,
                 elementsPerCore, true);
}

TEST(BernoulliMaskTiling, rejects_unsupported_output_dtype)
{
    BernoulliMaskCompileInfoForTest compileInfo;
    auto context = MakeContext(128, ge::DT_COMPLEX64, &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(BernoulliMaskTiling, rejects_platform_with_insufficient_ub)
{
    BernoulliMaskCompileInfoForTest compileInfo;
    auto context = MakeContext(128, ge::DT_FLOAT, &compileInfo, A2_VECTOR_CORE_NUM, 8 * 1024);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}
} // namespace
