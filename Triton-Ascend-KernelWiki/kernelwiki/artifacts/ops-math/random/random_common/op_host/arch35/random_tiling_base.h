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
 * \file random_tiling_base.h
 * \brief
 */
#ifndef RANDOM_TILING_BASE_H
#define RANDOM_TILING_BASE_H

#include <random>
#include <chrono>
#include <thread>
#include <cstdint>
#include <graph/utils/type_utils.h>
#include "random_tiling_arch35.h"
namespace optiling {

static inline std::mt19937_64& GetGlobalRng() {
    static std::mt19937_64 rng([]() -> uint64_t {
        auto now =std::chrono::high_resolution_clock::now();
        uint64_t seed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
        ).count();

        seed ^= std::hash<std::thread::id>()(std::this_thread::get_id());
        return seed;
    }());

    return rng;
}


inline uint64_t New64() {
    return GetGlobalRng()();
}

namespace RandomUtils {

template<int INPUT_INDEX, int OUTPUT_INDEX, bool CHECK_OUTPUT_SIZE = true>
ge::graphStatus GetAndCheckOutputSize(gert::TilingContext* ctx, int64_t& shapeSize)
{
    gert::Shape constShape;
    auto ret = ExtractTensorValue(ctx, INPUT_INDEX, constShape);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(ctx->GetNodeName(), "GetAndCheckOutputSize failed");
        return ret;
    }

    shapeSize = 1;
    uint32_t shapeRank = constShape.GetDimNum();
    for (uint32_t idx = 0; idx < shapeRank; idx++) {
        shapeSize *= static_cast<int64_t>(constShape.GetDim(idx));
    }
    if(CHECK_OUTPUT_SIZE) {
        OP_CHECK_IF(shapeSize == 0,
            OP_LOGE(ctx->GetNodeName(), "input shape should not be empty tensor."), 
            return ge::GRAPH_FAILED);
    }

    auto outputShape = ctx->GetOutputShape(OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, outputShape);
    auto outTensor = outputShape->GetStorageShape();
    int64_t outputSize = outTensor.GetShapeSize();
    OP_CHECK_IF(shapeSize != outputSize,
        OP_LOGE(ctx->GetNodeName(), "shape size:%ld is not equal to out size:%ld.", shapeSize, outputSize), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

template<int SEED_INDEX, int SEED2_INDEX>
ge::graphStatus GetKeyAndCounter(gert::TilingContext* ctx, uint32_t key[2], uint32_t counter[4])
{
    auto attrs = ctx->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(ctx, attrs);
    const auto* seedAttr = attrs->GetAttrPointer<int64_t>(SEED_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, seedAttr);
    const auto* seed2Attr = attrs->GetAttrPointer<int64_t>(SEED2_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, seed2Attr);
    
    int64_t seed = *seedAttr;
    int64_t seed2 = *seed2Attr;
    if (seed == 0 && seed2 == 0) {
        seed = static_cast<int64_t>(New64());
        seed2 = static_cast<int64_t>(New64());
    }
    constexpr uint32_t SHIFT_BITS = 32;
    key[0] = static_cast<uint32_t>(seed);
    key[1] = static_cast<uint32_t>(seed >> SHIFT_BITS);
    counter[0] = 0;
    counter[1] = 0;
    counter[2] = static_cast<uint32_t>(seed2);
    counter[3] = static_cast<uint32_t>(seed2 >> SHIFT_BITS);

    return ge::GRAPH_SUCCESS;
}

template <typename T>
std::string GetShapeStr(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

inline std::string GetTensorStr(
    const gert::StorageShape* shape, const gert::CompileTimeTensorDesc* tensor)
{
    if (shape == nullptr || tensor == nullptr) {
        return "nil ";
    }
    std::ostringstream oss;
    oss << "(dtype: " << ge::TypeUtils::DataTypeToSerialString(tensor->GetDataType()) << "),";
    oss << "(shape:" << GetShapeStr(shape->GetStorageShape()) << "),";
    oss << "(ori_shape:" << GetShapeStr(shape->GetOriginShape()) << "),";
    oss << "(format: "
        << ge::TypeUtils::FormatToSerialString(
                static_cast<ge::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat())))
        << "),";
    oss << "(ori_format: " << ge::TypeUtils::FormatToSerialString(tensor->GetOriginFormat()) << ") ";
    return oss.str();
}

inline std::string GetTilingContext(gert::TilingContext* ctx)
{
    std::ostringstream oss;
    for (size_t i = 0; i < ctx->GetComputeNodeInfo()->GetInputsNum(); ++i) {
        oss << "input" << i << ": ";
        oss << GetTensorStr(ctx->GetInputShape(i), ctx->GetInputDesc(i));
    }

    for (size_t i = 0; i < ctx->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
        oss << "output" << i << ": ";
        oss << GetTensorStr(ctx->GetOutputShape(i), ctx->GetOutputDesc(i));
    }
    return oss.str();
}
}

} // namespace optiling
#endif // RANDOM_TILING_BASE_H