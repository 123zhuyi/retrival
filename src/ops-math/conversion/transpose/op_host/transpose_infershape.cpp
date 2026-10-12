/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file transpose_infershape.cpp
 * @brief Transpose 算子的 Shape 推导实现
 *
 * 核心逻辑：TransposeInferCommon<T>
 *   1. 输出维度数 = 输入维度数
 *   2. 若 _inserted_by_fe == 0（非FE插入）：yShape[i] = xShape[perm[i]]（支持负数 perm）
 *   3. 若 _inserted_by_fe != 0（FE插入）：yShape[i] = xShape[i]（shape 不变）
 *   4. 支持 INT32 和 INT64 两种 perm 数据类型
 *
 * _inserted_by_fe 特殊逻辑说明：
 *   当 FE（Frontend）在构图时插入 Transpose 节点做格式转换时，输出 shape 与输入相同，
 *   不需要按 perm 重排 shape。此属性通过 IR 的私有属性 _inserted_by_fe 传入，默认值为0。
 *
 * 负值 perm 处理：
 *   perm[i] < 0 时映射为 perm[i] + inputDimSize，例如 perm=[-1] 表示最后一个维度。
 */
#include <graph/utils/type_utils.h>
#include "util/math_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_api/op_util.h"

using namespace ge;
namespace ops {
constexpr size_t TRANSPOSE_IDX_IN_X = 0;
constexpr size_t TRANSPOSE_IDX_IN_PERM = 1;
constexpr size_t TRANSPOSE_IDX_OUT_Y = 0;

/**
 * @brief Transpose Shape 推导通用函数
 *
 * 根据 perm 数组和 _inserted_by_fe 属性推导输出 shape：
 * - _inserted_by_fe == 0（默认）：yShape[i] = xShape[permV]，permV 支持负值
 * - _inserted_by_fe != 0（FE插入）：yShape[i] = xShape[i]，shape 不变
 *
 * @tparam T         perm 数据类型（int32_t 或 int64_t）
 * @param context    InferShape 上下文
 * @param xShape     输入 shape
 * @param permValue  perm 数组指针
 * @param yShape     [out] 输出 shape
 * @return true 推导成功；false 推导失败（perm 值越界）
 */
template <typename T>
static bool TransposeInferCommon(const gert::InferShapeContext* context, const gert::Shape* xShape, const T* permValue,
                                 gert::Shape* yShape)
{
    OP_LOGD(context->GetNodeName(), "start to do TransposeInferCommon");
    size_t inputDimSize = xShape->GetDimNum();
    yShape->SetDimNum(inputDimSize);
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    // 读取 _inserted_by_fe 私有属性（默认值0）
    // 当 FE 在构图时插入 Transpose 节点做格式转换时，该值非0，此时输出 shape = 输入 shape
    int64_t insertedByFe = 0;
    if (attrs->GetAttrNum() > 0) {
        const int64_t* insertedByFeFlag = attrs->GetInt(0);
        insertedByFe = insertedByFeFlag == nullptr ? 0 : *insertedByFeFlag;
    }
    if (insertedByFe == 0) {
        // 正常 Transpose 语义：按 perm 重排 shape
        for (size_t i = 0; i < inputDimSize; ++i) {
            OP_CHECK_IF(!IsDimValid(inputDimSize, permValue[i]),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context->GetNodeName(), "perm", std::to_string(permValue[i]).c_str(),
                            "Each value of perm must be in the range of [-xShapeDimNum, xShapeDimNum - 1]."
                            " The value of perm depends on the number of shape axes of x"),
                        return false);
            // 负值 perm 处理：perm[i] < 0 时映射为 perm[i] + inputDimSize
            T permV = permValue[i] < 0 ? permValue[i] + inputDimSize : permValue[i];
            yShape->SetDim(i, xShape->GetDim(permV)); // yShape[i] = xShape[perm[i]]
        }
    } else {
        // FE 插入场景：输出 shape = 输入 shape（不做 perm 重排）
        for (size_t i = 0; i < inputDimSize; ++i) {
            yShape->SetDim(i, xShape->GetDim(i));
        }
    }

    OP_LOGD(context->GetNodeName(), "end to do TransposeInferCommon");
    return true;
}

/**
 * @brief Transpose InferShape 入口函数
 *
 * 根据 perm 的数据类型（INT32 或 INT64）分发到 TransposeInferCommon<T>。
 * 同时校验 perm 大小必须等于输入维度数。
 */
static ge::graphStatus TransposeInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do TransposeInferShape");
    const gert::Shape* xShape = context->GetInputShape(TRANSPOSE_IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(TRANSPOSE_IDX_OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    const gert::Tensor* permTensor = context->GetInputTensor(TRANSPOSE_IDX_IN_PERM);
    OP_CHECK_NULL_WITH_CONTEXT(context, permTensor);

    // 校验 perm 大小必须等于输入维度数
    int64_t permSize = permTensor->GetShapeSize();
    size_t inputDimSize = xShape->GetDimNum();
    OP_CHECK_IF(permSize != static_cast<int64_t>(inputDimSize),
                OP_LOGE_FOR_INVALID_SHAPESIZE(context->GetNodeName(), "perm", ConcatString(permSize).c_str(),
                                              ConcatString(inputDimSize).c_str()),
                return ge::GRAPH_FAILED);

    // 按 perm 数据类型分发（INT32 或 INT64）
    ge::DataType permDtype = permTensor->GetDataType();
    switch (permDtype) {
        case ge::DT_INT32: {
            const int32_t* permValue = permTensor->GetData<int32_t>();
            if (!TransposeInferCommon(context, xShape, permValue, yShape)) {
                return ge::GRAPH_FAILED;
            }
            break;
        }
        case ge::DT_INT64: {
            const int64_t* permValue = permTensor->GetData<int64_t>();
            if (!TransposeInferCommon(context, xShape, permValue, yShape)) {
                return ge::GRAPH_FAILED;
            }
            break;
        }
        default:
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "perm", Ops::Base::ToString(permDtype).c_str(),
                                      "int32 or int64");
            return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "End to do TransposeInferShape");
    return ge::GRAPH_SUCCESS;
}
static int64_t privateDefaultValue = 0; ///< _inserted_by_fe 默认值
IMPL_OP_INFERSHAPE(Transpose)
    .InferShape(TransposeInferShape)
    .InputsDataDependency({TRANSPOSE_IDX_IN_PERM})        // perm 输入作为数据依赖
    .PrivateAttr("_inserted_by_fe", privateDefaultValue); // 注册私有属性：FE插入标志
} // namespace ops
