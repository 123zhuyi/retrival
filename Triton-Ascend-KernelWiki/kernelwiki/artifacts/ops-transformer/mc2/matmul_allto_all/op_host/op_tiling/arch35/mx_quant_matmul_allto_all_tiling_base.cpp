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
 * \file mx_quant_matmul_allto_all_tiling_base.cpp
 * \brief
 */
#include "op_mc2.h"
#include "mc2_log.h"
#include "mx_quant_matmul_allto_all_tiling_base.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;

namespace MC2Tiling {

/**
 * @brief 工具函数：判断指定value是否存在于list中
 *
 * @param list: 有效值列表
 * @param value: 给定值
 * @return
 */
static bool IsContains(const std::vector<uint32_t> &list, uint32_t value)
{
    return std::count(list.begin(), list.end(), value) > 0;
}

gert::StorageShape mxQuantStorageShape = gert::StorageShape();

/**
 * @brief 当前量化过程的准入条件
 * @return true
 */
bool MxQuantMatmulAllToAllTilingBase::IsCapable()
{
    int32_t x1QuantMode = 0;
    int32_t x2QuantMode = 0;
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    if (const int *ptr = attrs->GetAttrPointer<int>(ATTR_X1_QUANTMODE_INDEX)) {
        x1QuantMode = *ptr;
    }
    if (const int *ptr = attrs->GetAttrPointer<int>(ATTR_X2_QUANTMODE_INDEX)) {
        x2QuantMode = *ptr;
    }
    if (x1QuantMode == X1_QUANTMODE_VALUES && x2QuantMode == X2_QUANTMODE_VALUES) {
        OP_LOGI(opName_, "Start with MxQuantMatmulAlltoAll tiling.");
        return true;
    }
    OP_LOGI(opName_, "Skip MxQuantMatmulAlltoAll tiling when not MX_QUANT.");
    return false;
}

/**
 * @brief 校验输入信息是否合规:attr,Dtype,shape等，使用通用校验util中的check方法
 *
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::CheckOpInputInfo()
{
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckAttrsInfo(context_, opName_, MATMUL_ALLTOALL_INDEX_SCHEMA) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check Attrs failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckX2Transpose(context_, opName_, MATMUL_ALLTOALL_INDEX_SCHEMA) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check x2transpose failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckMxQuantTensorDataType(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "tiling check Dtype failed in mx quant matmul all to all."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckMxQuantShapeInfo(context_, opName_, MATMUL_ALLTOALL_INDEX_SCHEMA) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check mx quant shape info failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckMxQuantMatrixMulShapes(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check mx quant matrix shape failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 量化场景校验x2transpose是否一定为true
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName  算子名称
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::CheckX2Transpose(const gert::TilingContext *context, const char *opName, const OpAttrIndexSchema &indexSchema)
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    const bool *isTransX2 = attrs->GetAttrPointer<bool>(indexSchema.x2Transpose);
    OP_TILING_CHECK(!(*isTransX2), OP_LOGE(opName, "the mx quant input x2Transpose must be true, but actual is false."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 根据输入设置tiling参数
 *
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::InitTilingContextParameters()
{
    GE_ASSERT_GRAPH_SUCCESS(
        MatmulAlltoAllTilingUtil::SetAttrsInfo(context_, opName_, contextInfo, MATMUL_ALLTOALL_INDEX_SCHEMA));
    GE_ASSERT_GRAPH_SUCCESS(SetMxDataTypeInfo(context_, opName_, contextInfo)); 
    GE_ASSERT_GRAPH_SUCCESS(MatmulAlltoAllTilingUtil::SetShapeInfo(context_, contextInfo));
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置hccl参数；进行通算切分, 获取mm tiling等
 *
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::DoOpTiling()
{
    // 输入参数的校验:Attrs,Dtype,Shape等
    GE_ASSERT_GRAPH_SUCCESS(CheckOpInputInfo());
    // 参数校验通过后赋值给全局上下文变量
    GE_ASSERT_GRAPH_SUCCESS(InitTilingContextParameters());
    // 进行通算切分
    GE_ASSERT_GRAPH_SUCCESS(TileCommAndCompute());
    // 调用量化Matmul的tiling方法进行切分
    GE_ASSERT_GRAPH_SUCCESS(DoMxQuantMMTiling());
    // hccl的tiling参数赋值处理
    GE_ASSERT_GRAPH_SUCCESS(SetHcclTiling());
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 量化场景校验参数的DType
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName  算子名称
 * @param runInfo 过程信息
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::CheckMxQuantTensorDataType(const gert::TilingContext *context, const char *opName)
{
    // 获取并校验输入张量描述符
    auto x1TensorDesc = context->GetInputDesc(INPUT_X1_INDEX);
    auto x2TensorDesc = context->GetInputDesc(INPUT_X2_INDEX);
    OP_TILING_CHECK((x1TensorDesc == nullptr), OP_LOGE(opName, "the input x1 tensor is invalid."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x2TensorDesc == nullptr), OP_LOGE(opName, "the input x2 tensor is invalid."),
                    return ge::GRAPH_FAILED);
    // 获取数据类型并校验一致性与范围
    ge::DataType x1Dtype = x1TensorDesc->GetDataType();
    ge::DataType x2Dtype = x2TensorDesc->GetDataType();
    OP_TILING_CHECK(!IsContains(MX_QUANT_X_DTYPE_LIST, x1Dtype),
                    OP_LOGE(opName,
                            "The Input x1 Dtype should be in mx-quant range (float8_e4m3fn/float8_e5m2), but x1 is %s.",
                            Ops::Base::ToString(x1Dtype).c_str()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(MX_QUANT_X_DTYPE_LIST, x2Dtype),
                    OP_LOGE(opName,
                            "The Input x2 Dtype should be in mx-quant range (float8_e4m3fn/float8_e5m2), but x2 is %s.",
                            Ops::Base::ToString(x2Dtype).c_str()), return ge::GRAPH_FAILED);
    // 校验 bias 数据类型（如果存在）
    auto biasTensorDesc = context->GetOptionalInputDesc(INPUT_BIAS_INDEX);
    QuantMode mode = MatmulAlltoAllTilingUtil::GetQuantMode(context, opName);
    if (biasTensorDesc != nullptr) {
        ge::DataType biasDtype = biasTensorDesc->GetDataType();
        OP_TILING_CHECK(
            (biasDtype != ge::DT_FLOAT),
            OP_LOGE(opName, "bias Dtype should be float, but bias is %s.", Ops::Base::ToString(biasDtype).c_str()),
            return ge::GRAPH_FAILED);
    }
    // 校验 scale 张量不为空（量化场景）
    auto x1ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X1_SCALE_INDEX);
    auto x2ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X2_SCALE_INDEX);
    OP_TILING_CHECK((x1ScaleTensorDesc == nullptr),
                    OP_LOGE(opName, "x1scale tensors should not be null in mx quant mode."), return ge::GRAPH_FAILED);
    ge::DataType x1scaleDtype = x1ScaleTensorDesc->GetDataType();
    OP_TILING_CHECK((x1scaleDtype != ge::DataType::DT_FLOAT8_E8M0),
                    OP_LOGE(opName, "x1scale dtype should be DT_FLOAT8_E8M0 in mx quant mode, but is %s.",
                            Ops::Base::ToString(x1scaleDtype).c_str()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x2ScaleTensorDesc == nullptr),
                    OP_LOGE(opName, "x2scale tensors should not be null in mx quant mode."), return ge::GRAPH_FAILED);
    ge::DataType x2scaleDtype = x2ScaleTensorDesc->GetDataType();
    OP_TILING_CHECK((x2scaleDtype != ge::DataType::DT_FLOAT8_E8M0),
                    OP_LOGE(opName, "x2scale dtype should be DT_FLOAT8_E8M0 in mx quant mode, but is %s.",
                            Ops::Base::ToString(x2scaleDtype).c_str()), return ge::GRAPH_FAILED);
    // 校验输出张量数据类型
    auto yDesc = context->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK((yDesc == nullptr), OP_LOGE(opName, "output tensor y is nullptr."), return ge::GRAPH_FAILED);
    ge::DataType yDtype = yDesc->GetDataType();
    OP_TILING_CHECK(!IsContains(MX_QUANT_Y_DTYPE_LIST, yDtype),
                    OP_LOGE(opName, "output y Dtype should be float16, bfloat16 or float, but y is %s.",
                            Ops::Base::ToString(yDtype).c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验量化tiling scale shape的Dim数量信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @return ge::graphStatus
 */
static ge::graphStatus CheckMxScaleShapeDimensions(const gert::StorageShape *shape, const char *shapeName,
                                                 const char *opName)
{
    uint64_t dimNum = shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK((dimNum != 3), OP_LOGE(opName, "the %s dimNum should be three.", shapeName), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tiling shape的Dim数量信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @return ge::graphStatus
 */
static ge::graphStatus CheckShapeDimensions(const gert::StorageShape *shape, const char *shapeName, const char *opName)
{
    uint64_t dimNum = shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK((dimNum != 2), OP_LOGE(opName, "The %s dimNum should be two.", shapeName), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tiling输入的bias的shape信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @param indexSchema 存放算子index差异的结构体
 * @return ge::graphStatus
 */
static ge::graphStatus CheckBiasShape(const gert::TilingContext *context, const char *opName,
                                      const OpAttrIndexSchema &indexSchema)
{
    const gert::StorageShape *biasShape = context->GetOptionalInputShape(INPUT_BIAS_INDEX);
    if (biasShape != nullptr) {
        uint64_t biasShapeDimNum = biasShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK((biasShapeDimNum != 1), OP_LOGE(opName, "The input bias dimNum should be one."),
                        return ge::GRAPH_FAILED);
        uint64_t biasDim0 = biasShape->GetStorageShape().GetDim(0);
        const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
        const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2_INDEX);
        uint64_t x1Dim1 = x1Shape->GetStorageShape().GetDim(1);
        uint64_t x2Dim0 = x2Shape->GetStorageShape().GetDim(0);
        uint64_t x2Dim1 = x2Shape->GetStorageShape().GetDim(1);

        bool x2TransFlag = false;
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const bool *isTransX2 = attrs->GetAttrPointer<bool>(indexSchema.x2Transpose);
        if (isTransX2) {
            x2TransFlag = *isTransX2;
        }
        uint64_t nAxis = (x2TransFlag) ? x2Dim0 : x2Dim1;
        OP_TILING_CHECK((biasDim0 != nAxis),
                        OP_LOGE(opName, "The bias dimNum0 should be %lu, but actual value is %lu.", nAxis, biasDim0),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tiling输入Input的Dim范围信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @param indexSchema 存放算子index差异的结构体
 * @return ge::graphStatus
 */
static ge::graphStatus CheckShapeDimRange(const gert::TilingContext *context, const char *opName,
                                          const OpAttrIndexSchema &indexSchema)
{
    // attr非空在CheckAttrsInfo校验过
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2_INDEX);
    uint64_t x1Dim0 = x1Shape->GetStorageShape().GetDim(0);
    uint64_t x1Dim1 = x1Shape->GetStorageShape().GetDim(1);
    uint64_t x2Dim0 = x2Shape->GetStorageShape().GetDim(0);
    uint64_t x2Dim1 = x2Shape->GetStorageShape().GetDim(1);
    // 获取n轴
    bool x2TransFlag = false;
    const bool *isTransX2 = attrs->GetAttrPointer<bool>(indexSchema.x2Transpose);
    if (isTransX2) {
        x2TransFlag = *isTransX2;
    }
    uint64_t nAxis = (x2TransFlag) ? x2Dim0 : x2Dim1;
    uint64_t kAxis = (x2TransFlag) ? x2Dim1 : x2Dim0;
    // 校验M,当前M为0的话，走公式化tiling切分实际是不支持的,后面可去除
    OP_TILING_CHECK(x1Dim0 == 0, OP_LOGE(opName, "Invalid x1 shape: dim 0(m) cannot be 0."), return ge::GRAPH_FAILED);
    // 校验M不能大于int32的最大值
    OP_TILING_CHECK(x1Dim0 > MAX_INT32_VALUE, OP_LOGE(opName, "X1 dim 0(m) exceeds INT32_MAX, got %lu.", x1Dim0),
                    return ge::GRAPH_FAILED);
    // 校验K,K的范围应该在[1, 65535]
    OP_TILING_CHECK(kAxis > K_MAX_VALUE, OP_LOGE(opName, "The k dim exceeds max value 65535, got %lu.", kAxis),
                    return ge::GRAPH_FAILED);
    // 校验N, N不为空
    OP_TILING_CHECK(nAxis == 0, OP_LOGE(opName, "Invalid x2 shape: N cannot be 0."), return ge::GRAPH_FAILED);
    // 校验N不能大雨int32的
    OP_TILING_CHECK(nAxis > MAX_INT32_VALUE, OP_LOGE(opName, "N axis exceeds INT32_MAX, got %lu.", nAxis),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验量化tiling输入的shape信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @param indexSchema 存放输入参数索引差别的结构体
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::CheckMxQuantShapeInfo(const gert::TilingContext *context, const char *opName,
                                                                const OpAttrIndexSchema &indexSchema)
{
    ge::graphStatus status;
    // 校验输入量化Input Shape是否为空
    status = CheckMxQuantInputShapesValid(context, opName); 
    if (status != ge::GRAPH_SUCCESS)
        return status;

    // 校验维度数目是否合法
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2_INDEX);
    const gert::StorageShape *x1ScaleShape = context->GetOptionalInputShape(INPUT_X1_SCALE_INDEX);
    const gert::StorageShape *x2ScaleShape = context->GetOptionalInputShape(INPUT_X2_SCALE_INDEX);
    status = CheckShapeDimensions(x1Shape, "mx quant input x1", opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;
    status = CheckShapeDimensions(x2Shape, "mx quant input x2", opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;
    status = CheckMxScaleShapeDimensions(x1ScaleShape, "mx quant input x1scale", opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;
    status = CheckMxScaleShapeDimensions(x2ScaleShape, "mx quant input x2scale", opName);
    if (status != ge::GRAPH_SUCCESS)
        return status;
    // 校验输出
    const gert::StorageShape *yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_TILING_CHECK((yShape == nullptr), OP_LOGE(opName, "the yShape is nullptr."), return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(yShape, "mx quant output y", opName);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    // 校验bias的shape信息
    status = CheckBiasShape(context, opName, indexSchema);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    // 校验shape的dim范围
    status = CheckShapeDimRange(context, opName, indexSchema);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验量化tiling inputshape非空
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::CheckMxQuantInputShapesValid(const gert::TilingContext *context, const char *opName)
{
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2_INDEX);
    const gert::StorageShape *x1ScaleShape = context->GetOptionalInputShape(INPUT_X1_SCALE_INDEX);
    const gert::StorageShape *x2ScaleShape = context->GetOptionalInputShape(INPUT_X2_SCALE_INDEX);
    OP_TILING_CHECK((x1Shape == nullptr), OP_LOGE(opName, "the input x1 shape is invalid"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x2Shape == nullptr), OP_LOGE(opName, "the input x2 shape is invalid"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x1ScaleShape == nullptr), OP_LOGE(opName, "the input x1Scale shape is invalid"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((x2ScaleShape == nullptr), OP_LOGE(opName, "the input x2Scale shape is invalid"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验MX量化MatmulAlltoAll在不同转置情况下的x1,x2,output的shape关系,以及需要满足n/rankSize的整除关系
 * 需要满足 x1(BS,H1), x2(H2, H1) if trans else x2(H1, H2)
 * output(BS*rankSize, H2/rankSize)
 *
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::CheckMxQuantMatrixMulShapes(const gert::TilingContext *context, const char *opName)
{
    OP_TILING_CHECK(MatmulAllToAllTilingBase::Check2DMatrixMulShapes(context, opName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Mx quant tiling check x1 x2 and y shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckMxQuantScaleShapes(context, opName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Mx quant tiling check scale shape failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验MXFP8量化scale的维度
 *
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::CheckMxQuantScaleShapes(const gert::TilingContext *context, const char *opName)
{
    bool TransX2Flag = false;
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const bool *isX2TransX2 = attrs->GetAttrPointer<bool>(ATTR_X2_TRANSPOSE_INDEX);
    if (isX2TransX2) {
        TransX2Flag = *isX2TransX2;
    }
    Matrix2DShapes shapeInfo;
    MatmulAlltoAllTilingUtil::GetMatrix2DShapes(context, shapeInfo);
    uint64_t nAxis = TransX2Flag ? shapeInfo.x2Dim0 : shapeInfo.x2Dim1;
    const gert::StorageShape *x1ScaleShape = context->GetOptionalInputShape(INPUT_X1_SCALE_INDEX);
    const gert::StorageShape *x2ScaleShape = context->GetOptionalInputShape(INPUT_X2_SCALE_INDEX);
    uint64_t x1ScaleDim0 = x1ScaleShape->GetStorageShape().GetDim(0);
    uint64_t x1ScaleDim1 = x1ScaleShape->GetStorageShape().GetDim(1);
    uint64_t x1ScaleDim2 = x1ScaleShape->GetStorageShape().GetDim(2);
    uint64_t x2ScaleDim0 = x2ScaleShape->GetStorageShape().GetDim(0);
    uint64_t x2ScaleDim1 = x2ScaleShape->GetStorageShape().GetDim(1);
    uint64_t x2ScaleDim2 = x2ScaleShape->GetStorageShape().GetDim(2);
    uint64_t x1Dim1DivMxFp8Size = static_cast<uint64_t>(((static_cast<int64_t>(shapeInfo.x1Dim1) + 
                                                          MX_SCALE_OFFSET - 1) / MX_SCALE_OFFSET));
    uint64_t x2Dim0DivMxFp8Size = static_cast<uint64_t>(((static_cast<int64_t>(shapeInfo.x2Dim0) + 
                                                          MX_SCALE_OFFSET - 1) / MX_SCALE_OFFSET));
    uint64_t x2Dim1DivMxFp8Size = static_cast<uint64_t>(((static_cast<int64_t>(shapeInfo.x2Dim1) + 
                                                          MX_SCALE_OFFSET - 1) / MX_SCALE_OFFSET));
    OP_TILING_CHECK((x1ScaleDim0 != shapeInfo.x1Dim0) || (x1ScaleDim1 != x1Dim1DivMxFp8Size) || (x1ScaleDim2 != EVEN_ALIGN),
        OP_LOGE(opName, "In the Non-Transposed Scenario, Wrong shape of x1Scale! "
            "x1scaleDim0 should be equal to x1Dim0(%lu), "
            "x1scaleDim1 should be equal to (x1Dim1(%lu) + MX_SCALE_OFFSET(%lu) - 1) / MX_SCALE_OFFSET(%lu), x1scaleDim2 should be equal to 2, "
            "Expected Shape of x1Scale = (%lu, %lu, %lu), Actual Shape of x1Scale = (%lu, %lu, %lu).",
            shapeInfo.x1Dim0, shapeInfo.x1Dim1, MX_SCALE_OFFSET, MX_SCALE_OFFSET, shapeInfo.x1Dim0, x1Dim1DivMxFp8Size, EVEN_ALIGN, 
            x1ScaleDim0, x1ScaleDim1, x1ScaleDim2), return ge::GRAPH_FAILED);
    if (TransX2Flag){ // Transposed Scenario
        OP_TILING_CHECK((x2ScaleDim0 != shapeInfo.x2Dim0) || (x2ScaleDim1 != x2Dim1DivMxFp8Size) || (x2ScaleDim2 != EVEN_ALIGN),
            OP_LOGE(opName_, "In the Transposed Scenario, Wrong shape of x2Scale! "
                "x2scaleDim0 should be equal to x2Dim0(%lu), "
                "x2scaleDim1 should be equal to (x2Dim1(%lu) + MX_SCALE_OFFSET(%lu) - 1) / MX_SCALE_OFFSET(%lu), x2scaleDim2 should be equal to 2, "
                "Expected Shape of x2Scale = (%lu, %lu, %lu), Actual Shape of x2Scale = (%lu, %lu, %lu).",
                shapeInfo.x2Dim0, shapeInfo.x2Dim1, MX_SCALE_OFFSET, MX_SCALE_OFFSET, shapeInfo.x2Dim0, x2Dim1DivMxFp8Size, EVEN_ALIGN, 
                x2ScaleDim0, x2ScaleDim1, x2ScaleDim2), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((x2ScaleDim0 != x2Dim0DivMxFp8Size) || (x2ScaleDim1 != shapeInfo.x2Dim1) || (x2ScaleDim2 != EVEN_ALIGN),
            OP_LOGE(opName_, "In the Non-Transposed Scenario, Wrong shape of x2Scale! "
                "x2scaleDim0 should be equal to (x2Dim0(%lu) + MX_SCALE_OFFSET(%lu) - 1) / MX_SCALE_OFFSET(%lu), "
                "x2scaleDim1 should be equal to x2Dim1(%lu), x2scaleDim2 should be equal to 2, "
                "Expected Shape of x2Scale = (%lu, %lu, %lu), Actual Shape of x2Scale = (%lu, %lu, %lu).",
                shapeInfo.x2Dim0, MX_SCALE_OFFSET, MX_SCALE_OFFSET, shapeInfo.x2Dim1, x2Dim0DivMxFp8Size, shapeInfo.x2Dim1, EVEN_ALIGN, 
                x2ScaleDim0, x2ScaleDim1, x2ScaleDim2), return ge::GRAPH_FAILED);
    }     
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置算子的数据类型信息
 *
 * @param context 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName 算子名称
 * @param contextInfo 存储了tiling的过程信息
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::SetMxDataTypeInfo(const gert::TilingContext *context, const char *opName,
                                                            TilingContextInfo &contextInfo)
{
    const gert::StorageShape *matrixBias = context->GetOptionalInputShape(INPUT_BIAS_INDEX);
    ge::DataType aType = context->GetInputDesc(INPUT_X1_INDEX)->GetDataType();
    ge::DataType bType = context->GetInputDesc(INPUT_X2_INDEX)->GetDataType();
    ge::DataType cType = context->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    ge::DataType biasType;
    bool isBias = true;
    if (matrixBias == nullptr) {
        isBias = false;
        biasType = cType;
    } else {
        biasType = context->GetOptionalInputDesc(INPUT_BIAS_INDEX)->GetDataType();
    }

    contextInfo.args_.outputDtypeSize = mc2tiling::GetDataTypeSize(opName, cType);
    contextInfo.args_.inputDtypeSize = mc2tiling::GetDataTypeSize(opName, aType);
    contextInfo.args_.isBias = isBias;
    contextInfo.args_.geCType = cType;
    contextInfo.args_.geBiasType = biasType;
    contextInfo.args_.geAType = aType;
    contextInfo.args_.geBType = bType;
    contextInfo.args_.cType = mc2tiling::ConvertGeTypeToMmType(opName, cType);
    contextInfo.args_.aType = mc2tiling::ConvertGeTypeToMmType(opName, aType);
    contextInfo.args_.bType = mc2tiling::ConvertGeTypeToMmType(opName, bType);
    contextInfo.args_.biasType = mc2tiling::ConvertGeTypeToMmType(opName, biasType);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置hccl的config,进行hccl对应的通信任务设置
 *
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::SetHcclTiling()
{
    OP_TILING_CHECK(mc2tiling::ConvertGeTypeToHcclType(opName_, contextInfo.args_.geCType) ==
                    mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED,
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "Cannot find HcclDataType according to ge datatype = %d.",
                                                   static_cast<int32_t>(contextInfo.args_.geCType)), return ge::GRAPH_FAILED;);
    Mc2CcTilingConfigBuilder matmulAllToAllBuilder =
        Mc2CcTilingConfigBuilder::create(contextInfo.group, mc2tiling::AicpuComType::HCCL_CMD_ALLTOALL,
                                         Mc2CcTilingConfigBuilder::AlgConfigType::ALL_TO_ALL);
    AscendC::Mc2CcTilingConfig matmulAllToAllTilingConfig = 
        matmulAllToAllBuilder
            .withReduceType(opName_, AscendC::HcclReduceOp::HCCL_REDUCE_SUM, contextInfo.args_.geCType, contextInfo.args_.geCType)
            .withCommEngine(mc2tiling::A5_CCU_ENGINE)
            .build();
    if (!matmulAllToAllBuilder.isSuccess()) {
        OP_LOGE(opName_, "Mx quant matmul allto all build hccl tiling config failed: %s", matmulAllToAllBuilder.errorMsg().c_str());
        return ge::GRAPH_FAILED;
    }
    matmulAllToAllTilingConfig.GetTiling(localTilingData_.mc2InitTiling);
    matmulAllToAllTilingConfig.GetTiling(localTilingData_.mc2CcTiling);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 进行通算切分之后单个块的MM Tiling
 *
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::DoMxQuantMMTiling()
{
    // 设置MM切前信息
    mmMvalueLen = inferredInfo.tileM;
    MxQuantMatmulAlltoAllHelper mmTile(*this, localTilingData_.mc2QuantBmmV3TileTilingData, mmMvalueLen);
    GE_ASSERT_GRAPH_SUCCESS(mmTile.DoTiling());
    if (inferredInfo.tailCnt == 0) {
        return ge::GRAPH_SUCCESS;
    }
    mmMvalueLen = inferredInfo.tailM;
    MxQuantMatmulAlltoAllHelper mmTail(*this, localTilingData_.mc2QuantBmmV3TailTilingData, mmMvalueLen);
    GE_ASSERT_GRAPH_SUCCESS(mmTail.DoTiling());
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 重写获取MM index的信息
 * 由于本算子的context和MM不一样，需要重写获取MM index的一些信息，把我们的context传给Matmul，来达到可以调用MM策略的目的。
 * @return ge::graphStatus
 */
const gert::Shape MxQuantMatmulAlltoAllHelper::GetX1Shape(const size_t index)
{
    (void)index;
    return gert::Shape(
        {static_cast<int64_t>(mmLen_), static_cast<int64_t>(tilingProcesser_.contextInfo.args_.kValue)});
}
const gert::Shape MxQuantMatmulAlltoAllHelper::GetX2Shape(const size_t index)
{
    (void)index;
    if (tilingProcesser_.contextInfo.args_.isBTrans) {
        return gert::Shape(
            {static_cast<int64_t>(tilingProcesser_.contextInfo.args_.nValue), static_cast<int64_t>(tilingProcesser_.contextInfo.args_.kValue)});
    }
    return gert::Shape(
        {static_cast<int64_t>(tilingProcesser_.contextInfo.args_.kValue), static_cast<int64_t>(tilingProcesser_.contextInfo.args_.nValue)});
}

const gert::Shape& MxQuantMatmulAlltoAllHelper::GetScaleShape(const size_t index)
{
    (void)index;
    return context_->GetOptionalInputShape(static_cast<size_t>(INPUT_X2_SCALE_INDEX))->GetStorageShape();
}

const gert::StorageShape* MxQuantMatmulAlltoAllHelper::GetOffsetShape(const size_t index)
{
    (void) index; 
    return (gert::StorageShape*)nullptr;
}

const gert::StorageShape* MxQuantMatmulAlltoAllHelper::GetPertokenShape(const size_t index)
{
    (void)index;
    mxQuantStorageShape = gert::StorageShape({static_cast<int64_t>(mmLen_)},{static_cast<int64_t>(mmLen_)});
    return &mxQuantStorageShape;
}

const gert::StorageShape* MxQuantMatmulAlltoAllHelper::GetBiasShape(const size_t index)
{
    (void)index;
    return context_->GetOptionalInputShape(static_cast<size_t>(INPUT_BIAS_INDEX));
}

ge::graphStatus MxQuantMatmulAlltoAllHelper::GetShapeAttrsInfo()
{   
    OP_LOGD(tilingProcesser_.opName_, "Start assemble input params for matmul tiling");
    auto&& tilingArgs = tilingProcesser_.contextInfo.args_;
    inputParams_.opName = tilingProcesser_.opName_;
    inputParams_.transB = tilingArgs.isBTrans;
    inputParams_.hasBias = tilingArgs.isBias;
    inputParams_.transA = false;
    inputParams_.aDtype = tilingArgs.geAType;
    inputParams_.bDtype = tilingArgs.geBType;
    inputParams_.libApiWorkSpaceSize = tilingProcesser_.libApiWorkSpaceSize_;
    int yDType = *context_->GetAttrs()->GetAttrPointer<uint64_t>(ATTR_Y_DTYPE_INDEX);
    auto scaleTensorDesc = context_->GetOptionalInputDesc(INPUT_X2_SCALE_INDEX);
    OP_TILING_CHECK((scaleTensorDesc == nullptr),
                    VECTOR_INNER_ERR_REPORT_TILING(tilingProcesser_.opName_, "the scale tensor is invalid"),
                    return ge::GRAPH_FAILED);
    inputParams_.scaleDtype = scaleTensorDesc->GetDataType();                
    auto perTokenScaleTensorDesc = context_->GetOptionalInputDesc(INPUT_X1_SCALE_INDEX);
    OP_TILING_CHECK((perTokenScaleTensorDesc == nullptr),
                    VECTOR_INNER_ERR_REPORT_TILING(tilingProcesser_.opName_, "the perToken scale tensor is invalid"),
                    return ge::GRAPH_FAILED);
    inputParams_.perTokenScaleDtype = perTokenScaleTensorDesc->GetDataType();
    inputParams_.outDtype = static_cast<int64_t>(yDType);
    inputParams_.cDtype = static_cast<ge::DataType>(yDType);
    OP_LOGD(tilingProcesser_.opName_, "yDType is %ld", inputParams_.outDtype);
    inputParams_.biasDtype = tilingArgs.isBias ? tilingArgs.geBiasType : ge::DT_INT32;
    if((scaleTensorDesc->GetDataType() == ge::DataType::DT_FLOAT8_E8M0) && 
        (perTokenScaleTensorDesc->GetDataType() == ge::DataType::DT_FLOAT8_E8M0)) {
        inputParams_.groupSizeK = MX_SCALE_OFFSET;
    }
    GE_ASSERT_TRUE(AnalyzeInputs());
    PrintTilingInputParam(inputParams_);
    return ge::GRAPH_SUCCESS;
}

void MxQuantMatmulAlltoAllHelper::PrintTilingInputParam(Mc2QuantBatchMatmulInfo& quantMatmulInfo)
{
    OP_LOGD(tilingProcesser_.opName_,
            "aDtype_ %d bDtype_ %d cDtype_ %d biasDtype_ %d outDtype %ld scaleDtype %d perTokenScaleDtype %d",
            static_cast<int32_t>(quantMatmulInfo.aDtype), static_cast<int32_t>(quantMatmulInfo.bDtype),
            static_cast<int32_t>(quantMatmulInfo.cDtype), static_cast<int32_t>(quantMatmulInfo.biasDtype),
            quantMatmulInfo.outDtype, static_cast<int32_t>(quantMatmulInfo.scaleDtype),
            static_cast<int32_t>(quantMatmulInfo.perTokenScaleDtype));
    OP_LOGD(tilingProcesser_.opName_, "mSize_ %ld kSize_ %ld nSize_ %ld libApiWorkSpaceSize %u",
            quantMatmulInfo.mSize, quantMatmulInfo.kSize, quantMatmulInfo.nSize, quantMatmulInfo.libApiWorkSpaceSize);
    OP_LOGD(tilingProcesser_.opName_, "Check isPerTensor=%d, isDoubleScale=%d.", static_cast<int32_t>(quantMatmulInfo.isPerTensor), static_cast<int32_t>(quantMatmulInfo.isDoubleScale));
    OP_LOGD(tilingProcesser_.opName_, "Check groupSizeM=%d, groupSizeK=%d, groupSizeN=%d", 
            static_cast<int32_t>(quantMatmulInfo.groupSizeM), static_cast<int32_t>(quantMatmulInfo.groupSizeK), static_cast<int32_t>(quantMatmulInfo.groupSizeN));
}

ge::graphStatus MxQuantMatmulAlltoAllHelper::DoLibApiTiling()
{
    GE_ASSERT_GRAPH_SUCCESS(Mc2AdaptiveSlidingWindowTiling::DoLibApiTiling());
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 重写友元类PostTiling方法
 * PostTiling主要做的是拷贝tilingdata的活，但是本算子拷贝tilingdata是在大结构体中拷贝，不需要在此处拷贝。
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAlltoAllHelper::PostTiling()
{
    tilingProcesser_.workspaceSize_ = std::max(tilingProcesser_.workspaceSize_, workspaceSize_);
    OP_LOGD(tilingProcesser_.opName_, "set mm workspace size %lu to mc2", tilingProcesser_.workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 构造函数，创建一个MxQuantMatmulAlltoAllHelper对象
 *
 * @param context
 */
MxQuantMatmulAlltoAllHelper::MxQuantMatmulAlltoAllHelper(MxQuantMatmulAllToAllTilingBase& mxQuantMatmulAllToAllTilingBase, 
                                                     DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& data, uint64_t& mmMvalueLen)
    : Mc2AdaptiveSlidingWindowTiling(mxQuantMatmulAllToAllTilingBase.context_, &data), tilingProcesser_(mxQuantMatmulAllToAllTilingBase),
    mmLen_(mmMvalueLen)
{
}

/**
 * @brief 打印量化matmul tiling的信息
 *
 * @param opName
 * @param tiling
 */
void MxQuantMatmulAllToAllTilingBase::PrintMxQuantMMV3TilingData(const std::string &opName, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &tiling)
{
    PrintTCubeTilingData(opName, tiling.matmulTiling);
 	PrintExtendMatmulTiling(opName, tiling);
}

/**
 * @brief 打印执行过程中的matmul tiling信息
 *
 * @param opName
 * @param tiling
 */
void MxQuantMatmulAllToAllTilingBase::PrintExtendMatmulTiling(const std::string &opName, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &tiling)
 	 {
 	     OP_LOGD(opName, "QuantBmmV3Params.batchA=%u.", tiling.params.batchA);
         OP_LOGD(opName, "QuantBmmV3Params.batchA1=%u.", tiling.params.batchA1);
 	     OP_LOGD(opName, "QuantBmmV3Params.batchA2=%u.", tiling.params.batchA2);
         OP_LOGD(opName, "QuantBmmV3Params.batchA3=%u.", tiling.params.batchA3);
 	     OP_LOGD(opName, "QuantBmmV3Params.batchA4=%u.", tiling.params.batchA4);
 	     OP_LOGD(opName, "QuantBmmV3Params.batchB=%u.", tiling.params.batchB);
         OP_LOGD(opName, "QuantBmmV3Params.batchB1=%u.", tiling.params.batchB1);
 	     OP_LOGD(opName, "QuantBmmV3Params.batchB2=%u.", tiling.params.batchB2);
 	     OP_LOGD(opName, "QuantBmmV3Params.batchB3=%u.", tiling.params.batchB3);
 	     OP_LOGD(opName, "QuantBmmV3Params.batchB4=%u.", tiling.params.batchB4);
 	     OP_LOGD(opName, "QuantBmmV3Params.batchC=%u.", tiling.params.batchC);
 	     OP_LOGD(opName, "QuantBmmV3Params.batchC1=%u.", tiling.params.batchC1);
 	     OP_LOGD(opName, "QuantBmmV3Params.batchC2=%u.", tiling.params.batchC2);
 	     OP_LOGD(opName, "QuantBmmV3Params.batchC3=%u.", tiling.params.batchC3);
 	     OP_LOGD(opName, "QuantBmmV3Params.batchC4=%u.", tiling.params.batchC4);
 	     OP_LOGD(opName, "QuantBmmV3Params.singleCoreBatch=%u.", tiling.params.singleCoreBatch);
 	     OP_LOGD(opName, "QuantBmmV3Params.isPerTensor=%u.", tiling.params.isPerTensor);
 	     OP_LOGD(opName, "QuantBmmV3Params.isPertoken=%u.", tiling.params.isPertoken);
 	     OP_LOGD(opName, "QuantBmmV3Params.isDoubleScale=%u.", tiling.params.isDoubleScale);
 	     OP_LOGD(opName, "QuantBmmV3Params.biasThreeDim=%u.", tiling.params.biasThreeDim);
         OP_LOGD(opName, "QuantBmmV3Params.needUbBuffer=%u.", tiling.params.needUbBuffer);
 	     OP_LOGD(opName, "QuantBmmV3Params.ubCalcM=%u.", tiling.params.ubCalcM);
 	     OP_LOGD(opName, "QuantBmmV3Params.ubCalcN=%u.", tiling.params.ubCalcN);
 	     OP_LOGD(opName, "QuantBmmV3Params.realSingleCoreM=%u.", tiling.params.realSingleCoreM);
 	     OP_LOGD(opName, "QuantBmmV3Params.realSingleCoreN=%u.", tiling.params.realSingleCoreN);
 	     OP_LOGD(opName, "QuantBmmV3Params.biasDtype=%u.", tiling.params.biasDtype);
 	     OP_LOGD(opName, "QuantBmmV3Params.ubSize=%u.", tiling.params.ubSize);
 	     OP_LOGD(opName, "QuantBmmV3Params.isMClash=%u.", tiling.params.isMClash);
 	     OP_LOGD(opName, "QuantBmmV3Params.isNClash=%u.", tiling.params.isNClash);
 	     OP_LOGD(opName, "QuantBmmV3Params.groupSizeM=%u.", tiling.params.groupSizeM);
 	     OP_LOGD(opName, "QuantBmmV3Params.groupSizeK=%u.", tiling.params.groupSizeK);
 	     OP_LOGD(opName, "QuantBmmV3Params.groupSizeN=%u.", tiling.params.groupSizeN);
         OP_LOGD(opName, "AdaptiveSlidingWin.mTailTile=%u.", tiling.adaptiveSlidingWin.mTailTile);
 	     OP_LOGD(opName, "AdaptiveSlidingWin.nTailTile=%u.", tiling.adaptiveSlidingWin.nTailTile);
 	     OP_LOGD(opName, "TileL2cacheTiling.mTileCntL2=%u.", tiling.tileL2cacheTiling.mTileCntL2);
 	     OP_LOGD(opName, "TileL2cacheTiling.nTileCntL2=%u.", tiling.tileL2cacheTiling.nTileCntL2);
 	     OP_LOGD(opName, "TileL2cacheTiling.mTileBlock=%u.", tiling.tileL2cacheTiling.mTileBlock);
 	     OP_LOGD(opName, "TileL2cacheTiling.nTileBlock=%u.", tiling.tileL2cacheTiling.nTileBlock);
 	     OP_LOGD(opName, "TileL2cacheTiling.calOrder=%u.", tiling.tileL2cacheTiling.calOrder);
 	     OP_LOGD(opName, "TileL2cacheTiling.isBasicTiling=%u.", tiling.tileL2cacheTiling.isBasicTiling);
    }
     
/**
 * @brief 打印tilingInfo信息
 *
 * @param opName
 * @param tilingInfo
 */
void MxQuantMatmulAllToAllTilingBase::PrintMxQuantMatmulAlltoAllTilingInfo(const std::string &opName,
                                                               MatmulAlltoAllTilingInfo &tilingInfo)
{
    OP_LOGD(opName, "tilingInfo.rankDim: %u", tilingInfo.rankDim);
    OP_LOGD(opName, "tilingInfo.tileCnt: %u", tilingInfo.tileCnt);
    OP_LOGD(opName, "tilingInfo.tileM: %u", tilingInfo.tileM);
    OP_LOGD(opName, "tilingInfo.tailCnt: %u", tilingInfo.tailCnt);
    OP_LOGD(opName, "tilingInfo.tailM: %u", tilingInfo.tailM);
    OP_LOGD(opName, "tilingInfo.rankM: %u", tilingInfo.rankM);
    OP_LOGD(opName, "tilingInfo.rankK: %u", tilingInfo.rankK);
    OP_LOGD(opName, "tilingInfo.rankN: %u", tilingInfo.rankN);
    OP_LOGD(opName, "tilingInfo.biasLen: %u", tilingInfo.biasLen);
    OP_LOGD(opName, "tilingInfo.permuteLen: %u", tilingInfo.permuteLen);
    OP_LOGD(opName, "tilingInfo.mmResultLen: %u", tilingInfo.mmResultLen);
    OP_LOGD(opName, "tilingInfo.aicCoreNum: %u", tilingInfo.aicCoreNum);
    OP_LOGD(opName, "tilingInfo.hcclDataType: %u", tilingInfo.hcclDataType);
}

/**
 * @brief 打印传递给kernel的tilingData
 *
 * @param outTilingData tilingData参数
 */
void MxQuantMatmulAllToAllTilingBase::PrintMxQuantMatmulAlltoAllTilingData(QuantMatmulAlltoAllTilingData &outTilingData)
{
    PrintMxQuantMatmulAlltoAllTilingInfo(opName_, outTilingData.quantMatmulAlltoAllTilingInfo);
    PrintMxQuantMMV3TilingData(opName_, outTilingData.mc2QuantBmmV3TileTilingData);
    if (outTilingData.quantMatmulAlltoAllTilingInfo.tailCnt == 0) {
        return;
    }
    OP_LOGD(opName_, "MxQuantMatmulAlltoall has tail");
    PrintMxQuantMMV3TilingData(opName_, outTilingData.mc2QuantBmmV3TailTilingData);
}

/**
 * @brief 保存量化tiling数据到context
 *
 * @return ge::graphStatus
 */
ge::graphStatus MxQuantMatmulAllToAllTilingBase::PostTiling()
{
    context_->SetScheduleMode(1);
    SetTilingInfo(localTilingData_.quantMatmulAlltoAllTilingInfo);
    QuantMatmulAlltoAllTilingData *outTilingData = context_->GetTilingData<QuantMatmulAlltoAllTilingData>();
    size_t tilingBufCap = context_->GetRawTilingData()->GetCapacity();
    OP_TILING_CHECK((outTilingData == nullptr), OP_LOGE(opName_, "Failed to get tiling data from context"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((tilingBufCap < sizeof(localTilingData_)),
                    OP_LOGE(opName_, "TilingBuffer capacity too small, capacity = %zu, need = %zu.", 
                        tilingBufCap, sizeof(localTilingData_)), return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(outTilingData, tilingBufCap, &localTilingData_, sizeof(localTilingData_));
    if (ret != EOK) {
        OP_LOGE(opName_, "MatmulAlltoAll postTiling: memcpy_s tiling data failed, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(opName_, "Final tiling data size=%zu and context capacity size=%zu.", sizeof(QuantMatmulAlltoAllTilingData),
            context_->GetRawTilingData()->GetCapacity());
    context_->SetBlockDim(contextInfo.args_.aicCoreNum);        
    context_->GetRawTilingData()->SetDataSize(sizeof(QuantMatmulAlltoAllTilingData));
    PrintMxQuantMatmulAlltoAllTilingData(*outTilingData);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tilingInfo结构体
 *
 * @param tilingInfo 目标结构体
 */
void MxQuantMatmulAllToAllTilingBase::SetTilingInfo(MatmulAlltoAllTilingInfo &tilingInfo) const
{
    // 基本字段拷贝
    tilingInfo.tileCnt = inferredInfo.tileCnt;
    tilingInfo.tileM = inferredInfo.tileM;
    tilingInfo.tailCnt = inferredInfo.tailCnt;
    tilingInfo.tailM = inferredInfo.tailM;
    tilingInfo.rankM = contextInfo.args_.mValue;
    tilingInfo.rankN = contextInfo.args_.nValue;
    tilingInfo.rankK = contextInfo.args_.kValue;
    tilingInfo.biasLen = inferredInfo.biasLen;
    tilingInfo.mmResultLen = inferredInfo.mmResultLen;
    tilingInfo.permuteLen = inferredInfo.permuteLen;
    tilingInfo.rankDim = contextInfo.args_.rankDim;
    tilingInfo.aicCoreNum = contextInfo.args_.aicCoreNum;
    tilingInfo.hcclDataType =
        (static_cast<uint64_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, contextInfo.args_.geCType))); // hccl数据类型
}

/**
 * @brief 获取对应的tilingKey
 * 使用QUANT_MODE来区分tilingKey,此处的QUANT_MODE指的是x1,x2的QUANT模式组合，以x1为pertoken量化(K)，x2为perchannel量化(C)
 * 为例子，K-C量化就代表一种组合
 *
 * @return uint64_t tilingKey结果
 */
uint64_t MxQuantMatmulAllToAllTilingBase::GetTilingKey() const
{
    // 按照量化组合模式，是否转置，bias数据类型进行展开
    bool x2TransposeFlag = contextInfo.args_.isBTrans ? true : false;
    uint32_t biasDType = DTYPE_BIAS_FP32;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(MX_QUANT_MODE, x2TransposeFlag, biasDType);
    OP_LOGD(opName_, "MXQUANTMODE,X2TRANSPOSE,DTYPEBIAS: [%d,%d,%d], TilingKey is [%lu].", MX_QUANT_MODE,
            x2TransposeFlag, biasDType, tilingKey);
    return tilingKey;
}

/**
 * @brief 构造函数，创建一个MxQuantMatmulAllToAllTilingBase对象
 *
 * @param context
 */
MxQuantMatmulAllToAllTilingBase::MxQuantMatmulAllToAllTilingBase(gert::TilingContext *context) : MatmulAllToAllTilingBase(context)
{
}

// 注册tiling类
REGISTER_TILING_TEMPLATE_WITH_ARCH(MatmulAlltoAll, MxQuantMatmulAllToAllTilingBase,
                                         static_cast<int32_t>(NpuArch::DAV_3510), 2);

} // namespace optiling