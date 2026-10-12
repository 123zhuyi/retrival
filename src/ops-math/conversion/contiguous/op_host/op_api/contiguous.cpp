/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/shape_utils.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "opdev/framework_op.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "conversion/as_strided/op_api/as_strided.h"
#include "conversion/broadcast_to/op_api/broadcast_to.h"
#include "conversion/strided_slice/op_api/strided_slice.h"
#include "conversion/tensor_move/op_api/tensor_move.h"
#include "conversion/view_copy/op_api/view_copy.h"

using namespace op;

namespace l0op {

op::DataType TYPE_INT64 = op::ToOpDataType(ACL_INT64);
typedef FVector<std::pair<int64_t, std::pair<int64_t, int64_t>>, op::MAX_DIM_NUM> StrideIndexPairs;
static const uint64_t AS_STRIDED_MOVE_ALIGN_NUM = 64;

/**
 * Following are white list cases for AsStrided operations.
 */
static FVector<FVector<int64_t>> AS_STRIDED_SHAPE_LIST({
    {4, 288, 2, 2, 240, 240},
    {4, 512, 2, 2, 120, 120},
    {4, 10, 32, 128},
});

static FVector<FVector<int64_t>> AS_STRIDED_STRIDED_LIST({
    {67046400, 230400, 480, 1, 960, 2},
    {33177600, 57600, 240, 1, 480, 2},
    {122880, 128, 3840, 1},
});

const aclTensor* UnSafeReshape(const aclTensor* x, const op::Shape& shape, aclOpExecutor* executor)
{
    if (x->GetViewShape() == shape && x->GetStorageShape() == shape && x->GetOriginalShape() == shape) {
        return x;
    }
    return executor->CreateView(x, shape, x->GetViewOffset());
}

inline bool IsContiguous(StrideIndexPairs& strideIndexPairs)
{
    int64_t stride = 1;
    for (auto it = strideIndexPairs.rbegin(); it != strideIndexPairs.rend(); it++) {
        if (it->first != stride) {
            return false;
        }
        stride *= it->second.second;
    }
    return true;
}

inline bool SupportMoveAlign()
{
    auto socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    return !(
        socVersion == op::SocVersion::ASCEND910 || socVersion == op::SocVersion::ASCEND310 ||
        socVersion == op::SocVersion::ASCEND310P);
}

inline StrideIndexPairs BuildValidStrideIndexPairs(
    const op::Shape& viewShape, const op::Strides& strides, ContiguousParam& param)
{
    StrideIndexPairs strideIndexPairs;
    strideIndexPairs.reserve(strides.size());
    int64_t lastStride = INT64_MAX;
    int64_t permIdx = 0;
    for (size_t i = 0; i < strides.size(); i++) {
        if (strides[i] == 0) {
            param.mayBroadcast = true;
            param.broadcastSrcShape[i] = 1;
            continue;
        }
        if (viewShape[i] == 1) {
            continue;
        }
        if (lastStride < strides[i]) {
            param.mayTranspose = true;
        }
        lastStride = strides[i];
        strideIndexPairs.emplace_back(std::make_pair(strides[i], std::make_pair(permIdx, viewShape[i])));
        permIdx++;
    }
    return strideIndexPairs;
}

inline void SetTransposeShapeWithSrcShape(const op::Shape& srcShape, ContiguousParam& param)
{
    param.transposeSrcShape = srcShape;
    param.transposeDstShape = srcShape;
    for (size_t i = 0; i < param.transposeDstShape.GetDimNum(); i++) {
        param.transposeDstShape[i] = param.transposeSrcShape[param.perm[i]];
    }
}

inline bool OptimizeSlice(
    const op::Shape& simpleShape, const op::Strides& simpleStrides, ContiguousParam& param, int64_t& gcdValue)
{
    auto lastStride = simpleStrides[0];
    param.maySlice = simpleStrides[simpleStrides.size() - 1] == 1;
    for (size_t i = 1; i < simpleStrides.size(); i++) {
        if (lastStride % simpleStrides[i] != 0) {
            param.maySlice = false;
        }
        // All scenarios require that data should not overlap.
        if (lastStride < simpleStrides[i] * (simpleShape[i] - 1)) {
            return false;
        }
        if (i < simpleStrides.size() - 1) {
            gcdValue = std::gcd(gcdValue, simpleStrides[i]);
        }
        lastStride = simpleStrides[i];
    }

    if (param.maySlice) {
        return true;
    }

    param.mayStridedslice =
        gcdValue > 1 && gcdValue >= simpleStrides[simpleStrides.size() - 1] * simpleShape[simpleStrides.size() - 1];
    return param.mayStridedslice;
}

inline bool ValidateSliceParam(
    const ShapeVector& shape, op::FVector<int64_t, op::MAX_DIM_NUM>& offset,
    const op::FVector<int64_t, op::MAX_DIM_NUM>& size)
{
    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] < offset[i] + size[i]) {
            return false;
        }
    }
    return true;
}

inline bool BuildSliceParams(
    const op::Shape& simpleShape, const op::Strides& simpleStrides, int64_t offset, int64_t storageSize,
    ContiguousParam& param)
{
    param.sliceDstShape = simpleShape;
    // Infer origin shape
    auto dimNum = static_cast<int64_t>(simpleStrides.size());
    op::ShapeVector srcShape(dimNum);
    int64_t shapeSize = 1;
    for (int64_t i = dimNum - 2; i >= 0; i--) {
        // The value in the array simpleStrides cannot be zero.
        srcShape[i + 1] = simpleStrides[i] / simpleStrides[i + 1];
        shapeSize *= srcShape[i + 1];
    }
    srcShape[0] = storageSize / shapeSize;
    op::ToShape(srcShape, param.sliceSrcShape);
    // Infer params of offset and size.
    param.offset.resize(simpleStrides.size());
    param.size.resize(simpleStrides.size());
    for (size_t i = 0; i < simpleStrides.size(); i++) {
        param.offset[i] = offset / simpleStrides[i];
        param.size[i] = simpleShape[i];
        offset = offset % simpleStrides[i];
    }
    param.viewOffset = 0;
    return ValidateSliceParam(srcShape, param.offset, param.size);
}

static inline bool ValidateStridedSliceParam(
    const ShapeVector& shape, const op::Shape& targetShape, const ContiguousParam& param)
{
    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] < param.begin[i] || (targetShape[i] - 1) * param.strides[i] + param.begin[i] >= shape[i] ||
            shape[i] < targetShape[i]) {
            return false;
        }
    }
    return true;
}

inline bool BuildStridedSliceParams(
    const op::Shape& simpleShape, const op::Strides& simpleStrides, int64_t storageSize, int64_t gcdValue,
    ContiguousParam& param)
{
    int64_t offset = param.viewOffset;
    param.stridedsliceDstShape = simpleShape;
    // Infer origin shape and stride.
    auto dimNum = static_cast<int64_t>(simpleStrides.size());
    op::ShapeVector srcShape(dimNum);
    op::Strides srcStride(dimNum);
    srcShape[dimNum - 1] = gcdValue;
    srcStride[dimNum - 1] = 1;
    int64_t shapeSize = gcdValue;
    for (int64_t i = dimNum - 2; i >= 1; i--) {
        srcStride[i] = shapeSize;
        srcShape[i] = simpleStrides[i - 1] / srcStride[i];
        shapeSize *= srcShape[i];
    }
    srcShape[0] = storageSize / shapeSize;
    srcStride[0] = shapeSize;
    // Calculate params for strided slice.
    param.begin.resize(simpleStrides.size());
    param.end.resize(simpleStrides.size());
    param.strides.resize(simpleStrides.size());
    for (size_t i = 0; i < simpleStrides.size(); i++) {
        if (simpleStrides[i] % srcStride[i] != 0) {
            // target stride must be integer multiple of source stride
            return false;
        }
        param.begin[i] = offset / srcStride[i];
        param.strides[i] = simpleStrides[i] / srcStride[i];
        param.end[i] = param.begin[i] + simpleShape[i] * param.strides[i];
        offset = offset % srcStride[i];
    }
    op::ToShape(srcShape, param.stridedsliceSrcShape);
    param.viewOffset = 0;
    return ValidateStridedSliceParam(srcShape, simpleShape, param);
}

/**
 * White list for AsStrided operations.
 * @param strideIndexPairs
 * @return true or false
 */
inline bool CanOptimizeAsStridedContiguous(const StrideIndexPairs& strideIndexPairs)
{
    FVector<int64_t> shape(strideIndexPairs.size());
    FVector<int64_t> strides(strideIndexPairs.size());
    for (size_t i = 0; i < strideIndexPairs.size(); i++) {
        shape[i] = strideIndexPairs[i].second.second;
        strides[i] = strideIndexPairs[i].first;
    }
    for (size_t i = 0; i < AS_STRIDED_SHAPE_LIST.size(); i++) {
        if (shape == AS_STRIDED_SHAPE_LIST[i] && strides == AS_STRIDED_STRIDED_LIST[i]) {
            OP_LOGD("Better performance with AsStrided operation.");
            return true;
        }
    }
    return false;
}

inline bool CanReplaceSliceTransposeWithAsStrided(const op::Shape& viewShape, const op::Strides& strides)
{
    return (strides[strides.size() - 1] == 1) &&
           (viewShape[viewShape.GetDimNum() - 1] % AS_STRIDED_MOVE_ALIGN_NUM == 0);
}

/**
 * @brief 判断是否可以优化Contiguous转换流程
 *  识别输入Tensor是否可以通过以下一个或者多个算子组合完成非连续转连续，同时计算相应算子的参数
 *  Slice(StridedSlice) -> Transpose -> BroadcastTo
 * 识别原理：
 *  1. stride中存在0， 一定存在broadcast
 *  2. 去除stride为0的dim，剩余dim和stride满足连续，通过Broadcast后即是连续Tensor
 *  3. 连续Tensor的stride一定是降序排列的，如果存在非降序排列，则有可能可以通过Transpose转连续
 *  4.
 * 连续Tensor的stride序列，前一个值一定可以被后一个值整除，当降序排序后的stride序列满足该条件时，则一定存在一组slice参数转连续
 *  5.
 * 整除条件如果不满足，则求取前n-1个Stride的公约数，如果公约数大于1且大于最后一个stride值，则存在StrideSlice参数转连续
 * 处理过程：
 *  1. 过滤shape为1 shape和stride, 因为shape为1 的不管stride多少都没有意义
 *  2. 筛选stride为0的，但要保留对应shape，用于最后的broadcast（从数据量考虑，broadcast留到最后做）
 *      2.1 计算broadcast参数
 *  3. 降序排序，查看排序后的stride和shape特征
 *      3.1 满足连续--->计算Transpose参数
 *      3.2 满足slice特征-->计算Slice参数
 *      3.3 满足StridedSlice特征-->计算StridedSlice参数
 *  4. 调用Kernel
 *      4.1 如果存在slice,调用SliceKernel
 *      4.2 如果存在StrideSlice，调用StridedSliceKernel
 *      4.3 如果存在Transpose，调用TransposeKernel
 *      4.4 如果存在BroadcastTo，调用BroadcastToKernel
 *  5. 其他场景，使用AsStrided
 *                  [Shape]         [Stride]        [Offset]
 * Transpose Example:
 *  StorageShape:   (4, 5, 6, 7)    (210, 42, 7, 1)    0
 *  Perm:           (2, 3, 1, 0)
 *  ViewShape:      (6, 7, 5, 4)    (7, 1, 42, 210)    0
 * Slice Example:
 *  StorageShape:   (5, 6, 7)       (42, 7, 1)         0
 *  Param:          [2:4:1, 3:5:1, 4:7:1]
 *  ViewShape:      (2, 2, 3)       (42, 7, 1)        109
 * StridedSilce Example:
 *  StorageShape:   (5, 6, 7)       (42, 7, 1)         0
 *  Param           [1:4:1, 0:6:4, 4:7:2]
 *  ViewShape:      (3, 2, 2)       (42, 28, 2)        46
 * @param viewShape
 * @param strides
 * @param offset
 * @param storageSize
 * @param param
 * @return
 */
bool CanOptimizeContiguous(
    const op::Shape& viewShape, const op::Strides& strides, int64_t offset, int64_t storageSize, ContiguousParam& param)
{
    param.mayBroadcast = false;
    param.mayTranspose = false;
    param.maySlice = false;
    param.mayStridedslice = false;
    param.broadcastSrcShape = viewShape;
    param.broadcastDstShape = viewShape;
    param.viewOffset = offset;

    // Negative strides are currently use AsStrided operation.
    auto minStrideValue = *std::min_element(strides.begin(), strides.end());
    if (minStrideValue < 0) {
        return false;
    }

    auto strideIndexPairs = BuildValidStrideIndexPairs(viewShape, strides, param);
    if (CanOptimizeAsStridedContiguous(strideIndexPairs)) {
        return false;
    }
    // Broadcast
    if (param.mayBroadcast) {
        param.shape = op::ToShapeVector(viewShape);
        // If continuity is met before sorting stride, only BroadcastTo operation is required.
        if (IsContiguous(strideIndexPairs)) {
            return true;
        }
    }

    // Set transpose perm
    if (param.mayTranspose) {
        std::sort(strideIndexPairs.rbegin(), strideIndexPairs.rend());
        // Set perm params for transpose operation.
        param.perm.resize(strideIndexPairs.size());
        for (int64_t i = 0; i < static_cast<int64_t>(strideIndexPairs.size()); i++) {
            param.perm[strideIndexPairs[i].second.first] = static_cast<int64_t>(i);
        }
    }

    // Simple shape
    op::Shape simpleShape;
    op::Strides simpleStrides(strideIndexPairs.size());
    for (size_t i = 0; i < strideIndexPairs.size(); i++) {
        simpleShape.AppendDim(strideIndexPairs[i].second.second);
        simpleStrides[i] = strideIndexPairs[i].first;
    }

    // Set transpose shape
    if (param.mayTranspose) {
        if (IsContiguous(strideIndexPairs)) {
            // transpose->broadcast
            SetTransposeShapeWithSrcShape(simpleShape, param);
            return true;
        }
    }

    // When there is only one dimension, add one dimension to identify whether
    // optimization can be performed by slice operation.
    // Exp.
    //  Tensor with shape [10] stride [2] storage [20]
    //  can considered to be tensor with shape [10, 1] stride [2, 1] storage [10, 2],
    //  which can be sliced to contiguous tensor.
    if (strideIndexPairs.size() == 1) {
        simpleShape.AppendDim(1);
        simpleStrides.push_back(1);
    }

    // Slice or StridedSlice
    int64_t gcdValue = simpleStrides[0];
    if (!OptimizeSlice(simpleShape, simpleStrides, param, gcdValue)) {
        return false;
    }

    // If both Slice and Transpose exist, slice and then Transpose are performed.
    if (param.maySlice) {
        if (!BuildSliceParams(simpleShape, simpleStrides, offset, storageSize, param)) {
            return false;
        }
        if (param.mayTranspose) {
            if (CanReplaceSliceTransposeWithAsStrided(viewShape, strides)) {
                return false;
            }
            SetTransposeShapeWithSrcShape(param.sliceDstShape, param);
        }
        return true;
    }

    if (param.mayStridedslice) {
        if (!BuildStridedSliceParams(simpleShape, simpleStrides, storageSize, gcdValue, param)) {
            return false;
        }
        if (param.mayTranspose) {
            SetTransposeShapeWithSrcShape(param.stridedsliceDstShape, param);
        }
        return true;
    }
    return false;
}

const aclTensor* OptimizeContiguous(const aclTensor* tensor, ContiguousParam& param, aclOpExecutor* executor)
{
    auto dataType = tensor->GetDataType();
    // When the slice or sliced operation is involved, view_offset must start from 0.
    auto currentTensor = UnSafeReshape(tensor, tensor->GetViewShape(), executor);
    currentTensor->SetViewOffset(param.viewOffset);

    if (param.maySlice) {
        currentTensor = UnSafeReshape(currentTensor, param.sliceSrcShape, executor);
        auto offset = executor->AllocIntArray(param.offset.data(), param.offset.size());
        auto size = executor->AllocIntArray(param.size.data(), param.size.size());
        currentTensor = Slice(currentTensor, offset, size, executor);
        CHECK_RET(currentTensor != nullptr, nullptr);
    }

    if (param.mayStridedslice) {
        currentTensor = UnSafeReshape(currentTensor, param.stridedsliceSrcShape, executor);
        auto stridedSliceDstTensor = executor->AllocTensor(param.stridedsliceDstShape, dataType);
        auto begin = executor->ConvertToTensor(param.begin.data(), param.begin.size(), TYPE_INT64);
        auto end = executor->ConvertToTensor(param.end.data(), param.end.size(), TYPE_INT64);
        auto strides = executor->ConvertToTensor(param.strides.data(), param.strides.size(), TYPE_INT64);
        currentTensor = StridedSlice(currentTensor, stridedSliceDstTensor, begin, end, strides, executor);
        CHECK_RET(currentTensor != nullptr, nullptr);
    }

    if (param.mayTranspose) {
        currentTensor = UnSafeReshape(currentTensor, param.transposeSrcShape, executor); // transpose不改变view_offset
        auto transposeDstTensor =
            executor->AllocTensor(param.transposeDstShape, dataType, currentTensor->GetStorageFormat());
        auto perm = executor->ConvertToTensor(param.perm.data(), param.perm.size(), TYPE_INT64);
        currentTensor = Transpose(currentTensor, transposeDstTensor, perm, executor);
        CHECK_RET(currentTensor != nullptr, nullptr);
    }

    if (param.mayBroadcast) {
        currentTensor = UnSafeReshape(currentTensor, param.broadcastSrcShape, executor);
        if (currentTensor->GetDataType() == op::ToOpDataType(ACL_BOOL)) {
            currentTensor = l0op::Cast(currentTensor, op::ToOpDataType(ACL_INT32), executor);
            CHECK_RET(currentTensor != nullptr, nullptr);
            auto shapeArray = executor->AllocIntArray(param.shape.data(), param.shape.size());
            CHECK_RET(shapeArray != nullptr, nullptr);
            currentTensor = l0op::BroadcastTo(currentTensor, shapeArray, executor);
            CHECK_RET(currentTensor != nullptr, nullptr);
            currentTensor = l0op::Cast(currentTensor, op::ToOpDataType(ACL_BOOL), executor);
        } else {
            auto broadcastDstTensor = executor->AllocTensor(param.broadcastDstShape, dataType);
            auto shape = executor->ConvertToTensor(param.shape.data(), param.shape.size(), TYPE_INT64);
            currentTensor = BroadcastTo(currentTensor, broadcastDstTensor, shape, executor);
        }
        CHECK_RET(currentTensor != nullptr, nullptr);
    }
    currentTensor = UnSafeReshape(currentTensor, tensor->GetViewShape(), executor);
    return currentTensor;
}

const aclTensor* AsStridedToContiguous(const aclTensor* x, aclOpExecutor* executor)
{
    auto sizeV = op::ToShapeVector(x->GetViewShape());
    auto size = executor->ConvertToTensor(sizeV.data(), sizeV.size(), TYPE_INT64);

    auto strides = x->GetViewStrides();
    auto stride = executor->ConvertToTensor(strides.data(), strides.size(), TYPE_INT64);
    int64_t offset[1] = {0};
    auto storageOffset = executor->ConvertToTensor(offset, 1, TYPE_INT64);

    auto out = executor->AllocTensor(x->GetViewShape(), x->GetDataType());

    op::Shape newStorageShape{1};
    int64_t actualShapeSize = 1;
    auto tempStorageShape = x->GetStorageShape();
    for (uint64_t i= 0; i < tempStorageShape.GetDimNum(); i++) {
        actualShapeSize *= tempStorageShape.GetDim(i);
    }
    actualShapeSize -= x->GetViewOffset();
    newStorageShape.SetDim(0, actualShapeSize);

    auto xView = executor->CreateView(x, newStorageShape, x->GetViewOffset());
    xView->SetViewShape(x->GetViewShape());
    xView->SetViewStrides(x->GetViewStrides());

    return AsStrided(xView, out, size, stride, storageOffset, executor);
}

const aclTensor* ViewCopyToView(const aclTensor* x, const aclTensor* y, aclOpExecutor* executor)
{
    auto xView = const_cast<aclTensor*>(x);
    if (x->GetViewOffset() != 0) {
        executor->AbandonCache();
        xView = executor->CreateView(x, x->GetViewShape(), 0);
        xView->SetStorageAddr(x->GetStorageAddr());
        xView->SetStorageOffset(x->GetViewOffset() + x->GetStorageOffset());
    }

    // offset has been applied to attribute, set view_offset to zero to avoid calculation twice.
    auto yView = executor->CreateView(y, y->GetViewShape(), 0);
    yView->SetStorageShape(y->GetStorageShape());
    yView->SetViewStrides(y->GetViewStrides());
    yView->SetOriginalShape(y->GetOriginalShape());
    auto result = ViewCopy(
        yView, y->GetViewShape(), y->GetViewStrides(), y->GetViewOffset(), xView, xView->GetViewShape(),
        xView->GetViewStrides(), xView->GetViewOffset(), executor);
    CHECK_RET(result != nullptr, nullptr);
    return y;
}

/**
 * @brief 是否可以通过高效的AiCore算子实现View操作
 *  目标Tensor的view是storage上填满的一段数据（不要求满足连续），可以优化的主要是以下场景
 *  1. 通过Transpose重排数据后满足目标tensor的stride
 *  处理：
 *  1. 化简shape和stride
 *  2. 判断排序后是否满足连续，连续则需要Tranpose
 *  3. 其他场景，走ViewCopy
 * @param viewShape
 * @param strides
 * @param offset
 * @param storage_size
 * @param param
 * @return
 */
bool CanOptimizeView(const op::Shape& viewShape, const op::Strides& strides, int64_t offset, ContiguousParam& param)
{
    (void)offset;
    param.mayBroadcast = false;
    param.mayTranspose = false;
    param.maySlice = false;
    param.mayStridedslice = false;
    param.broadcastSrcShape = viewShape;
    param.broadcastDstShape = viewShape;
    auto strideIndexPairs = BuildValidStrideIndexPairs(viewShape, strides, param);
    if (param.mayBroadcast) {
        return false;
    }

    op::Shape simpleShape;
    op::Strides simpleStrides(strideIndexPairs.size());
    for (size_t i = 0; i < strideIndexPairs.size(); i++) {
        simpleShape.AppendDim(strideIndexPairs[i].second.second);
        simpleStrides[i] = strideIndexPairs[i].first;
    }

    // Set perm for transpose operation.
    if (param.mayTranspose) {
        std::sort(strideIndexPairs.rbegin(), strideIndexPairs.rend());
        param.perm.resize(strideIndexPairs.size());
        for (int64_t i = 0; i < static_cast<int64_t>(strideIndexPairs.size()); i++) {
            param.perm[i] = strideIndexPairs[i].second.first;
        }
    }

    // Update shapes for transpose.
    if (param.mayTranspose) {
        if (IsContiguous(strideIndexPairs)) {
            SetTransposeShapeWithSrcShape(simpleShape, param);
            return true;
        }
    }
    return false;
}

/**
 * @brief
 * @param x
 * @param y
 * @param param
 * @param executor
 * @return
 */
const aclTensor* OptimizeView(const aclTensor* x, const aclTensor* y, ContiguousParam& param, aclOpExecutor* executor)
{
    // x --> Reshape() -->  Transpose --> Reshape() --> y
    // x->Transpose->y
    if (param.mayTranspose) {
        auto yView = UnSafeReshape(y, param.transposeDstShape, executor);
        auto currentTensor = UnSafeReshape(x, param.transposeSrcShape, executor);
        auto perm = executor->ConvertToTensor(param.perm.data(), param.perm.size(), TYPE_INT64);
        currentTensor = Transpose(currentTensor, yView, perm, executor);
        CHECK_RET(currentTensor != nullptr, nullptr);
    }
    return y;
}

const aclTensor* ResetFormat(const aclTensor* x, const aclTensor* y)
{
    auto constX = const_cast<aclTensor*>(x);
    constX->SetViewFormat(y->GetViewFormat());
    constX->SetStorageFormat(y->GetStorageFormat());
    constX->SetOriginalFormat(y->GetOriginalFormat());
    return constX;
}

const aclTensor* Contiguous(const aclTensor* x, aclOpExecutor* executor)
{
    L0_DFX(Contiguous, x);

    if (op::IsContiguous(x)) {
        return UnSafeReshape(x, x->GetViewShape(), executor);
    }

    if (!Validate(x)) {
        OP_LOGE(ACL_ERROR_INVALID_PARAM, "Invalid input tensor: %s", x->ToString().GetString());
        return nullptr;
    }

    auto viewShape = x->GetViewShape();
    auto viewStrides = x->GetViewStrides();
    auto viewOffset = x->GetViewOffset();
    auto storageSize = x->GetStorageShape().GetShapeSize();

    ContiguousParam param;
    if (CanOptimizeContiguous(viewShape, viewStrides, viewOffset, storageSize, param)) {
        auto contiguousTensor = OptimizeContiguous(x, param, executor);
        if (contiguousTensor == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER, "OptimizeContiguous failed.");
            return nullptr;
        }
        return ResetFormat(contiguousTensor, x);
    }

    auto contiguousTensor = AsStridedToContiguous(x, executor);
    if (contiguousTensor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Convert tensor to contiguous failed.");
        return nullptr;
    }
    return ResetFormat(contiguousTensor, x);
}

inline bool CheckViewCopyParams(const aclTensor* x, const aclTensor* y)
{
    if (y->IsFromWorkspace()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output tensor should not from workspace.");
        return false;
    }
    // Check data type
    if (x->GetDataType() != y->GetDataType()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Input tensor's dtype[%s] should be same with output's dtype[%s].",
            op::ToString(x->GetDataType()).GetString(), op::ToString(y->GetDataType()).GetString());
        return false;
    }
    // Check format
    if (x->GetStorageFormat() != y->GetStorageFormat() &&
        x->GetStorageFormat() != static_cast<op::Format>(ACL_FORMAT_ND)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Input tensor's format[%s] should be same with output's format[%s].",
            op::ToString(x->GetStorageFormat()).GetString(), op::ToString(y->GetStorageFormat()).GetString());
        return false;
    }
    // Check shape
    auto const& xShape = x->GetViewShape();
    auto const& yShape = y->GetViewShape();
    if (xShape != yShape) {
        if (!(xShape.GetShapeSize() == 1 && yShape.GetShapeSize() == 1)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Input tensor's shape[%s] should be same with output's shape[%s].",
                op::ToString(x->GetViewShape()).GetString(), op::ToString(y->GetViewShape()).GetString());
            return false;
        }
    }

    return true;
}

const aclTensor* ViewCopy(const aclTensor* x, const aclTensor* y, aclOpExecutor* executor)
{
    L0_DFX(ViewCopy, x, y);
    OP_LOGD("ViewCopy data addr: %p", y->GetData());
    if (!CheckViewCopyParams(x, y)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Params check failed.");
        return nullptr;
    }

    if (op::IsContiguous(x) && op::IsContiguous(y)) {
        // If the input is not workspace, copy data from x to y.
        if (!x->IsFromWorkspace()) {
            // x and y point to the same tensor.
            if (x->GetData() == y->GetData()) {
                executor->AbandonCache(true);
                OP_LOGD("The input and output point to the same address.");
                return y;
            }
            auto yView =
                (x->GetStorageShape() == y->GetStorageShape()) ? y : UnSafeReshape(y, y->GetViewShape(), executor);
            OP_LOGD("The input and output are created by the user.");
            return TensorMove(x, yView, executor);
        }
        if (SupportMoveAlign() ||
            y->GetStorageShape().GetShapeSize() == (y->GetViewShape().GetShapeSize() + y->GetViewOffset())) {
            // If the input is workspace, change the input to the output address and modify the workspace flag.
            auto viewOut = const_cast<aclTensor*>(x);
            viewOut->SetFromWorkspace(y->IsFromWorkspace());
            viewOut->SetStorageAddr(y->GetStorageAddr());
            viewOut->SetStorageOffset(y->GetViewOffset() + y->GetStorageOffset());
            executor->AddTensorRelation(y, viewOut);
            return y;
        }
        OP_LOGD("Contiguous copy with move align.");
        // For some platform that do not support move align.
        auto yView = (x->GetStorageShape() == y->GetStorageShape()) ? y : UnSafeReshape(y, y->GetViewShape(), executor);
        if (op::CopyNpuToNpu(x, yView, executor) == ACLNN_SUCCESS) {
            return y;
        }
    }

    if (!IsContiguous(x)) {
        return ViewCopyToView(x, y, executor);
    }

    ContiguousParam param;
    auto viewShape = y->GetViewShape();
    auto viewStrides = y->GetViewStrides();
    auto viewOffset = y->GetViewOffset();
    if (CanOptimizeView(viewShape, viewStrides, viewOffset, param)) {
        return OptimizeView(x, y, param, executor);
    }

    return ViewCopyToView(x, y, executor);
}

const aclTensor* PickViewAsContiguous(const aclTensor* x, aclOpExecutor* executor)
{
    L0_DFX(PickViewAsContiguous, x);
    if (!op::CanPickViewAsContiguous(x)) {
        OP_LOGE(ACL_ERROR_INVALID_PARAM, "Tensor can not pick view as contiguous.");
        return nullptr;
    }
    // 如果是Transpose的
    return UnSafeReshape(x, x->GetViewShape(), executor);
}

const aclTensor* ReViewToOut(const aclTensor* x, const aclTensor* y, aclOpExecutor* executor)
{
    L0_DFX(ReViewToOut, x, y);
    if (!op::CanPickViewAsContiguous(y)) {
        OP_LOGE(ACL_ERROR_INVALID_PARAM, "Tensor can not review to out.");
        return nullptr;
    }

    if (y->IsFromWorkspace()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output tensor should not from workspace.");
        return nullptr;
    }

    auto viewOut = const_cast<aclTensor*>(x);
    viewOut->SetFromWorkspace(y->IsFromWorkspace());
    viewOut->SetStorageAddr(y->GetStorageAddr());
    viewOut->SetStorageOffset(y->GetViewOffset() + y->GetStorageOffset());
    executor->AddTensorRelation(y, viewOut);
    return y;
}

} // namespace l0op
