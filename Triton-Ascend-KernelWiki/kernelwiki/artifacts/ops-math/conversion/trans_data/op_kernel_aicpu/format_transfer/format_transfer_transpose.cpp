/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "format_transfer_transpose.h"

#include <memory>

#include "format_transfer_utils.h"
#include "formats_definitions.h"
#include "kernel_util.h"
#include "log.h"
#include "securec.h"
#include "status.h"

namespace aicpu {
namespace formats {
namespace {
std::map<Format, std::map<Format, std::vector<int64_t>>> perm_args{
    {FORMAT_NCHW,
     {{FORMAT_NHWC, std::vector<int64_t>({kNchwN, kNchwH, kNchwW, kNchwC})},
      {FORMAT_HWCN, std::vector<int64_t>({kNchwH, kNchwW, kNchwC, kNchwN})},
      {FORMAT_CHWN, std::vector<int64_t>({kNchwC, kNchwH, kNchwW, kNchwN})}}},
    {FORMAT_NHWC,
     {{FORMAT_NCHW, std::vector<int64_t>({kNhwcN, kNhwcC, kNhwcH, kNhwcW})},
      {FORMAT_CHWN, std::vector<int64_t>({kNhwcC, kNhwcH, kNhwcW, kNhwcN})},
      {FORMAT_HWCN, std::vector<int64_t>({kNhwcH, kNhwcW, kNhwcC, kNhwcN})}}},
    {FORMAT_HWCN,
     {{FORMAT_NCHW, std::vector<int64_t>({kHwcnN, kHwcnC, kHwcnH, kHwcnW})},
      {FORMAT_NHWC, std::vector<int64_t>({kHwcnN, kHwcnH, kHwcnW, kHwcnC})},
      {FORMAT_CHWN, std::vector<int64_t>({kHwcnC, kHwcnH, kHwcnW, kHwcnN})}}},
    {FORMAT_CHWN,
     {{FORMAT_NCHW, std::vector<int64_t>({kChwnN, kChwnC, kChwnH, kChwnW})},
      {FORMAT_NHWC, std::vector<int64_t>({kChwnN, kChwnH, kChwnW, kChwnC})},
      {FORMAT_HWCN, std::vector<int64_t>({kChwnH, kChwnW, kChwnC, kChwnN})}}},
};

bool ShapeArgValid(const std::vector<int64_t>& src_shape, const std::vector<int64_t>& perm_arg)
{
    if (src_shape.empty()) {
        KERNEL_LOG_ERROR("Failed to transpose, src shape is empty");
        return false;
    }
    for (auto dim : src_shape) {
        if (dim < 0) {
            KERNEL_LOG_ERROR("Failed to transpose, negative dim [%ld] in src shape [%s]", dim,
                             FmtToStr(VectorToString(src_shape)).c_str());
            return false;
        }
    }
    if (perm_arg.size() != src_shape.size()) {
        KERNEL_LOG_ERROR("Failed to transpose, the size of src shape [%s] and perm arg [%s] are "
                         "different",
                         FmtToStr(src_shape.size()).c_str(), FmtToStr(perm_arg.size()).c_str());
        return false;
    }

    std::vector<int64_t> exists(perm_arg.size());
    for (auto perm : perm_arg) {
        if (perm < 0 || static_cast<size_t>(perm) >= perm_arg.size() || ++exists[perm] > 1) {
            KERNEL_LOG_ERROR("Failed to transpose, invalid perm [%s], perm arg [%s]", FmtToStr(perm).c_str(),
                             FmtToStr(VectorToString(perm_arg)).c_str());
            return false;
        }
    }
    return true;
}

bool IsTransposeArgValid(const uint8_t* src, const std::vector<int64_t>& src_shape, DataType src_data_type,
                         const std::vector<int64_t>& perm_arg)
{
    if (src == nullptr) {
        KERNEL_LOG_ERROR("Src should not be nullptr");
        return false;
    }
    if (GetSizeByDataType(src_data_type) < 0) {
        KERNEL_LOG_ERROR("The data type [%s] is not support", DTypeStr(src_data_type).c_str());
        return false;
    }
    return ShapeArgValid(src_shape, perm_arg);
}

void GenHeads(const std::vector<int64_t>& shape, std::vector<int64_t>& heads)
{
    heads.resize(shape.size());
    heads[shape.size() - 1] = 1;
    for (auto i = static_cast<int64_t>(shape.size() - 2); i >= 0; --i) {
        heads[i] = shape[i + 1] * heads[i + 1];
    }
}

int64_t GenOffset(const std::vector<int64_t>& offsets, const std::vector<int64_t>& indexes)
{
    int64_t offset = 0;
    for (size_t i = 0; i < indexes.size(); ++i) {
        offset += offsets[i] * indexes[i];
    }
    return offset;
}

void AddOne(const std::vector<int64_t>& shape, std::vector<int64_t>& indexes)
{
    size_t i = indexes.size() - 1;
    indexes[i]++;
    while (i > 0) {
        if (indexes[i] >= shape[i]) {
            indexes[i] = 0;
            indexes[i - 1]++;
            --i;
        } else {
            break;
        }
    }
}

void TransShapeByPerm(const std::vector<int64_t>& src_shape, const std::vector<int64_t>& perm_arg,
                      std::vector<int64_t>& dst_shape)
{
    dst_shape.resize(src_shape.size());
    for (size_t i = 0; i < perm_arg.size(); ++i) {
        dst_shape[i] = src_shape[perm_arg[i]];
    }
}
} // namespace

uint32_t Transpose(const uint8_t* src, uint8_t* dst, const std::vector<int64_t>& src_shape, DataType src_data_type,
                   const std::vector<int64_t>& perm_arg)
{
    if (!IsTransposeArgValid(src, src_shape, src_data_type, perm_arg)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    std::vector<int64_t> dst_shape;
    TransShapeByPerm(src_shape, perm_arg, dst_shape);
    std::vector<int64_t> src_origin_ordered_heads;
    GenHeads(src_shape, src_origin_ordered_heads);
    std::vector<int64_t> src_heads;
    TransShapeByPerm(src_origin_ordered_heads, perm_arg, src_heads);

    int64_t dst_ele_num = GetItemNumByShape(dst_shape);
    int64_t data_size = GetSizeByDataType(src_data_type);
    int64_t dst_size = data_size * dst_ele_num;

    KERNEL_LOG_INFO("Begin to transpose, src shape [%s], perm arg [%s], dst shape [%s], data "
                    "type [%s]",
                    VectorToString(src_shape).c_str(), VectorToString(perm_arg).c_str(),
                    VectorToString(dst_shape).c_str(), DTypeStr(src_data_type).c_str());
    if (dst_ele_num == 0) {
        return KERNEL_STATUS_OK;
    }

    int64_t dst_index = 0;
    std::vector<int64_t> dst_indexes(dst_shape.size());
    while (dst_index < dst_ele_num) {
        auto src_offset = GenOffset(src_heads, dst_indexes) * data_size;
        auto dst_offset_bytes = dst_index * data_size;
        auto protected_size = dst_size - dst_offset_bytes < static_cast<int64_t>(SECUREC_MEM_MAX_LEN) ?
                                  dst_size - dst_offset_bytes :
                                  static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
        auto ret = memcpy_s(dst + dst_offset_bytes, static_cast<size_t>(protected_size), src + src_offset,
                            static_cast<size_t>(data_size));
        if (ret != EOK) {
            KERNEL_LOG_ERROR("Failed to transpose, src shape [%s], perm arg [%s], dst shape [%s], "
                             "failed to write to dst offset [%ld], current dim offset [%s]",
                             VectorToString(src_shape).c_str(), VectorToString(perm_arg).c_str(),
                             VectorToString(dst_shape).c_str(), dst_offset_bytes, VectorToString(dst_indexes).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
        }
        AddOne(dst_shape, dst_indexes);
        ++dst_index;
    }
    return KERNEL_STATUS_OK;
}

uint32_t GetPermByForamt(Format src_format, Format dst_format, std::vector<int64_t>& perm)
{
    auto dst_iter = perm_args.find(src_format);
    if (dst_iter == perm_args.end()) {
        KERNEL_LOG_ERROR("Failed to trans shape , do not support transpose from format [%s] to "
                         "[%s]",
                         FormatToSerialString(src_format).c_str(), FormatToSerialString(dst_format).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    auto iter = dst_iter->second.find(dst_format);
    if (iter == dst_iter->second.end()) {
        KERNEL_LOG_ERROR("Failed to trans shape , do not support transpose from format [%s] to "
                         "[%s]",
                         FormatToSerialString(src_format).c_str(), FormatToSerialString(dst_format).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    perm = iter->second;
    return KERNEL_STATUS_OK;
}

uint32_t FormatTransferTranspose::TransFormat(const TransArgs& args)
{
    std::vector<int64_t> expected_shape;
    auto ret = TransShape(args, expected_shape, false);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    if (!IsTransShapeDstCorrect(args, expected_shape)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }

    return Transpose(args.data, args.output, args.src_shape, args.src_data_type,
                     perm_args[args.src_format][args.dst_format]);
}

uint32_t FormatTransferTranspose::TransShape(const TransArgs& args, std::vector<int64_t>& dst_shape, bool reverse)
{
    (void)reverse;
    std::vector<int64_t> perm_arg;
    const std::vector<int64_t> src_shape = args.src_shape;
    if (GetPermByForamt(args.src_format, args.dst_format, perm_arg) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (!ShapeArgValid(src_shape, perm_arg)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    TransShapeByPerm(src_shape, perm_arg, dst_shape);
    return KERNEL_STATUS_OK;
}

REGISTER_FORMAT_TRANSFER(FormatTransferTranspose, FORMAT_NCHW, FORMAT_NHWC)
REGISTER_FORMAT_TRANSFER(FormatTransferTranspose, FORMAT_NCHW, FORMAT_HWCN)
REGISTER_FORMAT_TRANSFER(FormatTransferTranspose, FORMAT_NCHW, FORMAT_CHWN)
REGISTER_FORMAT_TRANSFER(FormatTransferTranspose, FORMAT_NHWC, FORMAT_NCHW)
REGISTER_FORMAT_TRANSFER(FormatTransferTranspose, FORMAT_NHWC, FORMAT_CHWN)
REGISTER_FORMAT_TRANSFER(FormatTransferTranspose, FORMAT_NHWC, FORMAT_HWCN)
REGISTER_FORMAT_TRANSFER(FormatTransferTranspose, FORMAT_HWCN, FORMAT_NCHW)
REGISTER_FORMAT_TRANSFER(FormatTransferTranspose, FORMAT_HWCN, FORMAT_NHWC)
REGISTER_FORMAT_TRANSFER(FormatTransferTranspose, FORMAT_HWCN, FORMAT_CHWN)
REGISTER_FORMAT_TRANSFER(FormatTransferTranspose, FORMAT_CHWN, FORMAT_NCHW)
REGISTER_FORMAT_TRANSFER(FormatTransferTranspose, FORMAT_CHWN, FORMAT_NHWC)
REGISTER_FORMAT_TRANSFER(FormatTransferTranspose, FORMAT_CHWN, FORMAT_HWCN)
} // namespace formats
} // namespace aicpu
