/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "format_transfer_fractal_nz.h"

#include "format_transfer_utils.h"
#include "cpu_kernel_utils.h"
#include "kernel_util.h"
#include "log.h"
#include "securec.h"
#include "status.h"

#include <atomic>

using namespace std;

namespace aicpu {
namespace formats {
namespace {
const int64_t kDimDefaultValue = 1;
const int kDimSize4D = 4;
const size_t kSingleDim = 1;
const size_t kNdDimIndexN = 0;
const size_t kNdDimIndexH = 1;
const size_t kNdDimIndexW = 2;
const size_t kDimDValueBNdFNz = 2; // dim d-value between Nd and FractalZz
const size_t kNdDimCountBackwardsW = 1;
const size_t kNdDimCountBackwardsWH = 2;
const size_t kFNzDimCountBackwardsW0 = 1;
const size_t kFNzDimCountBackwardsW0H0 = 2;
const size_t kFNzDimCountBackwardsW0H0H1 = 3;
const size_t kFNzDimCountBackwardsW0H0H1W1 = 4;
const int64_t kParallelDataSize = 128 * 1024;
const size_t kNumTwo = 2;

bool IsDataTypeSupport(DataType data_type) { return GetSizeByDataType(data_type) > 0; }

using ShapeVector = vector<int64_t>;
bool IsEvenValue(size_t value) { return value % kNumTwo == 0; }

uint32_t FLOAT4e2m1ShapeExtension(TransArgs& args)
{
    size_t size_src = args.src_shape.size();
    KERNEL_CHECK_FALSE((size_src >= kNdDimCountBackwardsW), KERNEL_STATUS_PARAM_INVALID,
                       "Size shape size must greater than kNdDimCountBackwardsW.");
    auto last_src_shape = args.src_shape[size_src - kNdDimCountBackwardsW];
    KERNEL_CHECK_FALSE(IsEvenValue(last_src_shape), KERNEL_STATUS_PARAM_INVALID,
                       "The last one src shape [%ld] is not even number, must be even.", last_src_shape);
    args.src_shape[size_src - kNdDimCountBackwardsW] = last_src_shape / kNumTwo;
    size_t size_dst = args.dst_shape.size();
    KERNEL_CHECK_FALSE((size_dst >= kFNzDimCountBackwardsW0H0H1W1), KERNEL_STATUS_PARAM_INVALID,
                       "Dst shape size must greater than kFNzDimCountBackwardsW0H0H1W1.");
    auto w1_dst_shape = args.dst_shape[size_dst - kFNzDimCountBackwardsW0H0H1W1];
    KERNEL_CHECK_FALSE(IsEvenValue(w1_dst_shape), KERNEL_STATUS_PARAM_INVALID,
                       "The w1 dst shape [%ld] is not even number, must be even.", w1_dst_shape);
    args.dst_shape[size_dst - kFNzDimCountBackwardsW0H0H1W1] = w1_dst_shape / kNumTwo;
    return KERNEL_STATUS_OK;
}

uint32_t FLOAT4e1m2ShapeExtension(TransArgs& args) { return FLOAT4e2m1ShapeExtension(args); }

std::map<DataType, std::function<uint32_t(TransArgs&)>> g_shape_extension_fun_map = {
    {DT_FLOAT4_E2M1, FLOAT4e2m1ShapeExtension}, {DT_FLOAT4_E1M2, FLOAT4e1m2ShapeExtension}};

bool CheckShape(Format format, const ShapeVector& shape)
{
    switch (format) {
        case FORMAT_ND:
            return IsShapeValid(shape);
        case FORMAT_NCHW:
        case FORMAT_NHWC:
            return CheckShapeValid(shape, kDimSize4D);
        default:
            string error = "Trans format between " + FmtToStr(FormatToSerialString(format)) +
                           " and [FORMAT_FRACTAL_NZ] is not supported.";
            KERNEL_LOG_ERROR("%s", error.c_str());
            return false;
    }
}

/**
 * After the conversion to two-dimensional matrix, the memory arrangement is
 * small z and large N.
 * @src_shape: N*H*W
 * @dst_shape: N*W1*H1*H0*w0 / N, K/K0, M/M0, M0(h0), K0(w0)
 * @return
 */
uint32_t TransShapeToFracNz(const ShapeVector& src_shape, const ShapeVector& ori_dst_shape, const int64_t& w,
                            ShapeVector& dst_shape, ShapeVector& hw_shape)
{
    dst_shape.clear();
    hw_shape.clear();
    auto w0 = w;
    int64_t h0 = kCubeSize;
    // transdata F_NZ shape is right, h0 may be 1 or 16, w0 may be 1/8/16/32
    auto shape_size = ori_dst_shape.size();
    if (shape_size >= kFNzDimCountBackwardsW0H0H1W1) {
        h0 = ori_dst_shape[shape_size - kFNzDimCountBackwardsW0H0];
        w0 = ori_dst_shape[shape_size - kFNzDimCountBackwardsW0];
    }
    KERNEL_CHECK_FALSE(IsEvenValue(w0), KERNEL_STATUS_PARAM_INVALID, "w0 must be even.");
    switch (src_shape.size()) {
        case kSingleDim:
            dst_shape.push_back(Ceil(src_shape[kNdDimIndexN], w0));
            dst_shape.push_back(kDimDefaultValue);
            dst_shape.push_back(h0);
            dst_shape.push_back(w0);
            hw_shape.push_back(kDimDefaultValue);
            hw_shape.push_back(kDimDefaultValue);
            hw_shape.push_back(src_shape[kNdDimIndexN]);
            if (!IsShapeValid(dst_shape)) {
                KERNEL_LOG_ERROR("Failed to check dst shape [%s]", VectorToString(dst_shape).c_str());
                return KERNEL_STATUS_PARAM_INVALID;
            }
            return KERNEL_STATUS_OK;
        default:
            auto size = src_shape.size();
            int64_t times = 1;
            for (size_t i = 0; i != size - kDimDValueBNdFNz; i++) {
                dst_shape.push_back(src_shape[i]);
                times *= src_shape[i];
            }
            dst_shape.push_back(Ceil(src_shape[size - kNdDimCountBackwardsW], w0));
            dst_shape.push_back(Ceil(src_shape[size - kNdDimCountBackwardsWH], h0));
            dst_shape.push_back(h0);
            dst_shape.push_back(w0);
            hw_shape.push_back(times);
            hw_shape.push_back(src_shape[size - kNdDimCountBackwardsWH]);
            hw_shape.push_back(src_shape[size - kNdDimCountBackwardsW]);
            if (!IsShapeValid(dst_shape)) {
                KERNEL_LOG_ERROR("Failed to check dst shape [%s]", VectorToString(dst_shape).c_str());
                return KERNEL_STATUS_PARAM_INVALID;
            }
            return KERNEL_STATUS_OK;
    }
}

uint32_t CheckShapeRelation(const TransArgs& args, ShapeVector& hw_shape)
{
    ShapeVector expect_src_shape;
    int64_t w0 = GetC0ValueForTransFormat(args.src_data_type, args.input_format, args.output_format);
    auto ret = TransShapeToFracNz(args.dst_shape, args.src_shape, w0, expect_src_shape, hw_shape);
    if (ret != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("Trans shape from [%s] to [%s], shape [%s] to [%s], data type [%s] "
                         "failed",
                         FormatToSerialString(args.dst_format).c_str(), FormatToSerialString(args.src_format).c_str(),
                         VectorToString(args.dst_shape).c_str(), VectorToString(args.src_shape).c_str(),
                         DTypeStr(args.src_data_type).c_str());
        return ret;
    }
    if (!IsTransShapeSrcCorrect(args, expect_src_shape)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

struct NdFracNzParams {
    int64_t size;
    int64_t dst_size;
    int64_t times;
    int64_t w;
    int64_t w0;
    int64_t hw;
    int64_t h1h0w0;
    int64_t w1h1h0w0;
    int64_t num_w1;
};

uint32_t CopyNdToFracNzBlock(const TransArgs& args, const NdFracNzParams& p, int64_t h1h0_head, int64_t src_h_head)
{
    for (int64_t w1_idx = 0; w1_idx < p.num_w1; w1_idx++) {
        const int64_t dst_offset = (h1h0_head + w1_idx * p.h1h0w0) * p.size;
        const int64_t src_offset = (src_h_head + w1_idx * p.w0) * p.size;
        const int64_t protected_size = (p.dst_size - dst_offset < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)) ?
                                           (p.dst_size - dst_offset) :
                                           static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
        const errno_t ret = memcpy_s(args.output + dst_offset, static_cast<size_t>(protected_size),
                                     args.data + src_offset, static_cast<size_t>(p.size * p.w0));
        if (ret != EOK) {
            KERNEL_LOG_ERROR("Failed to operate the dst memory at offset [%ld], "
                             "error-code "
                             "[%d]",
                             dst_offset, ret);
            return KERNEL_STATUS_INNER_ERROR;
        }
    }
    return KERNEL_STATUS_OK;
}

uint32_t CopyNdToFracNzTail(const TransArgs& args, const NdFracNzParams& p, int64_t h1h0_head, int64_t src_h_head)
{
    const int64_t w1_head = p.num_w1 * p.w0;
    for (int64_t w0_idx = 0; w1_head + w0_idx < p.w; w0_idx++) {
        const int64_t src_w_idx = w1_head + w0_idx;
        const int64_t dst_offset = (h1h0_head + p.num_w1 * p.h1h0w0 + w0_idx) * p.size;
        const int64_t src_offset = (src_h_head + src_w_idx) * p.size;
        const int64_t protected_size = (p.dst_size - dst_offset < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)) ?
                                           (p.dst_size - dst_offset) :
                                           static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
        const errno_t ret = memcpy_s(args.output + dst_offset, static_cast<size_t>(protected_size),
                                     args.data + src_offset, static_cast<size_t>(p.size));
        if (ret != EOK) {
            KERNEL_LOG_ERROR("Failed to operate the dst memory at offset [%ld], "
                             "protected_size is [%ld], size is[%ld], error-code "
                             "[%d]",
                             dst_offset, protected_size, p.size, ret);
            return KERNEL_STATUS_INNER_ERROR;
        }
    }
    return KERNEL_STATUS_OK;
}

void CopyNdToFracNzRange(const TransArgs& args, const NdFracNzParams& p, int64_t start, int64_t end,
                         std::atomic<bool>& failed)
{
    // 针对h轴开启多核并行
    for (int64_t h1h0_idx = start; h1h0_idx < end; h1h0_idx++) {
        if (failed.load(std::memory_order_relaxed)) {
            return;
        }
        const int64_t h1h0_offset = h1h0_idx * p.w0;
        const int64_t src_h_offset = h1h0_idx * p.w;
        for (int64_t times_idx = 0; times_idx < p.times; times_idx++) {
            const int64_t h1h0_head = h1h0_offset + times_idx * p.w1h1h0w0;
            const int64_t src_h_head = src_h_offset + times_idx * p.hw;
            if (CopyNdToFracNzBlock(args, p, h1h0_head, src_h_head) != KERNEL_STATUS_OK) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            if (CopyNdToFracNzTail(args, p, h1h0_head, src_h_head) != KERNEL_STATUS_OK) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
        }
    }
}

uint32_t TransFormatFromNdToFracNz(const TransArgs& args, const ShapeVector& hw_shape)
{
    NdFracNzParams p;
    p.size = GetSizeByDataType(args.src_data_type);
    // data size will not be greater than INT_MAX
    const int64_t dst_element_shape = GetItemNumByShape(args.dst_shape);
    p.dst_size = dst_element_shape * p.size;
    if (p.dst_size == 0) {
        KERNEL_LOG_DEBUG("Empty tensor");
        return KERNEL_STATUS_OK;
    }
    p.times = hw_shape.at(kNdDimIndexN);
    const int64_t h = hw_shape.at(kNdDimIndexH);
    p.w = hw_shape.at(kNdDimIndexW);
    p.hw = h * p.w;

    const auto shape_size = args.dst_shape.size();
    const int64_t w1 = args.dst_shape[shape_size - kFNzDimCountBackwardsW0H0H1W1];
    const int64_t h1 = args.dst_shape[shape_size - kFNzDimCountBackwardsW0H0H1];
    const int64_t h0 = args.dst_shape[shape_size - kFNzDimCountBackwardsW0H0];
    p.w0 = args.dst_shape[shape_size - kFNzDimCountBackwardsW0];
    const int64_t h1h0 = h1 * h0;
    p.h1h0w0 = h1h0 * p.w0;
    p.w1h1h0w0 = w1 * p.h1h0w0;
    // w0 not equal 0
    p.num_w1 = p.w / p.w0;
    (void)memset_s(args.output, static_cast<size_t>(p.dst_size), 0, static_cast<size_t>(p.dst_size));
    std::atomic<bool> sharder_failed{false};
    auto sharder = [&args, &p, &sharder_failed](int64_t start, int64_t end) {
        CopyNdToFracNzRange(args, p, start, end, sharder_failed);
    };

    const int64_t parallel_core_number = GetMaxCompilerCoreNum();
    if (parallel_core_number > 1 && dst_element_shape >= kParallelDataSize) {
        KERNEL_LOG_DEBUG("parallel_core_number is [%ld]", parallel_core_number);
        KERNEL_HANDLE_ERROR(aicpu::CpuKernelUtils::ParallelFor(*(args.ctx), h, 1, sharder),
                            "TransData ParallelFor Compute failed.")
    } else {
        sharder(0, h);
    }
    if (sharder_failed.load(std::memory_order_relaxed)) {
        KERNEL_LOG_ERROR("Failed to copy ND to FRACTAL_NZ in sharder.");
        return KERNEL_STATUS_INNER_ERROR;
    }
    return KERNEL_STATUS_OK;
}

uint32_t CopyFracNzToNdBlock(const TransArgs& args, const NdFracNzParams& p, int64_t h1h0_head, int64_t dst_h_head)
{
    for (int64_t w1_idx = 0; w1_idx < p.num_w1; w1_idx++) {
        const int64_t src_offset = (h1h0_head + w1_idx * p.h1h0w0) * p.size;
        const int64_t dst_offset = (dst_h_head + w1_idx * p.w0) * p.size;
        const int64_t protected_size = p.dst_size - dst_offset < static_cast<int64_t>(SECUREC_MEM_MAX_LEN) ?
                                           p.dst_size - dst_offset :
                                           static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
        const errno_t ret = memcpy_s(args.output + dst_offset, static_cast<size_t>(protected_size),
                                     args.data + src_offset, static_cast<size_t>(p.size * p.w0));
        if (ret != EOK) {
            KERNEL_LOG_ERROR("Failed to operate the dst memory at offset [%ld], error-code "
                             "[%d]",
                             dst_offset, ret);
            return KERNEL_STATUS_INNER_ERROR;
        }
    }
    return KERNEL_STATUS_OK;
}

uint32_t CopyFracNzToNdTail(const TransArgs& args, const NdFracNzParams& p, int64_t h1h0_head, int64_t dst_h_head)
{
    const int64_t w1_head = p.num_w1 * p.w0;
    for (int64_t w0_idx = 0; w1_head + w0_idx < p.w; w0_idx++) {
        const int64_t dst_w_idx = w1_head + w0_idx;
        const int64_t src_offset = (h1h0_head + p.num_w1 * p.h1h0w0 + w0_idx) * p.size;
        const int64_t dst_offset = (dst_h_head + dst_w_idx) * p.size;
        const int64_t protected_size = p.dst_size - dst_offset < static_cast<int64_t>(SECUREC_MEM_MAX_LEN) ?
                                           p.dst_size - dst_offset :
                                           static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
        const errno_t ret = memcpy_s(args.output + dst_offset, static_cast<size_t>(protected_size),
                                     args.data + src_offset, static_cast<size_t>(p.size));
        if (ret != EOK) {
            KERNEL_LOG_ERROR("Failed to operate the dst memory at offset [%ld], error-code "
                             "[%d]",
                             dst_offset, ret);
            return KERNEL_STATUS_INNER_ERROR;
        }
    }
    return KERNEL_STATUS_OK;
}

uint32_t TransFormatFromFracNzToNd(const TransArgs& args, const ShapeVector& dst_hw_shape)
{
    NdFracNzParams p;
    p.size = GetSizeByDataType(args.src_data_type);
    p.dst_size = GetItemNumByShape(args.dst_shape) * p.size;
    if (p.dst_size == 0) {
        return KERNEL_STATUS_OK;
    }

    p.times = dst_hw_shape.at(kNdDimIndexN);
    const int64_t h = dst_hw_shape.at(kNdDimIndexH);
    p.w = dst_hw_shape.at(kNdDimIndexW);
    p.hw = h * p.w;

    const auto shape_size = args.src_shape.size();
    const int64_t w1 = args.src_shape[shape_size - kFNzDimCountBackwardsW0H0H1W1];
    const int64_t h1 = args.src_shape[shape_size - kFNzDimCountBackwardsW0H0H1];
    const int64_t h0 = args.src_shape[shape_size - kFNzDimCountBackwardsW0H0];
    p.w0 = args.src_shape[shape_size - kFNzDimCountBackwardsW0];
    const int64_t h1h0 = h1 * h0;
    p.h1h0w0 = h1h0 * p.w0;
    p.w1h1h0w0 = w1 * p.h1h0w0;
    p.num_w1 = p.w / p.w0;

    for (int64_t times_idx = 0; times_idx < p.times; times_idx++) {
        const int64_t times_head = times_idx * p.w1h1h0w0;
        const int64_t dst_times_head = times_idx * p.hw;
        for (int64_t h1h0_idx = 0; h1h0_idx < h; h1h0_idx++) {
            const int64_t h1h0_head = times_head + h1h0_idx * p.w0;
            const int64_t dst_h_head = dst_times_head + h1h0_idx * p.w;
            uint32_t ret = CopyFracNzToNdBlock(args, p, h1h0_head, dst_h_head);
            if (ret != KERNEL_STATUS_OK) {
                return ret;
            }
            ret = CopyFracNzToNdTail(args, p, h1h0_head, dst_h_head);
            if (ret != KERNEL_STATUS_OK) {
                return ret;
            }
        }
    }
    return KERNEL_STATUS_OK;
}

uint32_t ValidateFractalNzArgs(const TransArgs& args, bool check_dst_format)
{
    if (!IsDataTypeSupport(args.src_data_type)) {
        KERNEL_LOG_ERROR("Trans format from [%s] to [%s], src shape [%s], dst shape [%s], data "
                         "type [%s] is not supported",
                         FormatToSerialString(args.src_format).c_str(), FormatToSerialString(args.dst_format).c_str(),
                         VectorToString(args.src_shape).c_str(), VectorToString(args.dst_shape).c_str(),
                         DTypeStr(args.src_data_type).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    const bool shape_ok = check_dst_format ?
                              (IsShapeValid(args.src_shape) && CheckShape(args.dst_format, args.dst_shape)) :
                              (CheckShape(args.src_format, args.src_shape) && IsShapeValid(args.dst_shape));
    if (!shape_ok) {
        KERNEL_LOG_ERROR("Trans format from [%s] to [%s], src shape [%s], dst shape [%s], data "
                         "type [%s] is not supported",
                         FormatToSerialString(args.src_format).c_str(), FormatToSerialString(args.dst_format).c_str(),
                         VectorToString(args.src_shape).c_str(), VectorToString(args.dst_shape).c_str(),
                         DTypeStr(args.src_data_type).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}
} // namespace

uint32_t FormatTransferFractalNz::TransFormat(const TransArgs& args)
{
    const uint32_t validate_ret = ValidateFractalNzArgs(args, false);
    if (validate_ret != KERNEL_STATUS_OK) {
        return validate_ret;
    }
    KERNEL_LOG_INFO("Begin to trans format from [%s] to [%s], src shape [%s], dst shape "
                    "[%s], data type [%s]",
                    FormatToSerialString(args.src_format).c_str(), FormatToSerialString(args.dst_format).c_str(),
                    VectorToString(args.src_shape).c_str(), VectorToString(args.dst_shape).c_str(),
                    DTypeStr(args.src_data_type).c_str());
    ShapeVector expect_shape;
    ShapeVector hw_shape;
    TransArgs trans_args = args;
    const int64_t w0 = GetC0ValueForTransFormat(trans_args.src_data_type, trans_args.input_format,
                                                trans_args.output_format);
    if (g_shape_extension_fun_map.find(trans_args.src_data_type) != g_shape_extension_fun_map.end()) {
        KERNEL_LOG_DEBUG("Begin exec shape extension function, data type [%s]",
                         DTypeStr(trans_args.src_data_type).c_str());
        KERNEL_HANDLE_ERROR(g_shape_extension_fun_map[trans_args.src_data_type](trans_args),
                            "Shape extension failed, data type [%s]", DTypeStr(trans_args.src_data_type).c_str());
    }
    auto ret = TransShapeToFracNz(trans_args.src_shape, trans_args.dst_shape, w0, expect_shape, hw_shape);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    if (!IsTransShapeDstCorrect(trans_args, expect_shape)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return TransFormatFromNdToFracNz(trans_args, hw_shape);
}

uint32_t FormatTransferFractalNz::TransShape(const TransArgs& args, std::vector<int64_t>& dst_shape, bool reverse)
{
    (void)dst_shape;
    (void)reverse;
    const DataType data_type = args.src_data_type;
    const Format src_format = args.src_format;
    const Format dst_format = args.dst_format;
    const std::vector<int64_t> src_shape = args.src_shape;
    KERNEL_LOG_ERROR("Trans format from [%s] to [%s], src shape [%s], data type [%s] is not "
                     "supported, because h0/w0 is not unique, replace it with FE's TransferShape(...)",
                     FormatToSerialString(src_format).c_str(), FormatToSerialString(dst_format).c_str(),
                     VectorToString(src_shape).c_str(), DTypeStr(data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
}

uint32_t FormatTransferFractalNzND::TransFormat(const TransArgs& args)
{
    const uint32_t validate_ret = ValidateFractalNzArgs(args, true);
    if (validate_ret != KERNEL_STATUS_OK) {
        return validate_ret;
    }
    KERNEL_LOG_INFO("Begin to trans format from [%s] to [%s], src shape [%s], dst shape "
                    "[%s], data type [%s]",
                    FormatToSerialString(args.src_format).c_str(), FormatToSerialString(args.dst_format).c_str(),
                    VectorToString(args.src_shape).c_str(), VectorToString(args.dst_shape).c_str(),
                    DTypeStr(args.src_data_type).c_str());

    ShapeVector hw_shape;
    auto ret = CheckShapeRelation(args, hw_shape);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    return TransFormatFromFracNzToNd(args, hw_shape);
}

uint32_t FormatTransferFractalNzND::TransShape(const TransArgs& args, std::vector<int64_t>& dst_shape, bool reverse)
{
    (void)dst_shape;
    (void)reverse;
    KERNEL_LOG_ERROR("The shape derivation from [%s] to [%s] is not unique. Trans shape is "
                     "not supported",
                     FormatToSerialString(args.src_format).c_str(), FormatToSerialString(args.dst_format).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
}

REGISTER_FORMAT_TRANSFER(FormatTransferFractalNz, FORMAT_ND, FORMAT_FRACTAL_NZ)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalNz, FORMAT_ND, FORMAT_FRACTAL_NZ_C0_16)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalNz, FORMAT_ND, FORMAT_FRACTAL_NZ_C0_32)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalNz, FORMAT_NCHW, FORMAT_FRACTAL_NZ)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalNz, FORMAT_NHWC, FORMAT_FRACTAL_NZ)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalNzND, FORMAT_FRACTAL_NZ, FORMAT_ND)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalNzND, FORMAT_FRACTAL_NZ, FORMAT_NCHW)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalNzND, FORMAT_FRACTAL_NZ, FORMAT_NHWC)
} // namespace formats
} // namespace aicpu
