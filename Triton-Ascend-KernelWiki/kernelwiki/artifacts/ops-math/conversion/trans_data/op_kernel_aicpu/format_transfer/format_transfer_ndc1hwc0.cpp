/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "format_transfer_ndc1hwc0.h"

#include "format_transfer_utils.h"
#include "formats_definitions.h"
#include "kernel_util.h"
#include "log.h"
#include "securec.h"
#include "status.h"

namespace aicpu {
namespace formats {
namespace {
std::map<Format, std::string> kFormatTable = {
    {FORMAT_NCDHW, "NCDHW"},
    {FORMAT_NDHWC, "NDHWC"},
};

KernelStatus CheckDataTypeSupport(DataType data_type)
{
    return GetSizeByDataType(data_type) > 0 ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

void CopyCLineToNdc1hwc0(const TransArgs& args, uint8_t* dst, int32_t data_size, int64_t c, int64_t c0_div,
                         int64_t src_base, int64_t src_stride_per_c, int64_t dst_base, int64_t hwc0)
{
    if (c0_div == 0) {
        KERNEL_LOG_ERROR("c0_div must not be 0.");
        return;
    }
    for (int64_t c_idx = 0; c_idx < c; ++c_idx) {
        const int64_t src_index = src_base + c_idx * src_stride_per_c;
        const int64_t div_result = (c0_div == 0) ? 0 : (c_idx / c0_div);
        const int64_t mod_result = (c0_div == 0) ? 0 : (c_idx % c0_div);
        const int64_t dst_index = dst_base + div_result * hwc0 + mod_result;
        uint8_t* dst_data = dst + dst_index * data_size;
        const uint8_t* src_data = args.data + src_index * data_size;
        for (int64_t index = 0; index < data_size; ++index) {
            *dst_data++ = *src_data++;
        }
    }
}

void TransSrcDataToDstData(const TransArgs& args, const std::vector<int64_t>& shape_ndhwc, uint8_t* dst, int64_t c0,
                           int32_t data_size)
{
    if (c0 <= 0) {
        KERNEL_LOG_ERROR("C0 must be greater than 0, now is [%ld]", c0);
        return;
    }
    const int64_t n = shape_ndhwc[0];
    const int64_t d = shape_ndhwc[1];
    const int64_t h = shape_ndhwc[2];
    const int64_t w = shape_ndhwc[3];
    const int64_t c = shape_ndhwc[4];
    const int64_t c0_div = c0;
    const int64_t c1 = ((c - 1) / c0_div) + 1;
    const int64_t hw = h * w;
    const int64_t hwc0 = hw * c0_div;
    const int64_t c1hwc0 = c1 * hwc0;
    const int64_t dc1hwc0 = d * c1hwc0;
    const int64_t dhw = d * hw;
    const int64_t dhwc = dhw * c;
    const int64_t hwc = hw * c;
    const int64_t wc = w * c;
    const bool is_ncdhw = (args.src_format == FORMAT_NCDHW);
    // NCDHW [N][C][D][H][W]: advancing c by 1 skips D*H*W elements.
    // NDHWC [N][D][H][W][C]: advancing c by 1 skips 1 element.
    const int64_t src_stride_per_c = is_ncdhw ? dhw : 1;

    for (int64_t n_idx = 0; n_idx < n; ++n_idx) {
        for (int64_t d_idx = 0; d_idx < d; ++d_idx) {
            for (int64_t h_idx = 0; h_idx < h; ++h_idx) {
                for (int64_t w_idx = 0; w_idx < w; ++w_idx) {
                    const int64_t src_base = is_ncdhw ? (n_idx * dhwc + d_idx * hw + h_idx * w + w_idx) :
                                                        (n_idx * dhwc + d_idx * hwc + h_idx * wc + w_idx * c);
                    const int64_t dst_base = n_idx * dc1hwc0 + d_idx * c1hwc0 + h_idx * (w * c0_div) + w_idx * c0_div;
                    CopyCLineToNdc1hwc0(args, dst, data_size, c, c0_div, src_base, src_stride_per_c, dst_base, hwc0);
                }
            }
        }
    }
}

uint32_t TransDstDataToNdc1hwc0(const TransArgs& args)
{
    const DataType data_type = args.src_data_type;
    const int32_t data_size = GetSizeByDataType(data_type);
    const auto dst_size = GetItemNumByShape(args.dst_shape) * data_size;
    // The input is empty tensor, we should return sucess directly
    if (dst_size == 0) {
        return KERNEL_STATUS_OK;
    }
    auto ret = BiggerMemSet(args.output, static_cast<size_t>(dst_size), 0, static_cast<size_t>(dst_size));
    if (!ret) {
        KERNEL_LOG_ERROR("memset failed.");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    auto iter = kFormatTable.find(args.src_format);
    if (iter == kFormatTable.end()) {
        KERNEL_LOG_ERROR("src_format is wrong, now format is [%d]", static_cast<int32_t>(args.src_format));
        return KERNEL_STATUS_PARAM_INVALID;
    }

    std::string cur_format = iter->second;
    size_t n_index = cur_format.find('N');
    size_t d_index = cur_format.find('D');
    size_t h_index = cur_format.find('H');
    size_t w_index = cur_format.find('W');
    size_t c_index = cur_format.find('C');
    std::vector<int64_t> shape_ndhwc;
    shape_ndhwc.push_back(args.src_shape.at(n_index));
    shape_ndhwc.push_back(args.src_shape.at(d_index));
    shape_ndhwc.push_back(args.src_shape.at(h_index));
    shape_ndhwc.push_back(args.src_shape.at(w_index));
    shape_ndhwc.push_back(args.src_shape.at(c_index));
    const int64_t c0 = GetC0ValueForTransFormat(data_type, args.input_format, args.output_format);
    if (c0 <= 0) {
        KERNEL_LOG_ERROR("Failed to get c0, c0 is [%ld]", c0);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    TransSrcDataToDstData(args, shape_ndhwc, args.output, c0, data_size);
    return KERNEL_STATUS_OK;
}

uint32_t TransShapeToNdc1hwc0(const std::vector<int64_t>& src_shape, const Format& src_format,
                              const DataType& data_type, std::vector<int64_t>& dst_shape, const int32_t format)
{
    auto iter = kFormatTable.find(src_format);
    if (iter == kFormatTable.end()) {
        KERNEL_LOG_ERROR("src_format is wrong, now format is [%d]", static_cast<int32_t>(src_format));
        return KERNEL_STATUS_PARAM_INVALID;
    }

    std::string cur_format = iter->second;
    size_t n_index = cur_format.find('N');
    size_t d_index = cur_format.find('D');
    size_t h_index = cur_format.find('H');
    size_t w_index = cur_format.find('W');
    size_t c_index = cur_format.find('C');
    const int64_t c0 = GetC0ValueForTransShape(data_type, format);
    if (c0 <= 0) {
        KERNEL_LOG_ERROR("Failed to get c0, c0 is [%ld]", c0);
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (!CheckShapeValid(src_shape, static_cast<int64_t>(cur_format.length()))) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    dst_shape.clear();
    dst_shape.push_back(src_shape.at(n_index));
    dst_shape.push_back(src_shape.at(d_index));
    dst_shape.push_back(Ceil(src_shape.at(c_index), c0));
    dst_shape.push_back(src_shape.at(h_index));
    dst_shape.push_back(src_shape.at(w_index));
    dst_shape.push_back(c0);
    if (!IsShapeValid(dst_shape)) {
        KERNEL_LOG_ERROR("Check shape failed, dst shape [%s]", VectorToString(dst_shape).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    return KERNEL_STATUS_OK;
}
} // namespace

uint32_t FormatTransferNdc1hwc0::TransFormat(const TransArgs& args)
{
    KERNEL_LOG_INFO("Begin to trans format from [%s] to [%s], src shape [%s], data type [%s], dst "
                    "shape [%s]",
                    FormatToSerialString(args.src_format).c_str(), FormatToSerialString(args.dst_format).c_str(),
                    VectorToString(args.src_shape).c_str(), DTypeStr(args.src_data_type).c_str(),
                    VectorToString(args.dst_shape).c_str());

    std::vector<int64_t> expect_shape;
    auto ret = TransShape(args, expect_shape, false);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    if (!IsTransShapeDstCorrect(args, expect_shape)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }

    return TransDstDataToNdc1hwc0(args);
}

uint32_t FormatTransferNdc1hwc0::TransShape(const TransArgs& args, std::vector<int64_t>& dst_shape, bool reverse)
{
    (void)reverse;
    const Format src_format = args.src_format;
    const DataType data_type = args.src_data_type;
    if (CheckDataTypeSupport(data_type) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (src_format != FORMAT_NCDHW && src_format != FORMAT_NDHWC) {
        KERNEL_LOG_ERROR("The current format is not supported, src_format is [%s]",
                         FormatToSerialString(src_format).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    return TransShapeToNdc1hwc0(args.src_shape, src_format, data_type, dst_shape, args.output_format);
}
REGISTER_FORMAT_TRANSFER(FormatTransferNdc1hwc0, FORMAT_NCDHW, FORMAT_NDC1HWC0)
REGISTER_FORMAT_TRANSFER(FormatTransferNdc1hwc0, FORMAT_NDHWC, FORMAT_NDC1HWC0)
} // namespace formats
} // namespace  aicpu
