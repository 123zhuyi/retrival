/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "trans_data_aicpu.h"

#include <algorithm>

#include "Eigen/Core"
#include "cpu_types.h"
#include "format_transfer/format_transfer_utils.h"
#include "format_transfer/formats_definitions.h"
#include "log.h"
#include "securec.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/kernel_util.h"

using namespace std;

namespace {
const char* const kTransData = "TransData";
constexpr int64_t kDimN0 = 16;
constexpr int64_t kCubeN = 16;
constexpr int64_t kGroupNum = 1;
constexpr int64_t kMaxDimsNumC = 4;
constexpr size_t kMinInputDimsNum = 4;
constexpr int32_t kCubeSize = 16;
constexpr int32_t kShapeOffset = 2;

using aicpu::DTypeStr;
using aicpu::FormatToSerialString;
using aicpu::GetPrimaryFormat;
using aicpu::GetSizeByDataType;
using aicpu::KERNEL_STATUS_INNER_ERROR;
using aicpu::KERNEL_STATUS_OK;
using aicpu::KERNEL_STATUS_PARAM_INVALID;
using aicpu::formats::Lcm;

template <typename T>
std::string VectorToString(const std::vector<T>& vec)
{
    std::stringstream ss;
    bool first = true;
    for (auto& ele : vec) {
        if (first) {
            first = false;
        } else {
            ss << ",";
        }
        ss << ele;
    }
    return ss.str();
}

int64_t VectorToNum(const std::vector<int64_t>& vec)
{
    int64_t result = 1;
    for (auto& ele : vec) {
        result *= ele;
    }
    return result;
}

void TransShapeByPerm(const std::vector<int64_t>& src_shape, const std::vector<int64_t>& perm_arg,
                      std::vector<int64_t>& dst_shape)
{
    dst_shape.resize(src_shape.size());
    for (size_t i = 0; i < perm_arg.size(); ++i) {
        dst_shape[i] = src_shape[perm_arg[i]];
    }
}

void GetIndexMap(const std::vector<int64_t>& perm_arg, std::map<int32_t, int32_t>& index_map)
{
    for (size_t i = 0; i < perm_arg.size(); i++) {
        index_map[perm_arg[i]] = static_cast<int32_t>(i);
    }
}

void GetShapeHead(const std::vector<int64_t>& shape, std::vector<int64_t>& shape_head)
{
    shape_head.resize(shape.size());
    shape_head[shape.size() - 1] = 1;
    for (int i = static_cast<int>(shape.size() - kShapeOffset); i >= 0; i--) {
        shape_head[i] = shape_head[i + 1] * shape[i + 1];
    }
}

int32_t GetSrcIndex(int64_t dst_index, const std::vector<int64_t>& src_shape, const std::vector<int64_t>& dst_shape,
                    const std::vector<int64_t>& src_shape_head, const std::vector<int64_t>& dst_shape_head,
                    std::map<int32_t, int32_t> index_map)
{
    std::vector<int32_t> src_vec(dst_shape.size());
    for (size_t i = 0; i < dst_shape.size(); i++) {
        src_vec[i] = dst_index / dst_shape_head[i];
        dst_index = dst_index % dst_shape_head[i];
    }
    int32_t src_index = 0;
    for (size_t i = 0; i < src_shape.size(); i++) {
        src_index += static_cast<int32_t>(src_shape_head[i] * src_vec[index_map[i]]);
    }
    return src_index;
}

// get the result of two number divisor and let result round up
static int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return -1;
    } else {
        int64_t ret = a / b;
        if ((a % b) != 0) {
            ret++;
        }
        return ret;
    }
}

struct TransDataTensorInfo {
    uint8_t* data;
    aicpu::DataType data_type;
    std::vector<int64_t> dims;
    aicpu::Format format;
};

uint32_t ExtractTensorInfo(aicpu::Tensor* tensor, TransDataTensorInfo& info, const char* who)
{
    KERNEL_CHECK_NULLPTR(tensor, KERNEL_STATUS_PARAM_INVALID, "%s get tensor failed, tensor is nullptr.", who);
    info.data = reinterpret_cast<uint8_t*>(tensor->GetData());
    info.data_type = tensor->GetDataType();
    auto shape = tensor->GetTensorShape();
    KERNEL_CHECK_NULLPTR(shape, KERNEL_STATUS_PARAM_INVALID, "%s get shape failed, shape is nullptr.", who);
    info.dims = shape->GetDimSizes();
    info.format = shape->GetFormat();
    return KERNEL_STATUS_OK;
}

struct DealDataCtx {
    aicpu::Format input_format;
    int64_t d_dim;
    int64_t h_dim;
    int64_t w_dim;
    int64_t c_dim;
    int64_t n_dim;
    int64_t cin_ori;
    int64_t cout_ori;
    int64_t cube_k;
    int64_t e_mult;
    int64_t cout_opt;
    int64_t c1_dim;
    int64_t size_output_data;
};

uint32_t BuildDealDataCtx(const aicpu::Tensor* input_tensor, int64_t group, int64_t cube_k, DealDataCtx& c)
{
    if (group == 0) {
        KERNEL_LOG_ERROR("Group must not be equal to 0");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (cube_k == 0) {
        KERNEL_LOG_ERROR("Cube_k must not be equal to 0, data type [%s]",
                         DTypeStr(static_cast<aicpu::DataType>(input_tensor->GetDataType())).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    auto input_shape = input_tensor->GetTensorShape();
    const aicpu::Format ge_input_format = input_shape->GetFormat();
    c.input_format = static_cast<aicpu::Format>(GetPrimaryFormat(static_cast<int32_t>(ge_input_format)));
    std::vector<int64_t> dims = input_shape->GetDimSizes();
    c.d_dim = 0;
    c.h_dim = 0;
    c.w_dim = 0;
    c.c_dim = 0;
    c.n_dim = 0;
    if (aicpu::formats::GetFormatDim(c.d_dim, c.h_dim, c.w_dim, c.c_dim, c.n_dim, c.input_format, dims) !=
        KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    c.cin_ori = c.c_dim;
    c.cout_ori = c.n_dim / group;
    if (aicpu::formats::CheckDimOri(c.cin_ori, c.cout_ori) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    c.cube_k = cube_k;
    c.e_mult = std::min(Lcm(Lcm(c.cin_ori, cube_k) / (c.cin_ori), Lcm(c.cout_ori, kCubeN) / (c.cout_ori)), group);
    if (c.e_mult == 0) {
        KERNEL_LOG_ERROR("E_mult must not be equal to 0");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    const int64_t cin_opt = Ceil(c.e_mult * c.cin_ori, cube_k) * cube_k;
    c.cout_opt = Ceil(c.e_mult * c.cout_ori, kCubeN) * kCubeN;
    c.c1_dim = cin_opt / cube_k;
    const int64_t g_dim = Ceil(group, c.e_mult);
    c.size_output_data = g_dim * c.d_dim * c.c1_dim * c.h_dim * c.w_dim * c.cout_opt * cube_k;
    return KERNEL_STATUS_OK;
}

inline int64_t ComputeSrcIndexForDealData(const DealDataCtx& c, int64_t d, int64_t h, int64_t w, int64_t co,
                                          int64_t src_co)
{
    if ((c.input_format == aicpu::FORMAT_DHWCN) || (c.input_format == aicpu::FORMAT_HWCN)) {
        return d * c.h_dim * c.w_dim * c.c_dim * c.n_dim + h * c.w_dim * c.c_dim * c.n_dim + w * c.c_dim * c.n_dim +
               co * c.n_dim + src_co;
    }
    if ((c.input_format == aicpu::FORMAT_NCDHW) || (c.input_format == aicpu::FORMAT_NCHW)) {
        return src_co * c.c_dim * c.d_dim * c.h_dim * c.w_dim + co * c.d_dim * c.h_dim * c.w_dim +
               d * c.h_dim * c.w_dim + h * c.w_dim + w;
    }
    if ((c.input_format == aicpu::FORMAT_NDHWC) || (c.input_format == aicpu::FORMAT_NHWC)) {
        return src_co * c.d_dim * c.h_dim * c.w_dim * c.c_dim + d * c.h_dim * c.w_dim * c.c_dim +
               h * c.w_dim * c.c_dim + w * c.c_dim + co;
    }
    return 0;
}

template <typename T>
void DealDataInner(const T* input_data, T* output_data, const DealDataCtx& c, int64_t g, int64_t d, int64_t co,
                   int64_t h, int64_t w)
{
    for (int64_t n = 0; n < c.cout_ori; n++) {
        const int64_t e_val = g % c.e_mult;
        const int64_t dst_ci = e_val * c.cin_ori + co;
        const int64_t dst_co = e_val * c.cout_ori + n;
        const int64_t src_co = g * c.cout_ori + n;
        const int64_t tempory = dst_ci % c.cube_k;
        const int64_t dst_inx = (g / c.e_mult) * c.d_dim * c.c1_dim * c.h_dim * c.w_dim * c.cout_opt * c.cube_k +
                                d * c.c1_dim * c.h_dim * c.w_dim * c.cout_opt * c.cube_k +
                                (dst_ci / c.cube_k) * c.h_dim * c.w_dim * c.cout_opt * c.cube_k +
                                h * c.w_dim * c.cout_opt * c.cube_k + w * c.cout_opt * c.cube_k + dst_co * c.cube_k +
                                tempory;
        const int64_t srx_inx = ComputeSrcIndexForDealData(c, d, h, w, co, src_co);
        output_data[dst_inx] = input_data[srx_inx];
    }
}

template <typename T>
void DealDataForHW(const T* input_data, T* output_data, const DealDataCtx& c, int64_t g, int64_t d, int64_t co)
{
    for (int64_t h = 0; h < c.h_dim; h++) {
        for (int64_t w = 0; w < c.w_dim; w++) {
            DealDataInner<T>(input_data, output_data, c, g, d, co, h, w);
        }
    }
}

constexpr int64_t kMaxPaddingBufferBytes = 1LL << 32; // 4 GiB upper bound for padding buffer

uint32_t AllocatePaddingBuffer(int64_t dst_byte_size, std::shared_ptr<uint8_t>& dst)
{
    if (dst_byte_size <= 0 || dst_byte_size > kMaxPaddingBufferBytes) {
        KERNEL_LOG_ERROR("Invalid padding buffer size [%ld], expect (0, %ld]", dst_byte_size, kMaxPaddingBufferBytes);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    dst.reset(new (std::nothrow) uint8_t[dst_byte_size], std::default_delete<uint8_t[]>());
    if (dst == nullptr) {
        KERNEL_LOG_ERROR("New Memory failed!");
        return KERNEL_STATUS_INNER_ERROR;
    }
    const errno_t ret_mem = memset_s(dst.get(), dst_byte_size, 0, dst_byte_size);
    if (ret_mem != 0) {
        KERNEL_LOG_ERROR("Memset failed, ret is [%d]", ret_mem);
        return KERNEL_STATUS_INNER_ERROR;
    }
    return KERNEL_STATUS_OK;
}
void LogTransBegin(const TransDataTensorInfo& in, const TransDataTensorInfo& out, int64_t group)
{
    KERNEL_LOG_DEBUG("Begin trans formats from [%s] to [%s], shape [%s] to [%s], data type "
                     "[%s] to [%s], group is [%ld]",
                     FormatToSerialString(in.format).c_str(), FormatToSerialString(out.format).c_str(),
                     VectorToString(in.dims).c_str(), VectorToString(out.dims).c_str(), DTypeStr(in.data_type).c_str(),
                     DTypeStr(out.data_type).c_str(), group);
}

void LogTransEnd(const TransDataTensorInfo& in, const TransDataTensorInfo& out)
{
    KERNEL_LOG_DEBUG("End trans formats from [%s] to [%s], shape [%s] to [%s], data type "
                     "[%s] to [%s]",
                     FormatToSerialString(in.format).c_str(), FormatToSerialString(out.format).c_str(),
                     VectorToString(in.dims).c_str(), VectorToString(out.dims).c_str(), DTypeStr(in.data_type).c_str(),
                     DTypeStr(out.data_type).c_str());
}

void LogTransUnsupported(const TransDataTensorInfo& in, const TransDataTensorInfo& out)
{
    KERNEL_LOG_WARN("Transfer from format[%s] to [%s], shape [%s] to [%s], data type [%s] "
                    "to [%s] is not supported",
                    FormatToSerialString(in.format).c_str(), FormatToSerialString(out.format).c_str(),
                    VectorToString(in.dims).c_str(), VectorToString(out.dims).c_str(), DTypeStr(in.data_type).c_str(),
                    DTypeStr(out.data_type).c_str());
}

void LogTransFailed(const TransDataTensorInfo& in, const TransDataTensorInfo& out)
{
    KERNEL_LOG_WARN("Failed to trans formats from[%s] to [%s], shape [%s] to [%s], data "
                    "type [%s]",
                    FormatToSerialString(in.format).c_str(), FormatToSerialString(out.format).c_str(),
                    VectorToString(in.dims).c_str(), VectorToString(out.dims).c_str(), DTypeStr(in.data_type).c_str());
}

aicpu::formats::TransArgs BuildTransArgs(const TransDataTensorInfo& in, const TransDataTensorInfo& out, int64_t group,
                                         const aicpu::CpuKernelContext* ctx)
{
    return aicpu::formats::TransArgs{in.data,
                                     out.data,
                                     static_cast<int32_t>(in.format),
                                     static_cast<int32_t>(out.format),
                                     static_cast<aicpu::Format>(GetPrimaryFormat(static_cast<int32_t>(in.format))),
                                     static_cast<aicpu::Format>(GetPrimaryFormat(static_cast<int32_t>(out.format))),
                                     in.dims,
                                     out.dims,
                                     in.data_type,
                                     group,
                                     ctx};
}

uint32_t RunFormatTransfer(const TransDataTensorInfo& in, const TransDataTensorInfo& out, int64_t group,
                           const aicpu::CpuKernelContext* ctx)
{
    LogTransBegin(in, out, group);
    const aicpu::formats::TransArgs trans_args = BuildTransArgs(in, out, group, ctx);
    if (in.data_type != out.data_type || in.dims.empty() || !aicpu::formats::FormatTransferExists(trans_args)) {
        LogTransUnsupported(in, out);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    const uint32_t ret = aicpu::formats::TransFormat(trans_args);
    if (ret != KERNEL_STATUS_OK) {
        LogTransFailed(in, out);
        return ret;
    }
    LogTransEnd(in, out);
    return KERNEL_STATUS_OK;
}
} // namespace

namespace aicpu {
bool TransDataCpuKernel::IsOriginSupportFormatTransfer(Format src_format, Format dst_format)
{
    static const map<Format, map<Format, int32_t>> kOriginSupportFormatTransfer = {
        {FORMAT_HWCN, {{FORMAT_FRACTAL_Z_C04, 1}}}};
    auto dst = kOriginSupportFormatTransfer.find(src_format);
    if (dst == kOriginSupportFormatTransfer.end()) {
        return false;
    }
    return dst->second.count(dst_format) > 0;
}

uint32_t TransDataCpuKernel::NewCompute(const CpuKernelContext& ctx)
{
    TransDataTensorInfo input_info;
    TransDataTensorInfo output_info;
    if (ExtractTensorInfo(ctx.Input(0), input_info, kTransData) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (ExtractTensorInfo(ctx.Output(0), output_info, kTransData) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    KERNEL_CHECK_NULLPTR(output_info.data, KERNEL_STATUS_PARAM_INVALID,
                         "%s get output_data failed, output_data is nullptr.", kTransData);
    int64_t group = kGroupNum;
    AttrValue* groups = ctx.GetAttr("groups");
    if (groups != nullptr) {
        group = groups->GetInt();
    }
    return RunFormatTransfer(input_info, output_info, group, &ctx);
}

template <typename T>
uint32_t TransDataCpuKernel::DealData(const T* input_data, T* output_data, const Tensor* input_tensor,
                                      Tensor* output_tensor, int64_t group)
{
    (void)output_tensor;
    DealDataCtx c{};
    if (BuildDealDataCtx(input_tensor, group, GetCubeSizeByDataType(static_cast<DataType>(input_tensor->GetDataType())),
                         c) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    const errno_t ret_mem = memset_s(output_data, static_cast<int64_t>(sizeof(T)) * c.size_output_data, 0,
                                     static_cast<int64_t>(sizeof(T)) * c.size_output_data);
    if (ret_mem != EOK) {
        KERNEL_LOG_ERROR("memset_s failed, ret [%d].", ret_mem);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    for (int64_t g = 0; g < group; g++) {
        for (int64_t d = 0; d < c.d_dim; d++) {
            for (int64_t co = 0; co < c.c_dim; co++) {
                DealDataForHW<T>(input_data, output_data, c, g, d, co);
            }
        }
    }
    return KERNEL_STATUS_OK;
}

// TransData supports input formats (NCDHW, DHWCN, NDHWC) convert to
// FORMAT_FRACTAL_Z_3D (GDC1HWN1N0C0), and also supports NHWC, NCHW, HWCN
// converte to FORMAT_FRACTAL_Z (GC1HWN1N0C0), HWCN to FZC04. The final effect
// achieved is for the data to be distributed diagonally. For example: When the
// input filter format is NCDHW, calculated the Correspondence of index between
// NCDHW and FORMAT_FRACTAL_Z_3D , then Convert the old filter to the new
// filter, and finally added 0 to the position where there is no data.
uint32_t TransDataCpuKernel::HandleHwcnToFzC04(const Tensor* input_tensor, Tensor* output_tensor)
{
    KERNEL_LOG_DEBUG("Begin trans formats from FORMAT_HWCN to FORMAT_FRACTAL_Z_C04");
    const DataType data_type = static_cast<DataType>(input_tensor->GetDataType());
    const int64_t cube = GetCubeSizeByDataType(data_type);
    if (cube < 0) {
        KERNEL_LOG_WARN("Don't support dtype[%s]", DTypeStr(data_type).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    const uint64_t data_type_size = output_tensor->GetDataSize();
    const uint64_t data_byte_size = GetSizeByDataType(data_type) * data_type_size;
    TransArgs args = {reinterpret_cast<uint8_t*>(input_tensor->GetData()),
                      input_tensor->GetTensorShape()->GetDimSizes(), output_tensor->GetTensorShape()->GetDimSizes(),
                      data_type};
    auto output_addr = reinterpret_cast<uint8_t*>(output_tensor->GetData());
    const int64_t c0_cube = formats::GetC0ValueForTransShape(
        args.src_data_type, static_cast<int32_t>(output_tensor->GetTensorShape()->GetFormat()));
    KERNEL_CHECK_FALSE((c0_cube > 0), KERNEL_STATUS_PARAM_INVALID, "c0_cube must greater than 0, now is [%ld].",
                       c0_cube);
    const uint32_t ret = FormatTransferHwcnToFZC04(args, output_addr, data_byte_size, c0_cube);
    if (ret != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("FormatTransferHwcnToFZC04 function failed");
        return ret;
    }
    KERNEL_LOG_DEBUG("Finish trans formats from FORMAT_HWCN to FORMAT_FRACTAL_Z_C04");
    return KERNEL_STATUS_OK;
}

uint32_t TransDataCpuKernel::DispatchDealData(DataType dt, void* input_data_temp, void* output_data_temp,
                                              const Tensor* input_tensor, Tensor* output_tensor, int64_t group)
{
    switch (dt) {
        case DT_INT8:
            return DealData(reinterpret_cast<int8_t*>(input_data_temp), reinterpret_cast<int8_t*>(output_data_temp),
                            input_tensor, output_tensor, group);
        case DT_FLOAT:
            return DealData(reinterpret_cast<float*>(input_data_temp), reinterpret_cast<float*>(output_data_temp),
                            input_tensor, output_tensor, group);
        case DT_FLOAT16:
            return DealData(reinterpret_cast<Eigen::half*>(input_data_temp),
                            reinterpret_cast<Eigen::half*>(output_data_temp), input_tensor, output_tensor, group);
        default:
            KERNEL_LOG_WARN("DateType is not DT_INT8 or DT_FLOAT or DT_FLOAT16, and current "
                            "DataType is [%d]",
                            static_cast<int32_t>(dt));
            return KERNEL_STATUS_PARAM_INVALID;
    }
}

uint32_t TransDataCpuKernel::Compute(CpuKernelContext& ctx)
{
    Tensor* input_tensor = ctx.Input(0);
    KERNEL_CHECK_NULLPTR(input_tensor, KERNEL_STATUS_PARAM_INVALID, "%s get input_tensor failed", kTransData);
    const auto input_format = GetPrimaryFormat(static_cast<int32_t>(input_tensor->GetTensorShape()->GetFormat()));
    Tensor* output_tensor = ctx.Output(0);
    KERNEL_CHECK_NULLPTR(output_tensor, KERNEL_STATUS_PARAM_INVALID, "%s get output_tensor failed", kTransData);
    const auto output_format = GetPrimaryFormat(static_cast<int32_t>(output_tensor->GetTensorShape()->GetFormat()));
    if (!IsOriginSupportFormatTransfer(static_cast<Format>(input_format), static_cast<Format>(output_format))) {
        return NewCompute(ctx);
    }
    if ((input_format == FORMAT_HWCN) && (output_format == FORMAT_FRACTAL_Z_C04)) {
        return HandleHwcnToFzC04(input_tensor, output_tensor);
    }
    const int32_t primary_out_put_format = GetPrimaryFormat(static_cast<int32_t>(output_format));
    if ((primary_out_put_format != static_cast<int32_t>(FORMAT_FRACTAL_Z)) &&
        (primary_out_put_format != static_cast<int32_t>(FORMAT_FRACTAL_Z_3D))) {
        KERNEL_LOG_EVENT("%s unsupport output_format [%d]", kTransData, primary_out_put_format);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    auto input_shape = input_tensor->GetTensorShape();
    KERNEL_CHECK_NULLPTR(input_shape, KERNEL_STATUS_PARAM_INVALID, "%s get input_shape failed", kTransData);
    const std::vector<int64_t> dims = input_shape->GetDimSizes();
    if ((dims.size()) < kMinInputDimsNum) {
        KERNEL_LOG_WARN("%s dims size [%zu] must >= 4", kTransData, dims.size());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    AttrValue* groups = ctx.GetAttr("groups");
    int64_t group = kGroupNum;
    if (groups != nullptr) {
        group = groups->GetInt();
        KERNEL_CHECK_FALSE((group != 0L), KERNEL_STATUS_PARAM_INVALID, "groups can't be 0.");
    }
    const DataType dt = static_cast<DataType>(input_tensor->GetDataType());
    auto input_data_temp = input_tensor->GetData();
    KERNEL_CHECK_NULLPTR(input_data_temp, KERNEL_STATUS_PARAM_INVALID, "%s get input_data failed", kTransData);
    auto output_data_temp = output_tensor->GetData();
    KERNEL_CHECK_NULLPTR(output_data_temp, KERNEL_STATUS_PARAM_INVALID, "%s get output_data failed", kTransData);
    return DispatchDealData(dt, input_data_temp, output_data_temp, input_tensor, output_tensor, group);
}

uint32_t TransDataCpuKernel::FormatTransferHwcnToFZC04(TransArgs& args, uint8_t* output_addr, uint64_t length,
                                                       int64_t c0_cube)
{
    KERNEL_LOG_DEBUG("Begin to trans format from HWCN to FZC04, src shape [%s], data type "
                     "[%s], dst shape [%s], c0_cube [%ld]",
                     VectorToString(args.src_shape).c_str(), DTypeStr(args.src_data_type).c_str(),
                     VectorToString(args.dst_shape).c_str(), c0_cube);
    std::shared_ptr<uint8_t> dst_padding_one(nullptr);
    uint32_t ret = PaddingOne(args, dst_padding_one);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    std::vector<int64_t> perm_arg_tmp_one = {3, 0, 1, 2};
    std::shared_ptr<uint8_t> dst_transpose_one(nullptr);
    ret = Transpose(args, perm_arg_tmp_one, dst_transpose_one);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    std::vector<int64_t>& src_shape = args.src_shape;
    std::vector<int64_t> src_shape_tmp = src_shape;
    constexpr size_t new_dims = 2;
    src_shape.resize(new_dims);
    src_shape[0] = src_shape_tmp[formats::kHwcnH];
    src_shape[1] = src_shape_tmp[formats::kHwcnW] * src_shape_tmp[formats::kHwcnC] * src_shape_tmp[formats::kHwcnN];
    std::shared_ptr<uint8_t> dst_padding_two(nullptr);
    ret = PaddingTwo(args, dst_padding_two, c0_cube);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }

    KERNEL_CHECK_FALSE((c0_cube > 0), KERNEL_STATUS_PARAM_INVALID, "c0_cube must greater than 0, now is [%ld].",
                       c0_cube); // for avoid warning for div 0
    src_shape_tmp = src_shape;
    src_shape.resize(formats::kHwcnDimsNum);
    src_shape[formats::kHwcnH] = src_shape_tmp[0] / kDimN0;
    src_shape[formats::kHwcnW] = kDimN0;
    src_shape[formats::kHwcnC] = src_shape_tmp[1] / c0_cube;
    src_shape[formats::kHwcnN] = c0_cube;
    std::vector<int64_t> perm_arg_tmp_two = {2, 0, 1, 3};
    std::shared_ptr<uint8_t> dst_transpose_two(nullptr);
    ret = Transpose(args, perm_arg_tmp_two, dst_transpose_two);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    auto ret_mem = BiggerMemCpy(output_addr, length, args.data,
                                VectorToNum(args.src_shape) * GetSizeByDataType(args.src_data_type));
    if (!ret_mem) {
        KERNEL_LOG_ERROR("BiggerMemCpy failed");
        return KERNEL_STATUS_INNER_ERROR;
    }
    return KERNEL_STATUS_OK;
}

uint32_t TransDataCpuKernel::PaddingOne(TransArgs& args, std::shared_ptr<uint8_t>& dst)
{
    DataType data_type = args.src_data_type;
    std::vector<int64_t> dst_shape;
    uint32_t ret = GetPaddingOneShape(args, dst_shape);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    int32_t type_size = GetSizeByDataType(data_type);
    int64_t dst_byte_size = VectorToNum(dst_shape) * type_size;
    if (AllocatePaddingBuffer(dst_byte_size, dst) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_INNER_ERROR;
    }
    std::vector<int64_t>& src_shape = args.src_shape;
    auto h = src_shape.at(0);
    auto w = src_shape.at(1);
    auto c = src_shape.at(2);
    auto n = src_shape.at(3);
    auto h_padding = dst_shape[0];
    auto w_padding = dst_shape[1];
    auto c_padding = dst_shape[2];
    auto n_padding = dst_shape[3];
    auto src_add = args.data;
    auto dst_add = dst.get();
    auto protect_size = h_padding * w_padding * c_padding * n_padding * type_size;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            for (int k = 0; k < c; k++) {
                auto dst_stride = ((i * w_padding + j) * c_padding + k) * n_padding * type_size;
                auto ret_cpy = memcpy_s(dst_add + dst_stride, protect_size - dst_stride,
                                        src_add + ((i * w + j) * c + k) * n * type_size, n * type_size);
                if (ret_cpy != 0) {
                    KERNEL_LOG_ERROR("Memcpy failed, ret is [%d]", ret_cpy);
                    return KERNEL_STATUS_INNER_ERROR;
                }
            }
        }
    }
    args.data = dst.get();
    src_shape = dst_shape;
    return KERNEL_STATUS_OK;
}

uint32_t TransDataCpuKernel::PaddingTwo(TransArgs& args, std::shared_ptr<uint8_t>& dst, int64_t c0_cube)
{
    DataType data_type = args.src_data_type;
    std::vector<int64_t> dst_shape;
    uint32_t ret = GetPaddingTwoShape(args, dst_shape, c0_cube);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    int32_t type_size = GetSizeByDataType(data_type);
    int64_t dst_byte_size = VectorToNum(dst_shape) * type_size;
    if (AllocatePaddingBuffer(dst_byte_size, dst) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_INNER_ERROR;
    }
    std::vector<int64_t>& src_shape = args.src_shape;
    auto n = src_shape.at(0);
    auto z = src_shape.at(1);
    auto n_padding = dst_shape[0];
    auto z_padding = dst_shape[1];
    auto src_add = args.data;
    auto dst_add = dst.get();
    auto protect_size = n_padding * z_padding * type_size;
    for (int i = 0; i < n; i++) {
        auto dst_stride = i * z_padding * type_size;
        auto ret_cpy = memcpy_s(dst_add + dst_stride, protect_size - dst_stride, src_add + i * z * type_size,
                                z * type_size);
        if (ret_cpy != 0) {
            KERNEL_LOG_ERROR("Memcpy failed, ret is [%d]", ret_cpy);
            return KERNEL_STATUS_INNER_ERROR;
        }
    }
    args.data = dst.get();
    src_shape = dst_shape;
    return KERNEL_STATUS_OK;
}

uint32_t TransDataCpuKernel::GetPaddingOneShape(const TransArgs& args, std::vector<int64_t>& dst_shape)
{
    int64_t cube = kDimN0;
    auto h = args.src_shape.at(formats::kHwcnH);
    auto w = args.src_shape.at(formats::kHwcnW);
    auto c = args.src_shape.at(formats::kHwcnC);
    auto n = args.src_shape.at(formats::kHwcnN);
    if (c > kMaxDimsNumC) {
        KERNEL_LOG_ERROR("Invalid dim c num[%lu].It should be in (0, %ld]", c, kMaxDimsNumC);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    dst_shape.resize(formats::kHwcnDimsNum);
    dst_shape[formats::kHwcnH] = h;
    dst_shape[formats::kHwcnW] = w;
    dst_shape[formats::kHwcnC] = kMaxDimsNumC;
    int64_t tmp = Ceil(n, cube);
    dst_shape[formats::kHwcnN] = tmp * cube;
    return KERNEL_STATUS_OK;
}

uint32_t TransDataCpuKernel::GetPaddingTwoShape(const TransArgs& args, std::vector<int64_t>& dst_shape, int64_t cube)
{
    auto n = args.src_shape.at(0);
    auto z = args.src_shape.at(1);
    constexpr size_t new_size = 2;
    dst_shape.resize(new_size);
    dst_shape[0] = n;
    int64_t tmp = Ceil(z, cube);
    dst_shape[1] = tmp * cube;
    return KERNEL_STATUS_OK;
}

uint32_t TransDataCpuKernel::Transpose(TransArgs& args, const std::vector<int64_t>& perm_arg,
                                       std::shared_ptr<uint8_t>& dst)
{
    std::vector<int64_t>& src_shape = args.src_shape;
    std::vector<int64_t> dst_shape;
    TransShapeByPerm(src_shape, perm_arg, dst_shape);
    DataType src_data_type = args.src_data_type;
    KERNEL_LOG_DEBUG("Begin to transpose, src shape [%s], perm arg [%s], dst shape [%s], data "
                     "type [%s]",
                     VectorToString(src_shape).c_str(), VectorToString(perm_arg).c_str(),
                     VectorToString(dst_shape).c_str(), DTypeStr(src_data_type).c_str());
    int64_t dst_ele_num = VectorToNum(dst_shape);
    int64_t data_size = GetSizeByDataType(src_data_type);
    int64_t dst_size = data_size * dst_ele_num;
    dst.reset(new (std::nothrow) uint8_t[dst_size], std::default_delete<uint8_t[]>());
    if (dst == nullptr) {
        KERNEL_LOG_ERROR("New Memory failed!");
        return KERNEL_STATUS_INNER_ERROR;
    }
    int64_t dst_index = 0;
    std::vector<int64_t> src_shape_head;
    GetShapeHead(src_shape, src_shape_head);
    std::vector<int64_t> dst_shape_head;
    GetShapeHead(dst_shape, dst_shape_head);
    std::map<int32_t, int32_t> index_map;
    GetIndexMap(perm_arg, index_map);
    while (dst_index < dst_ele_num) {
        auto src_index = GetSrcIndex(dst_index, src_shape, dst_shape, src_shape_head, dst_shape_head, index_map);
        const int64_t remain = dst_size - dst_index * data_size;
        const size_t protect_size = (remain < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)) ? static_cast<size_t>(remain) :
                                                                                           SECUREC_MEM_MAX_LEN;
        auto ret = memcpy_s(dst.get() + dst_index * data_size, protect_size, args.data + src_index * data_size,
                            static_cast<size_t>(data_size));
        if (ret != 0) {
            KERNEL_LOG_ERROR("Memcpy failed, ret is [%d]", ret);
            return KERNEL_STATUS_INNER_ERROR;
        }
        dst_index += 1;
    }
    src_shape = dst_shape;
    args.data = dst.get();
    return KERNEL_STATUS_OK;
}

int64_t TransDataCpuKernel::GetCubeSizeByDataType(DataType data_type)
{
    // Current cube does not support 4 bytes and longer data
    auto size = GetSizeByDataType(data_type);
    if (size <= 0) {
        KERNEL_LOG_ERROR("Failed to get cube size, the data type [%s] is invalid", DTypeStr(data_type).c_str());
        return -1;
    } else if (size == 1) {
        return kCubeSize * 2; // 32 bytes cube size
    } else {
        return kCubeSize;
    }
}
REGISTER_CPU_KERNEL(kTransData, TransDataCpuKernel);
} // namespace aicpu
