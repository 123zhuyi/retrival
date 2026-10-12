/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "get_dynamic_dims_aicpu.h"

#include "aicpu/math_aicpu_register.h"
#include "cpu_types.h"
#include "log.h"
#include "securec.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kGetDynamicDimsOutputNum = 1;
constexpr const char* kGetDynamicDims = "GetDynamicDims";
const std::vector<std::string> kDynamicDimsAttr = {"N", "shape_info"};
} // namespace

namespace aicpu {
uint32_t GetDynamicDimsCpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kDynamicInput, kGetDynamicDimsOutputNum, kDynamicDimsAttr),
                        "[%s] check params failed.", kGetDynamicDims);

    Tensor* output_tensor = ctx.Output(0);
    KERNEL_CHECK_NULLPTR(output_tensor, KERNEL_STATUS_PARAM_INVALID, "[%s] get output[0] failed.", kGetDynamicDims);
    DataType data_type = output_tensor->GetDataType();
    switch (data_type) {
        case DT_INT32:
            return DoCompute<int32_t>(ctx);
        case DT_INT64:
            return DoCompute<int64_t>(ctx);
        default:
            KERNEL_LOG_ERROR("[%s] output[0] data_type [%s] must be in {DT_INT32 DT_INT64}.", kGetDynamicDims,
                             DTypeStr(data_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
}

template <typename T>
uint32_t GetDynamicDimsCpuKernel::DoCompute(const CpuKernelContext& ctx)
{
    int64_t count = ctx.GetAttr("N")->GetInt();
    KERNEL_LOG_INFO("[%s] get attr:N [%ld].", kGetDynamicDims, count);

    std::vector<int64_t> shape_info = ctx.GetAttr("shape_info")->GetListInt();
    KERNEL_LOG_INFO("[%s] get attr:shape_info [%s].", kGetDynamicDims, VectorToString(shape_info).c_str());
    std::vector<std::vector<int64_t>> shape_infos;
    KERNEL_HANDLE_ERROR(GetShapeInfos(shape_info, shape_infos), "[%s] get shape infos failed.", kGetDynamicDims);

    uint32_t inputs_size = ctx.GetInputsSize();
    KERNEL_CHECK_FALSE((inputs_size == count), KERNEL_STATUS_PARAM_INVALID,
                       "[%s] inputs size [%u] is not match attr N [%ld].", kGetDynamicDims, inputs_size, count);
    KERNEL_CHECK_FALSE((inputs_size == shape_infos.size()), KERNEL_STATUS_PARAM_INVALID,
                       "[%s] inputs size [%u] is not match shape_infos size [%zu].", kGetDynamicDims, inputs_size,
                       shape_infos.size());

    std::vector<std::vector<T>> input_shapes;
    KERNEL_HANDLE_ERROR(GetInputShapes(ctx, input_shapes), "[%s] get input shapes failed.", kGetDynamicDims);

    std::vector<T> dims;
    for (uint32_t i = 0; i < inputs_size; ++i) {
        KERNEL_LOG_INFO("[%s] shape_infos[%u] [%s].", kGetDynamicDims, i, VectorToString(shape_infos[i]).c_str());
        KERNEL_LOG_INFO("[%s] get input[%u]'s shape [%s].", kGetDynamicDims, i,
                        VectorToString(input_shapes[i]).c_str());
        KERNEL_CHECK_FALSE((input_shapes[i].size() == shape_infos[i].size()), KERNEL_STATUS_PARAM_INVALID,
                           "[%s] input[%u] rank [%zu] is not match shape_infos[%u] rank [%zu].", kGetDynamicDims, i,
                           input_shapes[i].size(), i, shape_infos[i].size());

        for (size_t j = 0; j < input_shapes[i].size(); ++j) {
            if (shape_infos[i][j] == -1) {
                dims.push_back(input_shapes[i][j]);
            }
        }
    }
    return FillOutput<T>(ctx, dims);
}

template <typename T>
uint32_t GetDynamicDimsCpuKernel::FillOutput(const CpuKernelContext& ctx, std::vector<T>& dims)
{
    KERNEL_LOG_INFO("[%s] unknown dims: [%s].", kGetDynamicDims, VectorToString(dims).c_str());

    Tensor* output_tensor = ctx.Output(0);
    void* output_data = output_tensor->GetData();
    uint64_t output_size = output_tensor->GetDataSize();
    KERNEL_CHECK_FALSE((static_cast<size_t>(output_size) >= dims.size() * sizeof(T)), KERNEL_STATUS_INNER_ERROR,
                       "src size %zu can not exceed dst size %lu", dims.size() * sizeof(T), output_size);

    errno_t cpret = memcpy_s(output_data, output_size, dims.data(), dims.size() * sizeof(T));
    KERNEL_CHECK_FALSE((cpret == EOK), KERNEL_STATUS_INNER_ERROR,
                       "[%s] memcpy_s to output failed, destMax [%lu], count [%zu].", kGetDynamicDims, output_size,
                       dims.size() * sizeof(T));
    return KERNEL_STATUS_OK;
}

uint32_t GetDynamicDimsCpuKernel::GetShapeInfos(const std::vector<int64_t>& shape_info,
                                                std::vector<std::vector<int64_t>>& shape_infos) const
{
    for (size_t i = 0; i < shape_info.size();) {
        int64_t rank = shape_info[i++];
        KERNEL_CHECK_FALSE((rank >= 0), KERNEL_STATUS_PARAM_INVALID, "[%s] shape_info rank [%ld] must be non-negative.",
                           kGetDynamicDims, rank);
        KERNEL_CHECK_FALSE((i + static_cast<size_t>(rank) <= shape_info.size()), KERNEL_STATUS_PARAM_INVALID,
                           "[%s] shape_info is incomplete, rank [%ld].", kGetDynamicDims, rank);
        std::vector<int64_t> shape;
        for (int64_t j = 0; j < rank; ++j) {
            shape.push_back(shape_info[i++]);
        }
        shape_infos.push_back(std::move(shape));
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t GetDynamicDimsCpuKernel::GetInputShapes(const CpuKernelContext& ctx,
                                                 std::vector<std::vector<T>>& input_shapes) const
{
    for (uint32_t i = 0; i < ctx.GetInputsSize(); ++i) {
        Tensor* input_tensor = ctx.Input(i);
        KERNEL_CHECK_NULLPTR(input_tensor, KERNEL_STATUS_INNER_ERROR, "[%s] get input[%u] failed.", kGetDynamicDims, i);
        int64_t input_size = input_tensor->NumElements();
        T* input_data = static_cast<T*>(input_tensor->GetData());
        KERNEL_CHECK_NULLPTR(input_data, KERNEL_STATUS_INNER_ERROR, "[%s] get input[%u] data failed.", kGetDynamicDims,
                             i);
        std::vector<T> input_shape(input_data, input_data + input_size);
        input_shapes.emplace_back(std::move(input_shape));
    }

    return KERNEL_STATUS_OK;
}

OPS_MATH_REGISTER_CPU_KERNELV2(kGetDynamicDims, GetDynamicDimsCpuKernel);
} // namespace aicpu
