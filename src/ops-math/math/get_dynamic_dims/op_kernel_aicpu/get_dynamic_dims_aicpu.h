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

#ifndef AICPU_KERNELS_NORMALIZED_GET_DYNAMIC_DIMS_H_
#define AICPU_KERNELS_NORMALIZED_GET_DYNAMIC_DIMS_H_

#include <vector>

#include "cpu_kernel.h"

namespace aicpu {
class GetDynamicDimsCpuKernel : public CpuKernel {
public:
    ~GetDynamicDimsCpuKernel() = default;
    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    template <typename T>
    uint32_t DoCompute(const CpuKernelContext& ctx);

    template <typename T>
    uint32_t FillOutput(const CpuKernelContext& ctx, std::vector<T>& dims);

    uint32_t GetShapeInfos(const std::vector<int64_t>& shape_info,
                           std::vector<std::vector<int64_t>>& shape_infos) const;

    template <typename T>
    uint32_t GetInputShapes(const CpuKernelContext& ctx, std::vector<std::vector<T>>& input_shapes) const;
};
} // namespace aicpu
#endif // AICPU_KERNELS_NORMALIZED_GET_DYNAMIC_DIMS_H_
