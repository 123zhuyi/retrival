/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_MATH_CONVERSION_TRANS_DATA_AICPU_H
#define OPS_MATH_CONVERSION_TRANS_DATA_AICPU_H

#include "cpu_kernel.h"

namespace aicpu {
struct TransArgs {
    const uint8_t* data;
    std::vector<int64_t> src_shape;
    std::vector<int64_t> dst_shape;
    DataType src_data_type;
};

class TransDataCpuKernel : public CpuKernel {
public:
    ~TransDataCpuKernel() = default;
    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    template <typename T>
    uint32_t DealData(const T* input_Data, T* output_data, const Tensor* input_tensor, Tensor* out_put_tensor,
                      int64_t group);
    uint32_t FormatTransferHwcnToFZC04(TransArgs& args, uint8_t* output_addr, uint64_t length, int64_t c0_cube);
    uint32_t PaddingOne(TransArgs& args, std::shared_ptr<uint8_t>& dst);
    uint32_t PaddingTwo(TransArgs& args, std::shared_ptr<uint8_t>& dst, int64_t c0_cube);
    uint32_t GetPaddingOneShape(const TransArgs& args, std::vector<int64_t>& dst_shape);
    uint32_t GetPaddingTwoShape(const TransArgs& args, std::vector<int64_t>& dst_shape, int64_t cube);
    uint32_t Transpose(TransArgs& args, const std::vector<int64_t>& perm_arg, std::shared_ptr<uint8_t>& dst);
    int64_t GetCubeSizeByDataType(DataType data_type);

    bool IsOriginSupportFormatTransfer(Format src_fromat, Format dst_format);

    uint32_t NewCompute(const CpuKernelContext& ctx);
    uint32_t HandleHwcnToFzC04(const Tensor* input_tensor, Tensor* output_tensor);
    uint32_t DispatchDealData(DataType dt, void* input_data_temp, void* output_data_temp, const Tensor* input_tensor,
                              Tensor* output_tensor, int64_t group);
};
} // namespace aicpu
#endif
