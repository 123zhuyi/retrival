/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stft_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_STFT_OPS_H_
#define OPS_OP_PROTO_INC_STFT_OPS_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes the Fourier transform of short overlapping windows of the input. \n

* @par Inputs:
* @li x: A 1-D or 2-D tensor. Must be one of the following types:
* float32, double, complex64, complex128.
* @li window: An optional tensor. The optional window function.
* Must be one of the following types: float32, double, complex64, complex128.
* Default: None (treated as window of all 1 s) \n

* @par Attributes:
* @li n_fft: A required int. Size of Fourier transform
* @li hop_length: An optional int. The distance between neighboring sliding window frames.
* Default: 0 (treated as equal to floor(n_fft/4))
* @li win_length: An optional int. The size of window frame and STFT filter.
* Default: 0 (treated as equal to n_fft)
* @li normalized: An optional bool. Controls whether to return the normalized STFT results Default: False
* @li onesided: An optional bool. Controls whether to return half of results to avoid redundancy for real inputs.
* Default: True for real input and window, False otherwise.
* @li return_complex: An optional bool. Whether to return a complex tensor, or a real tensor
* with an extra last dimension for the real and imaginary components. Default: True. \n

* @par Outputs:
* y: A tensor containing the STFT result with shape described above. Must be
* one of the following types: float32, double, complex64, complex128. \n

* @par Third-party framework compatibility
* Compatible with pytorch STFT operator.
*/
REG_OP(STFT)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .OPTIONAL_INPUT(window, TensorType({DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .ATTR(hop_length, Int, 0)
    .ATTR(win_length, Int, 0)
    .ATTR(normalized, Bool, false)
    .ATTR(onesided, Bool, true)
    .ATTR(return_complex, Bool, true)
    .REQUIRED_ATTR(n_fft, Int)
    .OP_END_FACTORY_REG(STFT)

} // namespace ge

#endif