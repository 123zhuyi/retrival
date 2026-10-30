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
 * \file tiling_util.h
 * \brief
 */

#ifndef TILING_UTILS_H
#define TILING_UTILS_H

#include <stdexcept>

class TilingUtils {
public:
    template <typename T>
    static std::vector<int64_t> GetShape(T x)
    {
#if defined(TORCH_MODE)
        if constexpr (std::is_same_v<T, at::Tensor>) {
            return x.sizes().vec();
        } else {
            throw std::runtime_error("Unsupported template type: only at::Tensor is supported when using bisheng");
        }
#else
        if constexpr (std::is_same_v<T, gert::Shape>) {
            return x.GetDims();
        } else {
            throw std::runtime_error("Unsupported template type: only gert::Shape is supported");
        }
#endif
    }

    template <typename T>
    static int64_t GetDimNum(T x)
    {
#if defined(TORCH_MODE)
        if constexpr (std::is_same_v<T, at::Tensor>) {
            return x.dim();
        } else {
            throw std::runtime_error("Unsupported template type: only at::Tensor is supported when using bisheng");
        }
#else
        if constexpr (std::is_same_v<T, gert::Shape>) {
            return x.GetDimNum();
        } else {
            throw std::runtime_error("Unsupported template type: only gert::Shape is supported");
        }
#endif
    }

    template <typename T>
    static int64_t GetDim(T x, int64_t i)
    {
#if defined(TORCH_MODE)
        if constexpr (std::is_same_v<T, at::Tensor>) {
            return x.size(i);
        } else {
            throw std::runtime_error("Unsupported template type: only at::Tensor is supported when using bisheng");
        }
#else
        if constexpr (std::is_same_v<T, gert::Shape>) {
            return x.GetDim(i);
        } else {
            throw std::runtime_error("Unsupported template type: only gert::Shape is supported");
        }
#endif
    }
};

#endif // TILING_UTILS_H