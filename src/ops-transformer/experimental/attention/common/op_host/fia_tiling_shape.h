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
 * \file fia_tiling_shape.h
 * \brief
 */
#ifndef FIA_TILING_SHAPE_H
#define FIA_TILING_SHAPE_H

#include "fia_tiling_info.h"

namespace optiling {
template <typename T> using CompareFunc = bool (*)(const T&, const T&);

class FiaTilingShape {
    static constexpr int64_t invalidDimValue_ = std::numeric_limits<int64_t>::min();

public:
    FiaTilingShape(const gert::Shape &shape, FiaLayout layout, std::string name, std::string opName,
        int64_t N = std::numeric_limits<int64_t>::min()) :
        shape_(shape), layout_(layout), name_(name), opName_(opName){};

public:
    const gert::Shape &shape_;
    FiaLayout layout_;
    std::string name_ ;
    std::string opName_;
    bool hasSetN_ = false;
    int64_t N_ = 1;

    size_t GetDimNum() const { return shape_.GetDimNum(); }

    bool HasShapeB() const 
    { 
        return HasAxis(FiaAxis::B); 
    }
    bool HasShapeS() const 
    { 
        return HasAxis(FiaAxis::S); 
    }
    bool HasShapeN() const 
    { 
        return HasAxis(FiaAxis::N); 
    }
    bool HasShapeD() const
    {
        if (HasAxis(FiaAxis::D)) { return true; }
        return false;
    }

    int64_t GetShapeB() const { return GetAxisNum(FiaAxis::B); }
    int64_t GetShapeS() const { return GetAxisNum(FiaAxis::S); }
    int64_t GetShapeN() const { return GetAxisNum(FiaAxis::N); }
    int64_t GetShapeD() const
    {
        if (HasAxis(FiaAxis::D)) { return shape_.GetDim(GetAxisIdx(FiaAxis::D)); }
        return invalidDimValue_;
    }
private:
    bool HasAxis(const FiaAxis &axis) const;
    size_t GetAxisIdx(const FiaAxis &axis) const;
    int64_t GetAxisNum(const FiaAxis &axis) const;
};

} // namespace optiling
#endif // FIA_TILING_SHAPE_H