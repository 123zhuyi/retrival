/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_DEV_TESTS_UT_COMMON_ANY_VALUE_H
#define OPS_MATH_DEV_TESTS_UT_COMMON_ANY_VALUE_H

#include <memory>
#include <string>
#include <vector>

namespace Ops {
namespace Math {

class AnyValue {
public:
    enum ValueType {
        VT_NONE = 0,
        VT_BOOL,
        VT_INT,
        VT_FLOAT,
        VT_STRING,
        VT_LIST_BOOL,
        VT_LIST_INT,
        VT_LIST_FLOAT,
        VT_LIST_LIST_INT,
    };

    AnyValue() : type_(VT_NONE) {}

    explicit AnyValue(bool value) : type_(VT_BOOL), valuePtr_(new bool(value)) {}
    explicit AnyValue(int64_t value) : type_(VT_INT), valuePtr_(new int64_t(value)) {}
    explicit AnyValue(float value) : type_(VT_FLOAT), valuePtr_(new float(value)) {}
    explicit AnyValue(const std::string& value) : type_(VT_STRING), valuePtr_(new std::string(value)) {}
    explicit AnyValue(const std::vector<bool>& value) : type_(VT_LIST_BOOL), valuePtr_(new std::vector<bool>(value)) {}
    explicit AnyValue(const std::vector<int64_t>& value) : type_(VT_LIST_INT), valuePtr_(new std::vector<int64_t>(value)) {}
    explicit AnyValue(const std::vector<float>& value) : type_(VT_LIST_FLOAT), valuePtr_(new std::vector<float>(value)) {}
    explicit AnyValue(const std::vector<std::vector<int64_t>>& value) : type_(VT_LIST_LIST_INT), valuePtr_(new std::vector<std::vector<int64_t>>(value)) {}

    ValueType type_;
    std::shared_ptr<void> valuePtr_;
};

} // namespace Math
} // namespace Ops

#endif // OPS_MATH_DEV_TESTS_UT_COMMON_ANY_VALUE_H
