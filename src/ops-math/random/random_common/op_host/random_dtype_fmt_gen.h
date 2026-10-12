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
 * \file random_dtype_fmt_gen.h
 * \brief
 * 使用示例:
 * \code
 * // 1. 准备输入数据 
 *  std::vector<ge::DataType> types = {ge::DT_FLOAT, ge::DT_INT32};
 *  std::vector<ge::Format> fmts = {ge::FORMAT_NCHW, ge::FORMAT_NHWC};
 * // 2. 初始化生成器
 *  randomdef::RandomDtypeFmtGen gen({
 *  {"types", types},
 *  {"fmts", fmts}
 *  });
 * // 3. 获取生成后的对齐序列
 *  auto typeSeq = gen.GetSequence<ge::DataType>("dtype"); // ge::DataType可省略模板参数
 *  auto fmtSeq = gen.GetSequence<ge::Format>("format"); // ge::Format不可省略模板参数
 * // 可选调试打印
 *  gen.Print<ge::DataType>("dtype"); // ge::DataType可省略模板参数
 *  gen.Print<ge::Format>("format"); // ge::Format不可省略模板参数
 * \endcode
 */
#ifndef RANDOM_DTYPE_FMT_GEN_H
#define RANDOM_DTYPE_FMT_GEN_H

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <iomanip>

namespace randomdef {
namespace detail {
#define GE_CASE(VAL) case ge::VAL: return #VAL
inline std::string TypeToStr(ge::DataType type)
{
    switch (type) {
        GE_CASE(DT_FLOAT);  GE_CASE(DT_FLOAT16); GE_CASE(DT_BF16);
        GE_CASE(DT_INT8);   GE_CASE(DT_INT16);   GE_CASE(DT_INT32);  GE_CASE(DT_INT64);
        GE_CASE(DT_UINT8);  GE_CASE(DT_UINT16);  GE_CASE(DT_UINT32); GE_CASE(DT_UINT64);
        GE_CASE(DT_BOOL);
        default: return "DT_" + std::to_string(type);
    }
}

inline std::string TypeToStr(ge::Format fmt)
{
    switch (fmt) {
        GE_CASE(FORMAT_NCHW); GE_CASE(FORMAT_NHWC); GE_CASE(FORMAT_ND);
        default: return "FMT_" + std::to_string(fmt);
    }
}
#undef GE_CASE 

template <typename T>
inline void PrintByColsCore(const std::vector<T>& seq, const char* varName, size_t cols)
{
    static constexpr int COL_WIDTH = 15;
    if (cols == 0) {
        std::cerr << "[Warning] PrintByColsCore: cols is 0, doing nothing." << std::endl;
        return;
    }
    std::cout << ">>> Sequence '" << varName << "' (Total: " << seq.size() << ", Cols: " << cols << "):" << std::endl;
    for (size_t i = 0; i < seq.size(); ++i) {
        std::cout << std::left << std::setw(COL_WIDTH) << TypeToStr(seq[i]);
        if ((i + 1) % cols == 0) {
            std::cout << std::endl;
        }
    }
    if (seq.size() % cols != 0) {
        std::cout << std::endl;
    }
    std::cout << "------------------------------------------------------------" << std::endl;
}
} // namespace detail

struct InputOption {
    std::string name;
    std::vector<int64_t> data;
    template <typename T = ge::DataType>
    InputOption(std::string n, const std::vector<T>& v) : name(std::move(n))
    {
        static_assert(
            std::is_arithmetic<T>::value || std::is_enum<T>::value, "InputOption data must be arithmetic or enum");
        data.reserve(v.size());
        for (const auto& item : v) {
            data.push_back(static_cast<int64_t>(item));
        }
    }
};

class RandomDtypeFmtGen  {
public:
    static constexpr size_t MAX_COMBINATIONS_LIMIT = 100000000;

    RandomDtypeFmtGen(std::vector<InputOption> inputs)
    {
        size_t numInputs = inputs.size();
        rawInputs_.reserve(numInputs);
        names_.reserve(numInputs);
        strides_.resize(numInputs);

        if (inputs.empty()) {
            totalCombinations_ = 0;
            return;
        }
        totalCombinations_ = 1;
        size_t idx = 0;
        for (auto& item : inputs) {
            if (nameMap_.count(item.name) != 0U) {
                throw std::runtime_error("Duplicate input name: " + item.name);
            }
            nameMap_[item.name] = idx++;
            names_.push_back(std::move(item.name));
            rawInputs_.push_back(std::move(item.data));
        }

        size_t currentStride = 1;
        for (size_t i = numInputs; i-- > 0;) {
            strides_[i] = currentStride;
            size_t currentSize = rawInputs_[i].size();
            if (currentSize == 0) {
                totalCombinations_ = 0;
                return;
            }
            if (currentStride > std::numeric_limits<size_t>::max() / currentSize) {
                throw std::overflow_error("Total combinations overflow size_t");
            }
            if (currentStride * currentSize > MAX_COMBINATIONS_LIMIT) {
                throw std::length_error(
                    "Total combinations exceed safety limit (" + std::to_string(MAX_COMBINATIONS_LIMIT) + ")");
            }
            currentStride *= currentSize;
        }
        totalCombinations_ = currentStride;
    }

    template <typename T = ge::DataType>
    [[nodiscard]] std::vector<T> GetSequence(const std::string& name) const
    {
        auto it = nameMap_.find(name);
        if (it == nameMap_.end()) {
            throw std::invalid_argument("Input name not found: " + name);
        }
        std::vector<T> result;
        if (totalCombinations_ == 0)
            return result;
        FillColumn(it->second, result);
        return result;
    }

    template <typename T = ge::DataType>
    void Print(const std::string& name, size_t cols = 6) const
    {
        std::vector<T> seq = GetSequence<T>(name);
        detail::PrintByColsCore<T>(seq, name.c_str(), cols);
    }

private:
    std::vector<std::vector<int64_t>> rawInputs_;
    std::vector<std::string> names_;
    std::unordered_map<std::string, size_t> nameMap_;
    std::vector<size_t> strides_;
    size_t totalCombinations_ = 0;

    template <typename T>
    void FillColumn(size_t inputIdx, std::vector<T>& result) const
    {
        result.resize(totalCombinations_);

        const auto& sourceVec = rawInputs_[inputIdx];
        size_t stride = strides_[inputIdx];
        size_t srcSize = sourceVec.size();

        size_t srcIdx = 0;
        std::vector<T> convertedSource;
        convertedSource.reserve(srcSize);
        for (auto v : sourceVec)
            convertedSource.push_back(static_cast<T>(v));
        for (size_t destPtr = 0; destPtr < totalCombinations_; destPtr += stride) {
            T val = convertedSource[srcIdx];
            std::fill_n(result.begin() + destPtr, stride, val);
            if (++srcIdx == srcSize) {
                srcIdx = 0;
            }
        }
    }
};
} // namespace randomdef

#endif // RANDOM_DTYPE_FMT_GEN_H
