//===- ExtraBuffer.h --------------------------------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HFUSION_UTILS_EXTRABUFFER_H
#define BISHENGIR_DIALECT_HFUSION_UTILS_EXTRABUFFER_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace hfusion {
namespace util {

constexpr static unsigned int REDUCE_DEFAULT_FACTOR = 1;

enum class BufferSizeUnit {
  ELEMENT, // the buffer size is in unit of element
  FACTOR   // the buffer size is a factor of the input tensor/buffer size
};

/// Get extra buffer size needed for VBrcOp.
///
/// \param op `linalg.broadcast` op.
/// \param unit Buffer size unit. If it's equal to FACTOR, then the buffer size
/// is a factor of destination tensor/buffer size.
std::optional<int64_t> getExtraBufferSizeForBroadcastOp(Operation *op,
                                                        BufferSizeUnit unit);

/// Get extra buffer size needed for VReduceOp.
///
/// \param op `linalg.reduce` op.
/// \param unit Buffer size unit. If it's equal to FACTOR, then the buffer size
/// is a factor reduction op's tensor/buffer size.
std::optional<int64_t> getExtraBufferSizeForReduceOp(Operation *op,
                                                     BufferSizeUnit unit);

} // namespace util
} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_UTILS_EXTRABUFFER_H
