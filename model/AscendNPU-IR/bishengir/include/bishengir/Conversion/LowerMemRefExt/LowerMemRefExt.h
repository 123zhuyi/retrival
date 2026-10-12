//===----- LowerMemRefExt.h - Lower Extended MemRef Dialect -----*- C++ -*-===//
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
//
// Define conversions from the MemRefExt dialect to the HIVM IR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_LOWERMEMREFEXT_LOWERMEMREFEXT_H
#define BISHENGIR_CONVERSION_LOWERMEMREFEXT_LOWERMEMREFEXT_H

#include <memory>
#include <string>

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_LOWERMEMREFEXT
#include "bishengir/Conversion/Passes.h.inc"

std::unique_ptr<Pass> createMemrefExtLoweringPass();
} // namespace mlir
#endif
