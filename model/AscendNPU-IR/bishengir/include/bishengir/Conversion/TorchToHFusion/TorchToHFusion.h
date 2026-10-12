//===- TorchToHFusion.h - Main pass entry for Torch to HFusion ---*- C++-*-===//
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

#ifndef BISHENGIR_CONVERSION_TORCHTOHFUSION_TORCHTOHFUSION_H
#define BISHENGIR_CONVERSION_TORCHTOHFUSION_TORCHTOHFUSION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mlir {
#define GEN_PASS_DECL_CONVERTTORCHTOHFUSION
#include "bishengir/Conversion/Passes.h.inc"

/// Creates a pass to convert torch dialect ops to linalg/hfusion ops
std::unique_ptr<OperationPass<func::FuncOp>> createConvertTorchToHFusionPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertTorchToHFusionPass(const ConvertTorchToHFusionOptions &options);

} // namespace mlir

#endif // BISHENGIR_CONVERSION_TORCHTOHFUSION_ATENTONAMEDOP_H
