//===- Passes.h - BiShengIR Torch pipeline entry points ---------*- C++ -*-===//
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
// This header file defines prototypes of all BiShengIR Torch pipelines.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_TORCH_PIPELINES_PASSES_H
#define BISHENGIR_DIALECT_TORCH_PIPELINES_PASSES_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace bishengir {
struct TorchToNamedOpPipelineOptions
    : public mlir::PassPipelineOptions<TorchToNamedOpPipelineOptions> {
  // -------------------------------------------------------------------------//
  //                       feature control options
  // -------------------------------------------------------------------------//
  PassOptions::Option<bool> ensureNoImplicitBroadcast{
      *this, "ensure-no-implicit-broadcast",
      llvm::cl::desc("Whether to ensure that there is no implicit broadcast "
                     "semantics. If there is a dynamic to dynamic dim "
                     "broadcast, raise a runtime error."),
      llvm::cl::init(false)};
};

void registerTorchToHFusionPipelines();

/// Creates a pipeline that lowers from the torch backend contract to the
/// linalg-on-tensors backend contract.
void createTorchBackendToNamedOpBackendPipeline(
    mlir::OpPassManager &pm, const TorchToNamedOpPipelineOptions &options);
} // namespace bishengir

#endif // BISHENGIR_DIALECT_TORCH_PIPELINES_PASSES_H
