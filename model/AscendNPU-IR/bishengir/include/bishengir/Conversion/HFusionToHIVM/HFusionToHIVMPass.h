//===- HFusionToHIVMPass.h - HFusion to HIVM Conversion Pass ----*- C++ -*-===//
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
// Provides passes to convert HFusion dialect to HIVM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_HFUSIONTOHIVM_HFUSIONTOHIVMPASS_H
#define BISHENGIR_CONVERSION_HFUSIONTOHIVM_HFUSIONTOHIVMPASS_H

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

#define GEN_PASS_DECL_CONVERTHFUSIONTOHIVM
#include "bishengir/Conversion/Passes.h.inc"

/// Creates a pass to convert the HFusion dialect to the HIVM dialect.
std::unique_ptr<Pass> createHFusionToHIVMConversionPass();

std::unique_ptr<Pass>
createHFusionToHIVMConversionPass(const ConvertHFusionToHIVMOptions &option);

} // namespace mlir

#endif // BISHENGIR_CONVERSION_HFUSIONTOHIVM_HFUSIONTOHIVMPASS_H
