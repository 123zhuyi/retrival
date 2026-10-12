//===- FusibleBlockOutliner.h ------------------------------------*- C++-*-===//
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

#include "bishengir/Dialect/HFusion/Pipelines/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleBlock.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_FUSIBLEBLOCKOUTLINER_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_FUSIBLEBLOCKOUTLINER_H

namespace mlir {
namespace hfusion {
namespace opfusion {
class FusibleBlockOutliner {
public:
  FusibleBlockOutliner(FusibleBlocks &fusibleBlocks,
                       const OutlineFuncOptions &options,
                       bool shouldRemoveDuplicateAliasOuts = false);

  SmallVector<func::FuncOp> getOutlinedFuncs() const;

  static void setOutlineFuncAttributes(func::FuncOp &func,
                                       const FusionKind &fusionKind,
                                       OpBuilder &builder, bool isCallerHost);
  bool outline(const std::string &prefixOutline = "");

  // get the number uses of an operation outside of the function
  size_t getNumOutsideUsesOfOp(SetVector<Operation *> &opsWithAuxs,
                               Value out) const;

private:
  size_t funcCnt_{0};
  std::string getNewFusionName(llvm::StringRef symbolName,
                               llvm::StringRef prefixName);
  void eraseTriviallyDeadOps(ArrayRef<Operation *> ops);
  func::FuncOp outlineFunc(FusibleBlock &curBlock,
                           const std::string &prefixOutline = "");
  func::CallOp createInvoke(func::FuncOp newFunc, FusibleBlock &fusionBlock);
  void removeDuplicatedAliasOutputs();

  FusibleBlocks &fusibleBlocks_;
  OutlineFuncOptions options_;
  SmallVector<func::FuncOp> outlinedFuncs_;
};
} // namespace opfusion
} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_FUSIBLEBLOCKOUTLINER_H
