//===-- DeadFunctionElimination.cpp -----------------------------*- C++ -*-===//
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

#include "bishengir/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dead-function-elimination"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace bishengir {
#define GEN_PASS_DEF_DEADFUNCTIONELIMINATION
#include "bishengir/Transforms/Passes.h.inc"
} // namespace bishengir

using namespace mlir;

void bishengir::eliminateDeadFunctions(
    ModuleOp module, const bishengir::DeadFunctionEliminationOptions &options) {
  module->walk([&](FunctionOpInterface funcLikeOp) {
    if (!SymbolTable::symbolKnownUseEmpty(funcLikeOp, module)) {
      LDBG("Symbol: @" << funcLikeOp.getName() << " is still in use.");
      return;
    }

    if (!options.filterFn(funcLikeOp)) {
      LDBG("Symbol: @" << funcLikeOp.getName()
                       << " doesn't satisfy requirement.");
      return;
    }
    funcLikeOp.erase();
  });
}

namespace {

struct DeadFunctionEliminationPass
    : public bishengir::impl::DeadFunctionEliminationBase<
          DeadFunctionEliminationPass> {
  explicit DeadFunctionEliminationPass(
      const bishengir::DeadFunctionEliminationOptions &options)
      : options(options) {}

  void runOnOperation() override {
    bishengir::eliminateDeadFunctions(getOperation(), options);
  }

private:
  bishengir::DeadFunctionEliminationOptions options;
};

} // namespace

std::unique_ptr<mlir::Pass> bishengir::createDeadFunctionEliminationPass(
    const bishengir::DeadFunctionEliminationOptions &options) {
  return std::make_unique<DeadFunctionEliminationPass>(options);
}
