//===- UnfoldSymbolicDim.cpp ----------------------------------------------===//
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
// This file implements tensor.dim source replacer optimization
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleHelper.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dim-source-replacer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_UNFOLDSYMBOLICDIM
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

namespace mlir {
namespace hfusion {

using namespace opfusion;

namespace {
LogicalResult unfoldSymbolic(OpBuilder &builder, SymbolicDimOp symOp) {
  Operation *op = symOp->getParentOp();
  auto funcOp = cast<func::FuncOp>(op);
  builder.setInsertionPointToStart(&(*funcOp.getRegion().begin()));
  bool found = false;
  for (auto arg : funcOp.getArguments()) {
    auto symbolicArrs = getSymbolicTensor(arg.getType());
    if (!symbolicArrs.has_value())
      continue;
    for (auto [idxAttr, attr] : llvm::enumerate(symbolicArrs.value())) {
      if (attr == symOp.getSymbolName()) {
        auto newDim =
            builder.create<tensor::DimOp>(symOp.getLoc(), arg, idxAttr);
        symOp.getResult().replaceAllUsesWith(newDim);
        found = true;
        break;
      }
    }
    if (found)
      break;
  }
  return success(found);
}

} // namespace
} // namespace hfusion

} // namespace mlir

struct UnfoldSymbolicDimPass
    : public impl::UnfoldSymbolicDimBase<UnfoldSymbolicDimPass> {
  void runOnOperation() override {
    LDBG("Running UnfoldSymbolicDim");

    auto funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());
    // Walk through all tensor.empty operations
    funcOp.walk([&](mlir::hfusion::SymbolicDimOp symbolicDimOp) {
      auto res = unfoldSymbolic(builder, symbolicDimOp);
      if (res.failed()) {
        return signalPassFailure();
      }
    });
  }
};
std::unique_ptr<Pass> mlir::hfusion::createUnfoldSymbolicDimPass() {
  return std::make_unique<UnfoldSymbolicDimPass>();
}
