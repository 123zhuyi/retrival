//===- FusibleBlock.h --------------------------------------------*- C++-*-===//
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

#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleHelper.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <queue>

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_FUSIBLEBLOCK_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_FUSIBLEBLOCK_H

namespace mlir {
namespace hfusion {
namespace opfusion {
class FusibleBlock {
  using InclusionCheck = std::function<bool(Operation *, Operation *)>;

public:
  explicit FusibleBlock(const llvm::ArrayRef<Operation *> ops,
                        const FusibleHelper *fusibleHelper)
      : fusibleHelper_(fusibleHelper), ops_(ops.begin(), ops.end()){};
  explicit FusibleBlock(const llvm::ArrayRef<Operation *> ops,
                        const FusibleHelper *fusibleHelper,
                        const llvm::ArrayRef<Operation *> mod)
      : fusibleHelper_(fusibleHelper), ops_(ops.begin(), ops.end()),
        outsModification_(mod.begin(), mod.end()){};

  Operation *getLastOp() { return getOutputs().back().getDefiningOp(); }
  template <typename T> T getParentOfType() const {
    return getOps().back()->getParentOfType<T>();
  }
  Location getLoc() const { return getOps().back()->getLoc(); }

  llvm::ArrayRef<Operation *> getOps() const { return ops_.getArrayRef(); }

  llvm::ArrayRef<Value> getInputs() {
    if (ins_.empty())
      visitInValues();
    return ins_.getArrayRef();
  }

  llvm::ArrayRef<Value> getOutputs() {
    if (outs_.empty())
      visitOutValues();
    return outs_.getArrayRef();
  }

  llvm::ArrayRef<Operation *> getOpWithAuxs() {
    if (opWithAuxs_.empty())
      visitAuxiliaryOps();
    return opWithAuxs_.getArrayRef();
  }
  void dump();

  const FusibleHelper *fusibleHelper_;

private:
  void visitOutValues();
  void fillNonEdgeOps();
  void visitAuxiliaryOps();
  void visitInValues();
  void processOperandForBFS(const Value &operand, Operation *pivotOp,
                            DenseSet<Operation *> &visited,
                            std::queue<Operation *> &workQueue,
                            const InclusionCheck &shouldInclude,
                            const DenseSet<Operation *> &blocker);
  void auxBFS(const SetVector<Operation *> &initialOps,
              DenseSet<Operation *> &visited,
              const InclusionCheck &shouldInclude,
              const DenseSet<Operation *> &blocker);
  void auxBFSDown(const SetVector<Operation *> &initialOps,
                  DenseSet<Operation *> &visited,
                  const InclusionCheck &shouldInclude,
                  const DenseSet<Operation *> &blocker);

  bool isPossibleCountingAux(Operation *defOp);
  bool isValidAuxOrBuffer(Operation *defOp, Operation *pivotOp);
  bool isValidBuffer(Operation *defOp, Operation *pivotOp);

  mutable llvm::SetVector<Operation *> ops_;
  mutable llvm::SetVector<Operation *> outsModification_;
  mutable llvm::SetVector<Operation *> opWithAuxs_;
  mutable llvm::SetVector<Operation *> nonEdgeOps_;
  mutable llvm::SetVector<Value> ins_;
  mutable llvm::SetVector<Value> outs_;
};
} // namespace opfusion
} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_FUSIBLEBLOCK_H
