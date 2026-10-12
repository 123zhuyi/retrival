//===- ReshapeAnalyzer.cpp ------------------------------------------------===//
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

#include "bishengir/Dialect/HFusion/Analysis/ReshapeAnalyzer.h"
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleHelper.h"
#include "bishengir/Dialect/Tensor/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

using namespace mlir;
using namespace mlir::hfusion;
using namespace mlir::hfusion::reshape_utils;
using namespace mlir::utils::debugger;

#define DEBUG_TYPE "reshape-analyzer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

namespace mlir {
namespace hfusion {
namespace detail {

ReshapeAnalyzer::ReshapeAnalyzer(func::FuncOp funcOp) : func(funcOp) {
  computeReshapeInputs();
  computeUnreshapedOutputs();
  funcOp->walk([this](Operation *op) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto val : block.getArguments()) {
          if (valueDependency.contains(val))
            continue;
          valueDependency[val] = valueDependency.size();
        }
      }
    }
    for (auto val : op->getResults()) {
      if (valueDependency.contains(val))
        continue;
      valueDependency[val] = valueDependency.size();
    }
  });
}

void ReshapeAnalyzer::computeReshapeInputs() {
  // Loop all the arguments
  SmallVector<ReshapeValue> simplifiedWorkList;
  // This is all the reshape heads
  for (BlockArgument arg : func.getArguments()) {
    if (!isa<RankedTensorType>(arg.getType()))
      continue;
    getReshapeDescendants(arg, simplifiedWorkList);
  }
  argIdxToReshapedInput.resize(func.getNumArguments());
  for (auto el : simplifiedWorkList) {
    int argNumber =
        static_cast<int>(cast<BlockArgument>(el.source).getArgNumber());
    argIdxToReshapedInput[argNumber].insert(el.endTarget->get());
    reshapedInputToArgIdx[el.endTarget->get()] = argNumber;
    reshapedInputs.insert(el.endTarget->get());
  }
}

void ReshapeAnalyzer::computeUnreshapedOutputs() {
  auto returnOp = dyn_cast_if_present<func::ReturnOp>(
      func.getBlocks().back().getTerminator());
  if (!returnOp)
    return;
  retIdxToUnreshapedOutputs.resize(returnOp.getNumOperands());
  for (auto &val : returnOp->getOpOperands()) {
    if (!isa<TensorType>(val.get().getType()))
      continue;
    auto reshapeHead = getReshapeHead(val.get());
    unreshapedOutputToRetIdx[reshapeHead] =
        static_cast<int64_t>(val.getOperandNumber());
    unreshapedOutputs.insert(reshapeHead);
    retIdxToUnreshapedOutputs[val.getOperandNumber()] = reshapeHead;
  }
}

SmallVector<Value> ReshapeAnalyzer::getReshapeChain(Value val) {
  SmallVector<Value> reshapeChain;
  do {
    reshapeChain.push_back(val);
    if (auto expandShape = val.getDefiningOp<tensor::ExpandShapeOp>()) {
      val = expandShape.getSrc();
    } else if (auto collapseShape =
                   val.getDefiningOp<tensor::CollapseShapeOp>()) {
      val = collapseShape.getSrc();
    } else {
      break;
    }
  } while (true);
  return reshapeChain;
}

SmallVector<Operation *>
ReshapeAnalyzer::getOpsFromReshapeValue(SmallVector<Value> chain) {
  SmallVector<Operation *> res;
  for (auto val : chain) {
    if (!isReshapeOp(val.getDefiningOp())) {
      continue;
    }
    res.push_back(val.getDefiningOp());
  }
  return res;
}

Value ReshapeAnalyzer::getReshapeHead(Value val) {
  auto chain = getReshapeChain(val);
  if (chain.size() < 1) {
    llvm::report_fatal_error("reshape chain is empty");
  }
  return chain.back();
}

Value ReshapeAnalyzer::getFirstReshape(Value val) {
  auto chain = getReshapeChain(val);
  return getFirstReshape(chain);
}

Value ReshapeAnalyzer::getFirstReshape(SmallVector<Value> &chain) {
  if (chain.size() < 2) {
    llvm::report_fatal_error("reshape chain is less than 2");
  }
  return chain[chain.size() - 2]; // get the first reshape is the 2nd back
}

void ReshapeAnalyzer::getReshapeDescendants(
    Value val, SmallVector<ReshapeValue> &descendants) {
  //  Arg (Depth = 0)  --> Collapse (Depth = 1) -------> Expand (Depth = 2)
  //                          Collapse (Depth = 3) <-------|

  std::queue<ReshapeValue> workQueue;
  auto relaxWorkList = [&](Value &curSource, Value &lastReshapeResult,
                           int currentDepth) -> void {
    LDBG("relaxing " << lastReshapeResult << " -- at " << currentDepth);
    for (OpOperand &user : lastReshapeResult.getUses()) {
      LDBG("Found usage " << *user.getOwner());
      ReshapeValue nextWork(curSource, user, currentDepth + 1);
      if (mlir::hfusion::isReshapeOp(user.getOwner())) {
        workQueue.push(nextWork);
      } else if (isExplicitlyAllowedCollapseOp(user.getOwner()) ||
                 isa<DestinationStyleOpInterface>(user.getOwner())) {
        LDBG("Found as descendant: " << nextWork.endTarget->get());
        descendants.push_back(nextWork);
      }
    }
  };
  relaxWorkList(val, val, 0);
  while (!workQueue.empty()) {
    auto currentWork = workQueue.front();
    workQueue.pop();
    Operation *currentOp = currentWork.endTarget->getOwner();
    Value reshapeResult = currentOp->getResult(0);
    relaxWorkList(currentWork.source, reshapeResult, currentWork.depth);
  }
}

void ReshapeAnalyzer::getReshapeDescendants(Value val,
                                            SetVector<Value> &descendants) {
  SmallVector<ReshapeValue> reshapedOpOperands;
  getReshapeDescendants(val, reshapedOpOperands);
  llvm::sort(reshapedOpOperands,
             [this](const ReshapeValue &a, const ReshapeValue &b) {
               return valueDependency.at(a.endTarget->get()) <
                      valueDependency.at(b.endTarget->get());
             });
  for (auto el : reshapedOpOperands) {
    descendants.insert(el.endTarget->get());
  }
}
} // namespace detail
} // namespace hfusion
} // namespace mlir
