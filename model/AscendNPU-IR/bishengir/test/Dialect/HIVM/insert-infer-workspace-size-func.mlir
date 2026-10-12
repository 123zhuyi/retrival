// RUN: bishengir-opt --hivm-insert-infer-workspace-size-func -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// A3 / mem-based (Ascend910B4)
//===----------------------------------------------------------------------===//

// Single-offset workspace allocs: total size is max(offset + nbytes).
// 200 + 800 * sizeof(i32) = 3400.
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  // CHECK: module attributes {hacc.target = #hacc.target<"Ascend910B4">}
  // CHECK: func.func @insert_infer_workspace_size_func_infer_workspace_shape_function() -> index
  // CHECK-SAME: attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>}
  // CHECK: %[[BYTE_SIZE:.*]] = arith.constant 3400 : index
  // CHECK: return %[[BYTE_SIZE]]
  func.func @insert_infer_workspace_size_func(
                %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) {
    %cst_0 = arith.constant 0 : index
    %cst_100 = arith.constant 100 : index
    %cst_200 = arith.constant 200 : index
    memref_ext.alloc_workspace() from %arg1 offset = [%cst_0] : from memref<?xi8> to memref<100xi8>
    memref_ext.alloc_workspace() from %arg1 offset = [%cst_100] : from memref<?xi8> to memref<800xi8>
    memref_ext.alloc_workspace() from %arg1 offset = [%cst_200] : from memref<?xi8> to memref<800xi32>
    return
  }
}

// -----

// Double-buffer offsets (size == 2): size uses the last offset.
// 100 + 50 = 150.
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  // CHECK: func.func @double_buffer_offset_infer_workspace_shape_function() -> index
  // CHECK-SAME: attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>}
  // CHECK: %[[BYTE_SIZE:.*]] = arith.constant 150 : index
  // CHECK: return %[[BYTE_SIZE]]
  func.func @double_buffer_offset(
                %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) {
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    memref_ext.alloc_workspace() from %arg1 offset = [%c0, %c100] : from memref<?xi8> to memref<50xi8>
    return
  }
}

// -----

// Multi-buffer offsets (size == 4, A3 mem-based only): size uses the last offset.
// 300 + 80 = 380.
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  // CHECK: func.func @multi_buffer_offset_4_infer_workspace_shape_function() -> index
  // CHECK-SAME: attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>}
  // CHECK: %[[BYTE_SIZE:.*]] = arith.constant 380 : index
  // CHECK: return %[[BYTE_SIZE]]
  func.func @multi_buffer_offset_4(
                %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) {
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %c200 = arith.constant 200 : index
    %c300 = arith.constant 300 : index
    memref_ext.alloc_workspace() from %arg1 offset = [%c0, %c100, %c200, %c300] : from memref<?xi8> to memref<80xi8>
    return
  }
}

// -----

// Multiple multi-buffer allocs: take the max end across all of them.
// first: 100 + 50 = 150; second: 400 + 100 = 500 -> 500.
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  // CHECK: func.func @multi_alloc_multi_buffer_infer_workspace_shape_function() -> index
  // CHECK: %[[BYTE_SIZE:.*]] = arith.constant 500 : index
  // CHECK: return %[[BYTE_SIZE]]
  func.func @multi_alloc_multi_buffer(
                %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) {
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %c200 = arith.constant 200 : index
    %c400 = arith.constant 400 : index
    memref_ext.alloc_workspace() from %arg1 offset = [%c0, %c100] : from memref<?xi8> to memref<50xi8>
    memref_ext.alloc_workspace() from %arg1 offset = [%c200, %c400] : from memref<?xi8> to memref<100xi8>
    return
  }
}

// -----

// Multi-dimensional memref with non-byte element type.
// 16 * 16 * sizeof(f16) = 512.
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  // CHECK: func.func @multi_dim_f16_infer_workspace_shape_function() -> index
  // CHECK: %[[BYTE_SIZE:.*]] = arith.constant 512 : index
  // CHECK: return %[[BYTE_SIZE]]
  func.func @multi_dim_f16(
                %arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) {
    %c0 = arith.constant 0 : index
    memref_ext.alloc_workspace() from %arg0 offset = [%c0] : from memref<?xi8> to memref<16x16xf16>
    return
  }
}

// -----

// Nested AllocWorkspaceOp is still collected via walk.
// 1024 + 16*16*sizeof(f32) = 2048.
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  // CHECK: func.func @nested_alloc_workspace_infer_workspace_shape_function() -> index
  // CHECK: %[[BYTE_SIZE:.*]] = arith.constant 2048 : index
  // CHECK: return %[[BYTE_SIZE]]
  func.func @nested_alloc_workspace(
                %arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %lb: index, %ub: index, %step: index) {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    scf.for %i = %lb to %ub step %step {
      memref_ext.alloc_workspace() from %arg0 offset = [%c0] : from memref<?xi8> to memref<16x16xf32>
      memref_ext.alloc_workspace() from %arg0 offset = [%c1024] : from memref<?xi8> to memref<16x16xf32>
    }
    return
  }
}

// -----

// No AllocWorkspaceOp: do not insert the host callback.
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  // CHECK: module attributes {hacc.target = #hacc.target<"Ascend910B4">}
  // CHECK-LABEL: func.func @no_alloc_workspace_a3
  // CHECK-NOT: infer_workspace_shape_function
  func.func @no_alloc_workspace_a3(
                %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) {
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// A5 / reg-based (Ascend950PR_9589)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @insert_infer_workspace_size_func_a5_infer_workspace_shape_function() -> index
// CHECK-SAME: attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>}
// CHECK: %[[BYTE_SIZE:.*]] = arith.constant 3400 : index
// CHECK: return %[[BYTE_SIZE]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @insert_infer_workspace_size_func_a5(
                %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) {
    %cst_0 = arith.constant 0 : index
    %cst_100 = arith.constant 100 : index
    %cst_200 = arith.constant 200 : index
    memref_ext.alloc_workspace() from %arg1 offset = [%cst_0] : from memref<?xi8> to memref<100xi8>
    memref_ext.alloc_workspace() from %arg1 offset = [%cst_100] : from memref<?xi8> to memref<800xi8>
    memref_ext.alloc_workspace() from %arg1 offset = [%cst_200] : from memref<?xi8> to memref<800xi32>
    return
  }
}

// -----

// No AllocWorkspaceOp: do not insert host callback.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">}
// CHECK-NOT: infer_workspace_shape_function
// CHECK-LABEL: func.func @no_alloc_workspace
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @no_alloc_workspace(
                %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) {
    return
  }
}

// -----

// Mix AIC kernel: strip `_mix_aic` before constructing host callback name.
// CHECK-LABEL: func.func @kernel_infer_workspace_shape_function() -> index
// CHECK-SAME: attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>}
// CHECK: %[[BYTE_SIZE:.*]] = arith.constant 64 : index
// CHECK: return %[[BYTE_SIZE]]
// CHECK-LABEL: func.func @kernel_mix_aic
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @kernel_mix_aic(
                %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>})
      attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
    %c0 = arith.constant 0 : index
    memref_ext.alloc_workspace() from %arg1 offset = [%c0] : from memref<?xi8> to memref<64xi8>
    return
  }
}

// -----

// Mix AIV kernel: strip `_mix_aiv` before constructing host callback name.
// CHECK-LABEL: func.func @kernel_infer_workspace_shape_function() -> index
// CHECK-SAME: attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>}
// CHECK: %[[BYTE_SIZE:.*]] = arith.constant 128 : index
// CHECK: return %[[BYTE_SIZE]]
// CHECK-LABEL: func.func @kernel_mix_aiv
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @kernel_mix_aiv(
                %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>})
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
    %c0 = arith.constant 0 : index
    memref_ext.alloc_workspace() from %arg1 offset = [%c0] : from memref<?xi8> to memref<128xi8>
    return
  }
}

// -----

// Double offset: workspace size uses the last (end) offset.
// CHECK-LABEL: func.func @kernel_double_offset_infer_workspace_shape_function() -> index
// CHECK: %[[BYTE_SIZE:.*]] = arith.constant 96 : index
// CHECK: return %[[BYTE_SIZE]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @kernel_double_offset(
                %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    memref_ext.alloc_workspace() from %arg1 offset = [%c0, %c64] : from memref<?xi8> to memref<32xi8>
    return
  }
}

// -----

// Sub-workspace without existing host callback: create callback and rebind to workspace.
// CHECK-LABEL: func.func @kernel_sub_infer_workspace_shape_function() -> index
// CHECK-SAME: attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>}
// CHECK: %[[BYTE_SIZE:.*]] = arith.constant 300 : index
// CHECK: return %[[BYTE_SIZE]]
// CHECK-LABEL: func.func @kernel_sub
// CHECK-SAME: %[[WS:[a-zA-Z0-9]+]]: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}
// CHECK-SAME: %[[SUB_WS:[a-zA-Z0-9]+]]: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sub_workspace>}
// CHECK: memref_ext.alloc_workspace() from %[[WS]] offset = [%{{.*}}] : from memref<?xi8> to memref<100xi8>
// CHECK: memref_ext.alloc_workspace() from %[[WS]] offset = [%{{.*}}] : from memref<?xi8> to memref<200xi8>
// CHECK-NOT: memref_ext.alloc_workspace() from %[[SUB_WS]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @kernel_sub(
                %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sub_workspace>}) {
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8> to memref<100xi8>
    memref_ext.alloc_workspace() from %arg2 offset = [%c100] : from memref<?xi8> to memref<200xi8>
    return
  }
}

// -----

// Sub-workspace with existing host callback: accumulate size and shift sub-workspace offsets.
// CHECK-LABEL: func.func @kernel_sub_infer_workspace_shape_function() -> index
// CHECK: %[[BYTE_SIZE:.*]] = arith.constant 1150 : index
// CHECK: return %[[BYTE_SIZE]]
// CHECK-LABEL: func.func @kernel_sub
// CHECK-SAME: %[[WS:[a-zA-Z0-9]+]]: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}
// CHECK: %[[BASE:.*]] = arith.constant 1000 : index
// CHECK: memref_ext.alloc_workspace() from %[[WS]] offset = [%[[BASE]]] : from memref<?xi8> to memref<50xi8>
// CHECK: %[[OFFSET1:.*]] = arith.constant 1050 : index
// CHECK: memref_ext.alloc_workspace() from %[[WS]] offset = [%[[OFFSET1]]] : from memref<?xi8> to memref<100xi8>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @kernel_sub_infer_workspace_shape_function() -> index
      attributes {hacc.function_kind = #hacc.function_kind<HOST>,
                  hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c1000 = arith.constant 1000 : index
    return %c1000 : index
  }
  func.func @kernel_sub(
                %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sub_workspace>}) {
    %c0 = arith.constant 0 : index
    %c50 = arith.constant 50 : index
    memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8> to memref<50xi8>
    memref_ext.alloc_workspace() from %arg2 offset = [%c50] : from memref<?xi8> to memref<100xi8>
    return
  }
}

// -----

// Mix AIC + AIV with sub-workspace: shared host callback accumulates both cores' sizes,
// and later core's allocations are offset by the earlier core's workspace size.
// CHECK-LABEL: func.func @kernel_infer_workspace_shape_function() -> index
// CHECK: %[[BYTE_SIZE:.*]] = arith.constant 300 : index
// CHECK: return %[[BYTE_SIZE]]
// CHECK-LABEL: func.func @kernel_mix_aic
// CHECK-SAME: %[[WS_AIC:[a-zA-Z0-9]+]]: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}
// CHECK: memref_ext.alloc_workspace() from %[[WS_AIC]] offset = [%{{.*}}] : from memref<?xi8> to memref<100xi8>
// CHECK-LABEL: func.func @kernel_mix_aiv
// CHECK-SAME: %[[WS_AIV:[a-zA-Z0-9]+]]: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}
// CHECK: %[[OFFSET:.*]] = arith.constant 100 : index
// CHECK: memref_ext.alloc_workspace() from %[[WS_AIV]] offset = [%[[OFFSET]]] : from memref<?xi8> to memref<200xi8>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @kernel_mix_aic(
                %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sub_workspace>})
      attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
    %c0 = arith.constant 0 : index
    memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8> to memref<100xi8>
    return
  }
  func.func @kernel_mix_aiv(
                %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sub_workspace>})
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
    %c0 = arith.constant 0 : index
    memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8> to memref<200xi8>
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @kernel_dyn_offset(
                %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %dyn: index) {
    // expected-error@+1 {{just support `AllocWorkspaceOp` with static offset}}
    memref_ext.alloc_workspace() from %arg1 offset = [%dyn] : from memref<?xi8> to memref<64xi8>
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @kernel_dyn_shape(
                %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %dyn: index) {
    %c0 = arith.constant 0 : index
    // expected-error@+1 {{just support `AllocWorkspaceOp` with static shape result}}
    memref_ext.alloc_workspace(%dyn) from %arg1 offset = [%c0] : from memref<?xi8> to memref<?xi8>
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  // expected-error@+1 {{WorkspaceShapeFunction does not have static shape}}
  func.func @kernel_sub_infer_workspace_shape_function(%arg: index) -> index
      attributes {hacc.function_kind = #hacc.function_kind<HOST>,
                  hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    return %arg : index
  }
  func.func @kernel_sub(
                %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
                %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sub_workspace>}) {
    %c0 = arith.constant 0 : index
    memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8> to memref<50xi8>
    return
  }
}
