// RUN: bishengir-opt --hivm-bind-workspace-arg='enable-sub-workspace=true' %s | FileCheck %s

// CHECK-LABEL: func.func @bind_sub_workspace_arg
// CHECK-SAME: %[[WORKSPACE:[a-zA-Z0-9]+]]: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}
// CHECK-SAME: %[[SUB_WORKSPACE:[a-zA-Z0-9]+]]: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sub_workspace>}
func.func @bind_sub_workspace_arg(
    %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
    %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}) {
  // CHECK: memref_ext.alloc_workspace() from %[[SUB_WORKSPACE]]
  memref_ext.alloc_workspace() : memref<100xi32>
  return
}
