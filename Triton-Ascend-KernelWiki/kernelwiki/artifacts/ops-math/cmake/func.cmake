# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 为target link依赖库 usage: add_modules(MODE SUB_LIBS EXTERNAL_LIBS) SUB_LIBS 为内部创建的target, EXTERNAL_LIBS为外部依赖的target
function(add_modules)
  set(oneValueArgs MODE)
  set(multiValueArgs TARGETS SUB_LIBS EXTERNAL_LIBS)

  cmake_parse_arguments(ARGS "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  foreach(target ${ARGS_TARGETS})
    if(TARGET ${target} AND TARGET ${ARGS_SUB_LIBS})
      target_link_libraries(${target} ${ARGS_MODE} ${ARGS_SUB_LIBS})
    endif()
    if(TARGET ${target} AND ARGS_EXTERNAL_LIBS)
      target_link_libraries(${target} ${ARGS_MODE} ${ARGS_EXTERNAL_LIBS})
    endif()
  endforeach()
endfunction()

# 添加opbase object
function(add_opbase_modules)
  if(TARGET opbase_infer_objs OR TARGET opbase_tiling_objs OR TARGET opbase_util_objs)
    return()
  endif()
  file(GLOB_RECURSE OPS_BASE_INFER_SRC
    ${OPBASE_SOURCE_PATH}/src/op_common/op_host/infershape_broadcast_util.cpp
    ${OPBASE_SOURCE_PATH}/src/op_common/op_host/infershape_elewise_util.cpp
    ${OPBASE_SOURCE_PATH}/src/op_common/op_host/infershape_reduce_util.cpp
  )

  file(GLOB_RECURSE OPS_BASE_TILING_SRC
    ${OPBASE_SOURCE_PATH}/src/op_common/atvoss/elewise/*.cpp
    ${OPBASE_SOURCE_PATH}/src/op_common/atvoss/broadcast/*.cpp
    ${OPBASE_SOURCE_PATH}/src/op_common/atvoss/reduce/*.cpp
    ${OPBASE_SOURCE_PATH}/src/op_common/op_host/tiling_util.cpp
  )

  file(GLOB_RECURSE OPS_BASE_UTIL_SRC
    ${OPBASE_SOURCE_PATH}/src/op_common/op_host/util/*.cpp
    ${OPBASE_SOURCE_PATH}/src/op_common/log/*.cpp
  )

  if(OPS_BASE_INFER_SRC)
    add_library(opbase_infer_objs OBJECT ${OPS_BASE_INFER_SRC})
    target_include_directories(opbase_infer_objs PRIVATE ${OP_PROTO_INCLUDE})
    target_compile_options(opbase_infer_objs
        PRIVATE
        $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
        -fvisibility=hidden
    )
    target_link_libraries(
      opbase_infer_objs
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
      )
  endif()

  if(OPS_BASE_TILING_SRC)
    add_library(opbase_tiling_objs OBJECT ${OPS_BASE_TILING_SRC})
    target_include_directories(opbase_tiling_objs PRIVATE ${OP_TILING_INCLUDE})
    target_compile_options(opbase_tiling_objs
        PRIVATE
        $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                                        -fvisibility=hidden -fno-strict-aliasing
    )
    target_link_libraries(
      opbase_tiling_objs
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
              tiling_api
      )
  endif()

  if(OPS_BASE_UTIL_SRC)
    add_library(opbase_util_objs OBJECT ${OPS_BASE_UTIL_SRC})
    target_include_directories(opbase_util_objs PRIVATE ${OP_TILING_INCLUDE} ${ASCEND_DIR}/pkg_inc)
    target_compile_options(opbase_util_objs
        PRIVATE
        $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
        -fvisibility=hidden
    )
    target_link_libraries(
      opbase_util_objs
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
      )
  endif()
endfunction()

# 添加infer object
function(add_infer_modules)
  if(NOT TARGET ${OPHOST_NAME}_infer_obj)
    add_library(${OPHOST_NAME}_infer_obj OBJECT)
    if(ENABLE_CUSTOM)
      string(TOUPPER "${VENDOR_NAME}" UPPER_VENDOR_NAME)
      set(OP_SUBMOD_NAME "OPS_MATH_${UPPER_VENDOR_NAME}")
    else()
      set(OP_SUBMOD_NAME "OPS_MATH")
    endif()
    target_include_directories(${OPHOST_NAME}_infer_obj PRIVATE ${OP_PROTO_INCLUDE})
    target_compile_definitions(
      ${OPHOST_NAME}_infer_obj PRIVATE OPS_UTILS_LOG_SUB_MOD_NAME="OP_PROTO" OP_SUBMOD_NAME="${OP_SUBMOD_NAME}"
                                       $<$<BOOL:${ENABLE_TEST}>:ASCEND_OPSPROTO_UT>
      )
    target_compile_options(
      ${OPHOST_NAME}_infer_obj PRIVATE $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                                       -fvisibility=hidden
      )
    target_link_libraries(
      ${OPHOST_NAME}_infer_obj
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
              $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>> 
              $<$<TARGET_EXISTS:opbase_infer_objs>:$<TARGET_OBJECTS:opbase_infer_objs>>
      )
  endif()
endfunction()

# 添加tiling object
function(add_tiling_modules)
  if(NOT TARGET ${OPHOST_NAME}_tiling_obj)
    add_library(${OPHOST_NAME}_tiling_obj OBJECT)
    if(ENABLE_CUSTOM)
      string(TOUPPER "${VENDOR_NAME}" UPPER_VENDOR_NAME)
      set(OP_SUBMOD_NAME "OPS_MATH_${UPPER_VENDOR_NAME}")
    else()
      set(OP_SUBMOD_NAME "OPS_MATH")
    endif()
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${OP_TILING_INCLUDE})
    target_compile_definitions(
      ${OPHOST_NAME}_tiling_obj PRIVATE OPS_UTILS_LOG_SUB_MOD_NAME="OP_TILING" OP_SUBMOD_NAME="${OP_SUBMOD_NAME}"
                                        $<$<BOOL:${ENABLE_TEST}>:ASCEND_OPTILING_UT>
      )
    target_compile_options(
      ${OPHOST_NAME}_tiling_obj PRIVATE $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                                        -fvisibility=hidden -fno-strict-aliasing
      )
    target_link_libraries(
      ${OPHOST_NAME}_tiling_obj
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
              $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
              $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>> 
              $<$<TARGET_EXISTS:opbase_tiling_objs>:$<TARGET_OBJECTS:opbase_tiling_objs>>
              tiling_api
      )
  endif()
endfunction()

# 添加opapi object
function(add_opapi_modules)
  if(NOT TARGET ${OPHOST_NAME}_opapi_obj)
    add_library(${OPHOST_NAME}_opapi_obj OBJECT)
    unset(opapi_ut_depends_inc)
    if(ENABLE_TEST)
      set(opapi_ut_depends_inc ${UT_PATH}/op_api/stub ${UT_PATH}/op_api/stub/dev/aicpu)
    endif()
    target_include_directories(${OPHOST_NAME}_opapi_obj PRIVATE
            ${opapi_ut_depends_inc}
            ${OPAPI_INCLUDE})
    target_compile_options(${OPHOST_NAME}_opapi_obj PRIVATE -Dgoogle=ascend_private -DACLNN_LOG_FMT_CHECK -DOP_LOG_LIBOPAPI_ONLY)
    target_link_libraries(
      ${OPHOST_NAME}_opapi_obj
      PUBLIC $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
      PRIVATE $<BUILD_INTERFACE:adump_headers> $<BUILD_INTERFACE:dlog_headers>
              $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
      )
  endif()
endfunction()

function(add_aicpu_kernel_modules)
  if(NOT TARGET ${OPHOST_NAME}_aicpu_obj)
    add_library(${OPHOST_NAME}_aicpu_obj OBJECT)
    target_include_directories(${OPHOST_NAME}_aicpu_obj PRIVATE ${AICPU_INCLUDE})
    target_compile_definitions(
      ${OPHOST_NAME}_aicpu_obj PRIVATE _FORTIFY_SOURCE=2 google=ascend_private
                                       $<$<BOOL:${ENABLE_TEST}>:ASCEND_AICPU_UT>
      )
    target_compile_options(
      ${OPHOST_NAME}_aicpu_obj PRIVATE $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                                       -fvisibility=hidden ${AICPU_DEFINITIONS}
      )
    target_link_libraries(
      ${OPHOST_NAME}_aicpu_obj
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
      )
  endif()
endfunction()

function(add_aicpu_cust_kernel_modules op_name aicpu_sources)
  set(target_name ${op_name}_obj)
  if(NOT TARGET ${target_name})
    add_library(${target_name} OBJECT)
    target_include_directories(${target_name} PRIVATE ${AICPU_INCLUDE})
    target_compile_definitions(
      ${target_name} PRIVATE
                    _FORTIFY_SOURCE=2 _GLIBCXX_USE_CXX11_ABI=1
                    google=ascend_private
                    $<$<BOOL:${ENABLE_TEST}>:ASCEND_AICPU_UT>
      )
    target_compile_options(
      ${target_name} PRIVATE
                    $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                    -fvisibility=hidden ${AICPU_DEFINITIONS}
      )
    target_link_libraries(
      ${target_name}
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
              -Wl,--no-whole-archive
              Eigen3::Eigen
      )
    if (NOT (UT_TEST_ALL OR OP_KERNEL_AICPU_UT))
      set_property(TARGET ${target_name} PROPERTY 
        CXX_COMPILER_LAUNCHER ${ASCEND_DIR}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++)
    endif()    
    target_sources(${target_name} PRIVATE ${aicpu_sources})
    if (NOT ${target_name} IN_LIST AICPU_CUST_OBJ_TARGETS)
      set(AICPU_CUST_OBJ_TARGETS ${AICPU_CUST_OBJ_TARGETS} ${target_name} CACHE INTERNAL "All aicpu cust obj targets")
    endif()
  endif()
endfunction()

# Compiles aicpu source as OBJECT for host side (x86).
# Collects into AICPU_HOST_OBJ_TARGETS; linking into libopconstant_folding_math.so
# is done in symbol.cmake gen_aicpu_const_symbol().
function(add_aicpu_host_kernel_modules host_target_name)
  message(STATUS "add_aicpu_host_kernel_modules for ${host_target_name}")
  if(NOT TARGET ${host_target_name})
    add_library(${host_target_name} OBJECT)
    target_include_directories(${host_target_name} PRIVATE
        ${AICPU_INCLUDE}
        ${CANN_3RD_LIB_PATH}/eigen
    )
    target_compile_definitions(
      ${host_target_name} PRIVATE
                    _FORTIFY_SOURCE=2
                    google=ascend_private
                    OPS_MATH_AICPU_HOST_KERNEL
      )
    target_compile_options(
      ${host_target_name} PRIVATE
                    -Dgoogle=ascend_private
                    -fvisibility=hidden ${AICPU_DEFINITIONS}
      )
    target_link_libraries(
      ${host_target_name}
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
              Eigen3::Eigen
      )
    if (NOT ${host_target_name} IN_LIST AICPU_HOST_OBJ_TARGETS)
      set(AICPU_HOST_OBJ_TARGETS
          ${AICPU_HOST_OBJ_TARGETS} ${host_target_name}
          CACHE INTERNAL "All aicpu host builtin obj targets")
    endif()
  endif()
endfunction()

function(add_op_graph_modules)
  if(NOT TARGET ${GRAPH_PLUGIN_NAME}_obj)
    add_library(${GRAPH_PLUGIN_NAME}_obj OBJECT)
    target_include_directories(${GRAPH_PLUGIN_NAME}_obj PRIVATE ${OP_PROTO_INCLUDE})
    target_compile_definitions(${GRAPH_PLUGIN_NAME}_obj PRIVATE OPS_UTILS_LOG_SUB_MOD_NAME="OP_GRAPH")

    if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
      target_compile_options(
        ${GRAPH_PLUGIN_NAME}_obj PRIVATE -Dgoogle=ascend_private -fvisibility=hidden
      )
    else()
      target_compile_options(
        ${GRAPH_PLUGIN_NAME}_obj PRIVATE $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                                       -fvisibility=hidden
      )
    endif()

    target_link_libraries(
      ${GRAPH_PLUGIN_NAME}_obj
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
              $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>> 
              $<$<TARGET_EXISTS:opbase_infer_objs>:$<TARGET_OBJECTS:opbase_infer_objs>>
      )
  endif()
endfunction()

function(add_onnx_plugin_modules)
  if (NOT TARGET ${ONNX_PLUGIN_NAME}_obj)
    set(ge_onnx_proto_srcs
      ${ASCEND_DIR}/include/proto/ge_onnx.proto)

    protobuf_generate_external(onnx ge_onnx_proto_cc ge_onnx_proto_h ${ge_onnx_proto_srcs})

    add_library(${ONNX_PLUGIN_NAME}_obj OBJECT ${ge_onnx_proto_h})
    # 为特定目标设置C++14标准
    set_target_properties(${ONNX_PLUGIN_NAME}_obj PROPERTIES
      CXX_STANDARD 14
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
    )
    target_include_directories(${ONNX_PLUGIN_NAME}_obj
      PRIVATE
      ${OP_PROTO_INCLUDE}
      ${HOST_PROTOC_SRC}
      ${HOST_PROTOC_PATH}
      ${PROTOBUF_INCLUDE_DIRS}
      ${CMAKE_BINARY_DIR}/proto
      ${ONNX_PLUGIN_COMMON_INCLUDE}
      ${JSON_INCLUDE_DIR}
      ${ABS_INSTALL_DIR}
    )
    target_compile_definitions(${ONNX_PLUGIN_NAME}_obj PRIVATE OPS_UTILS_LOG_SUB_MOD_NAME="ONNX_PLUGIN")

    if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
      target_compile_options(
        ${ONNX_PLUGIN_NAME}_obj PRIVATE -Dgoogle=ascend_private -fvisibility=hidden -Wno-shadow -Wno-unused-parameter
      )
    else()
      target_compile_options(
        ${ONNX_PLUGIN_NAME}_obj PRIVATE $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                                       -fvisibility=hidden -Wno-shadow -Wno-unused-parameter
      )
    endif()

    target_link_libraries(
      ${ONNX_PLUGIN_NAME}_obj
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx14,intf_pub_cxx14>>
              $<BUILD_INTERFACE:dlog_headers>
              $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>> 
              json
      )
  endif()
endfunction()

# usage: add_modules_sources(OPTYPE ACLNNTYPE) ACLNNTYPE 支持类型aclnn/aclnn_inner/aclnn_exclude OPTYPE 和 ACLNNTYPE
# 需一一对应
macro(add_modules_sources)
  set(multiValueArgs OPTYPE ACLNNTYPE)

  cmake_parse_arguments(MODULE "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  add_opbase_modules()
  # opapi l0 默认全部编译
  file(GLOB OPAPI_L0_SRCS ${SOURCE_DIR}/op_api/*.cpp)
  list(FILTER OPAPI_L0_SRCS EXCLUDE REGEX ".*/aclnn_[^/]*$")
  if(OPAPI_L0_SRCS)
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_L0_SRCS})
  endif()

  # 获取算子层级目录名称，判断是否编译该算子
  get_filename_component(PARENT_DIR ${SOURCE_DIR} DIRECTORY)
  get_filename_component(OP_NAME ${PARENT_DIR} NAME)
  if(NEED_COMPILE_OPS AND (NOT OP_NAME IN_LIST NEED_COMPILE_OPS))
    # NEED_COMPILE_OPS 为空表示全部编译
    return()
  endif()
  
  if(OP_NAME IN_LIST COMPILED_OPS)
    # 已经编译过，忽略
    message(STATUS "already compiled ${OP_NAME}, skip")
    return()
  endif()

  # 记录全局的COMPILED_OPS和COMPILED_OP_DIRS，其中COMPILED_OP_DIRS只记录到算子名，例如math/abs
  set(COMPILED_OPS
      ${COMPILED_OPS} ${OP_NAME}
      CACHE STRING "Compiled Ops" FORCE
    )
  set(COMPILED_OP_DIRS
      ${COMPILED_OP_DIRS} ${PARENT_DIR}
      CACHE STRING "Compiled Ops Dirs" FORCE
    )

  file(GLOB OPAPI_HEADERS ${SOURCE_DIR}/op_api/aclnn_*.h)
  file(GLOB OPAPI_HEADERS_ACL ${SOURCE_DIR}/op_api/acl_*.h)
  list(APPEND OPAPI_HEADERS ${OPAPI_HEADERS_ACL})
  if(OPAPI_HEADERS)
    target_sources(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE ${OPAPI_HEADERS})
  endif()

  file(GLOB OPAPI_L2_SRCS ${SOURCE_DIR}/op_api/aclnn_*.cpp)
  file(GLOB OPAPI_HEADERS_ACL ${SOURCE_DIR}/op_api/acl_*.h)
  list(APPEND OPAPI_HEADERS ${OPAPI_HEADERS_ACL})
  if(OPAPI_L2_SRCS)
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_L2_SRCS})
  endif()

  file(GLOB OPINFER_SRCS ${SOURCE_DIR}/*_infershape*.cpp)
  if(OPINFER_SRCS)
    add_infer_modules()
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${OPINFER_SRCS})
  endif()

  file(GLOB OPTILING_SRCS ${SOURCE_DIR}/*_tiling*.cpp)
  if(OPTILING_SRCS)
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${OPTILING_SRCS})
  endif()

  file(GLOB AICPU_SRCS ${SOURCE_DIR}/*_aicpu.cpp)
  file(GLOB AICPU_OP_DEF_SRCS ${SOURCE_DIR}/*_aicpu_def.cpp)
  file(GLOB AICPU_JSON_FILES ${SOURCE_DIR}/*.json)
  if(AICPU_OP_DEF_SRCS)
    set_property(GLOBAL APPEND PROPERTY AICPU_OPDEF_FILES ${AICPU_OP_DEF_SRCS})
  endif()
  if(AICPU_JSON_FILES)
    set_property(GLOBAL APPEND PROPERTY AICPU_JSON_FILES ${AICPU_JSON_FILES})
  endif()
  if(AICPU_SRCS)
    add_aicpu_kernel_modules()
    target_sources(${OPHOST_NAME}_aicpu_obj PRIVATE ${AICPU_SRCS})
  endif()

  if(MODULE_OPTYPE)
    list(LENGTH MODULE_OPTYPE OpTypeLen)
    list(LENGTH MODULE_ACLNNTYPE AclnnTypeLen)
    if(NOT ${OpTypeLen} EQUAL ${AclnnTypeLen})
      message(FATAL_ERROR "OPTYPE AND ACLNNTYPE Should be One-to-One")
    endif()
    math(EXPR index "${OpTypeLen} - 1")
    foreach(i RANGE ${index})
      list(GET MODULE_OPTYPE ${i} OpType)
      list(GET MODULE_ACLNNTYPE ${i} AclnnType)
      if(${AclnnType} STREQUAL "aclnn"
         OR ${AclnnType} STREQUAL "aclnn_inner"
         OR ${AclnnType} STREQUAL "aclnn_exclude"
        )
        file(GLOB OPDEF_SRCS ${SOURCE_DIR}/${OpType}_def*.cpp)
        if(OPDEF_SRCS)
          target_sources(${OPHOST_NAME}_opdef_${AclnnType}_obj INTERFACE ${OPDEF_SRCS})
        endif()
      else()
        message(FATAL_ERROR "ACLNN TYPE UNSPPORTED, ONLY SUPPORT aclnn/aclnn_inner/aclnn_exclude")
      endif()
    endforeach()
  else()
    file(GLOB OPDEF_SRCS ${SOURCE_DIR}/*_def*.cpp)
    if(OPDEF_SRCS)
      message(
        FATAL_ERROR
          "Should Manually specify aclnn/aclnn_inner/aclnn_exclude\n"
          "usage: add_modules_sources(OPTYPE optypes ACLNNTYPE aclnntypes)\n"
          "example: add_modules_sources(OPTYPE add ACLNNTYPE aclnn_exclude)"
        )
    endif()
  endif()
endmacro()

# 添加待编译的算子
function(add_need_compile_ops op_name)
  if(NOT NEED_COMPILE_OPS)
    # 为空则不需要更新
    return()
  endif()

  set(NEW_OP_NAMES ${NEED_COMPILE_OPS} ${op_name})
  list(REMOVE_DUPLICATES NEW_OP_NAMES)
  set(NEED_COMPILE_OPS
      ${NEW_OP_NAMES}
      CACHE STRING "Ascend op names to compile" FORCE
    )
endfunction()

function(add_dependent_ops dependent_ops)
  foreach(dep_op ${dependent_ops})
    # 查询依赖算子所在目录
    set(dep_op_path_list "")
    if(ENABLE_EXPERIMENTAL)
      foreach(ops_category ${OPS_CATEGORY_LIST})
        file(GLOB dep_op_path "${PROJECT_SOURCE_DIR}/experimental/${ops_category}/${dep_op}")
        list(APPEND dep_op_path_list ${dep_op_path})
      endforeach()
    endif()
    set(outside_experimental FALSE)
    # 如果非 experimental，或 experimental 下没找到，则去常规目录查找
    if(NOT dep_op_path_list)
      foreach(ops_category ${OPS_CATEGORY_LIST})
        file(GLOB dep_op_path "${PROJECT_SOURCE_DIR}/${ops_category}/${dep_op}")
        list(APPEND dep_op_path_list ${dep_op_path})
      endforeach()
      set(outside_experimental ${ENABLE_EXPERIMENTAL})
    endif()
    # 检查依赖存在
    if(NOT dep_op_path_list)
      message(FATAL_ERROR "dependent operator(${dep_op}) not exists")
    endif()
    list(LENGTH ${dep_op_path_list} find_dep_ops_count)
    if(find_dep_ops_count GREATER 1)
      message(FATAL_ERROR "dependent operator(${dep_op}) is not unique, the found operators:${dep_op_path_list}")
    endif()

    # NEED_COMPILE_OPS 为空表示全部编译，则不需要特意添加目录；但指定experimental时未扫描常规算子，如果依赖常规算子，需要添加目录
    # 已在待编译列表，则不需要加入
    # 已在编译列表，则不需要重复加入
    if((NEED_COMPILE_OPS OR outside_experimental)
        AND (NOT (dep_op IN_LIST NEED_COMPILE_OPS))
        AND (NOT (dep_op IN_LIST COMPILED_OPS))
    )
      # 加入依赖并去重
      add_need_compile_ops("${dep_op}")
      # 添加目录
      get_filename_component(dep_op_path "${dep_op_path_list}" ABSOLUTE)
      get_filename_component(dep_op_parent "${dep_op_path}" DIRECTORY)
      get_filename_component(parent_path "${dep_op_parent}" NAME)
      message(STATUS "add dependent operator: ${parent_path}/${dep_op}, path: ${dep_op_path}")
      add_subdirectory("${dep_op_path}" "${CMAKE_BINARY_DIR}/dependent-ops/${parent_path}/${dep_op}")
    endif()
  endforeach()
endfunction()

# 从两个长度一致的列表中查找相同位置的元素
function(find_value_by_key key_list value_list search_key result)
  list(LENGTH key_list key_list_length)
  list(LENGTH value_list value_list_length)
  if(NOT ${key_list_length} EQUAL ${value_list_length})
    message(FATAL_ERROR "key_list length is ${key_list_length}, value_list length is ${value_list_length}, not equal")
  endif()
  set(found_value "")
  if(key_list_length GREATER 0)
    list(FIND key_list ${search_key} index)
    if(NOT ${index} EQUAL -1)
      list(GET value_list ${index} found_value)
    endif()
  endif()
  set(${result} ${found_value} PARENT_SCOPE)
endfunction()

function(add_tiling_sources tiling_dir disable_in_opp)
  if(NOT disable_in_opp)
    set(disable_in_opp FALSE)
  endif()
  if(NOT BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG AND disable_in_opp)
    message(STATUS "don't need add tiling sources")
    return()
  endif()

  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  if("${tiling_dir}" STREQUAL "")
    file(GLOB OPTILING_SRCS ${SOURCE_DIR}/op_host/*_tiling*.cpp)
  else()
    file(GLOB OPTILING_SRCS ${SOURCE_DIR}/op_host/*_tiling*.cpp ${SOURCE_DIR}/op_host/${tiling_dir}/*_tiling*.cpp)
  endif()

  if(OPTILING_SRCS)
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${OPTILING_SRCS})
  endif()
endfunction()

# usage: add_all_modules_sources(OPTYPE ACLNNTYPE DEPENDENCIES COMPUTE_UNIT TILING_DIR DISABLE_IN_OPP)
# ACLNNTYPE 支持类型aclnn/aclnn_inner/aclnn_exclude
# OPTYPE 和 ACLNNTYPE 需一一对应
# DEPENDENCIES 指定依赖的算子名称列表，如果开启 experimental，则会优先加载 experimental 下的算子
# COMPUTE_UNIT 设置支持芯片版本号，必须与TILING_DIR一一对应，示例：ascend910b ascend950
# TILING_DIR 设置所支持芯片类型对应的tiling文件目录，必须与COMPUTE_UNIT一一对应，示例：arch32 arch35
# DISABLE_IN_OPP 设置是否在opp包中编译tiling文件，布尔类型：TRUE，FALSE
macro(add_all_modules_sources)
  set(oneValueArgs DISABLE_IN_OPP HOSTCPU)
  set(multiValueArgs OPTYPE ACLNNTYPE DEPENDENCIES COMPUTE_UNIT TILING_DIR)

  cmake_parse_arguments(MODULE "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  add_opbase_modules()
  # opapi l0 默认全部编译
  file(GLOB OPAPI_L0_SRCS ${SOURCE_DIR}/op_api/*.cpp)
  list(FILTER OPAPI_L0_SRCS EXCLUDE REGEX ".*/aclnn_[^/]*$")
  if(OPAPI_L0_SRCS)
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_L0_SRCS})
  endif()

  # 获取算子层级目录名称，判断是否编译该算子
  get_filename_component(OP_NAME ${SOURCE_DIR} NAME)
  if(NEED_COMPILE_OPS AND (NOT OP_NAME IN_LIST NEED_COMPILE_OPS))
    # NEED_COMPILE_OPS 为空表示全部编译
    return()
  endif()

  if(OP_NAME IN_LIST COMPILED_OPS)
    # 已经编译过，忽略
    message(STATUS "already compiled ${OP_NAME}, skip")
    return()
  endif()

  # 记录全局的COMPILED_OPS和COMPILED_OP_DIRS，其中COMPILED_OP_DIRS只记录到算子名，例如math/abs
  set(COMPILED_OPS
      ${COMPILED_OPS} ${OP_NAME}
      CACHE STRING "Compiled Ops" FORCE
    )
  set(COMPILED_OP_DIRS
      ${COMPILED_OP_DIRS} ${SOURCE_DIR}
      CACHE STRING "Compiled Ops Dirs" FORCE
    )

  # 添加依赖算子
  add_dependent_ops("${MODULE_DEPENDENCIES}")

  file(GLOB OPAPI_HEADERS ${SOURCE_DIR}/op_api/aclnn_*.h)
  if(OPAPI_HEADERS)
    target_sources(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE ${OPAPI_HEADERS})
  endif()

  file(GLOB OPAPI_L2_SRCS ${SOURCE_DIR}/op_api/aclnn_*.cpp)
  if(OPAPI_L2_SRCS)
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_L2_SRCS})
  endif()

  file(GLOB OPINFER_SRCS ${SOURCE_DIR}/op_host/*_infershape*.cpp)
  if(OPINFER_SRCS)
    add_infer_modules()
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${OPINFER_SRCS})
  endif()

  # 添加tiling文件
  find_value_by_key("${MODULE_COMPUTE_UNIT}" "${MODULE_TILING_DIR}" "${ASCEND_COMPUTE_UNIT}" TILING_SOC_DIR)
  add_tiling_sources("${TILING_SOC_DIR}" "${MODULE_DISABLE_IN_OPP}")

  file(GLOB AICPU_SRCS ${SOURCE_DIR}/op_kernel_aicpu/*_aicpu.cpp)
  file(GLOB AICPU_OP_DEF_SRCS ${SOURCE_DIR}/op_kernel_aicpu/*_aicpu_def.cpp)
  file(GLOB AICPU_JSON_FILES ${SOURCE_DIR}/op_kernel_aicpu/*.json)
  if(AICPU_OP_DEF_SRCS)
    set_property(GLOBAL APPEND PROPERTY AICPU_OPDEF_FILES ${AICPU_OP_DEF_SRCS})
  endif()
  if(AICPU_JSON_FILES)
    set_property(GLOBAL APPEND PROPERTY AICPU_JSON_FILES ${AICPU_JSON_FILES})
  endif()
  if(ENABLE_CUSTOM AND AICPU_SRCS AND NOT DISABLE_AICPU)
    if(NOT BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
      add_aicpu_kernel_modules()
      target_sources(${OPHOST_NAME}_aicpu_obj PRIVATE ${AICPU_SRCS})
    else()
      add_aicpu_cust_kernel_modules(${OP_NAME} "${AICPU_SRCS}")
    endif()
  endif()

  # Host-side constant folding: compile aicpu source as x86 OBJECT for libopconstant_folding_math.so
  # Only enabled when HOSTCPU TRUE is passed to add_all_modules_sources
  if(MODULE_HOSTCPU AND BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG AND NOT ENABLE_CUSTOM)
    file(GLOB AICPU_HOST_SRCS ${SOURCE_DIR}/op_kernel_aicpu/*_aicpu.cpp)
    if(AICPU_HOST_SRCS AND NOT DISABLE_AICPU)
      set(HOST_OBJ_NAME ${OP_NAME}_host_const_obj)
      add_aicpu_host_kernel_modules(${HOST_OBJ_NAME})
      target_sources(${HOST_OBJ_NAME} PRIVATE ${AICPU_HOST_SRCS})
    endif()
  endif()

  if(MODULE_OPTYPE)
    list(LENGTH MODULE_OPTYPE OpTypeLen)
    list(LENGTH MODULE_ACLNNTYPE AclnnTypeLen)
    if(NOT ${OpTypeLen} EQUAL ${AclnnTypeLen})
      message(FATAL_ERROR "OPTYPE AND ACLNNTYPE Should be One-to-One")
    endif()
    math(EXPR index "${OpTypeLen} - 1")
    foreach(i RANGE ${index})
      list(GET MODULE_OPTYPE ${i} OpType)
      list(GET MODULE_ACLNNTYPE ${i} AclnnType)
      if(${AclnnType} STREQUAL "aclnn"
         OR ${AclnnType} STREQUAL "aclnn_inner"
         OR ${AclnnType} STREQUAL "aclnn_exclude"
        )
        file(GLOB OPDEF_SRCS ${SOURCE_DIR}/op_host/${OpType}_def*.cpp)
        add_ext_def_sources()
        if(OPDEF_SRCS)
          target_sources(${OPHOST_NAME}_opdef_${AclnnType}_obj INTERFACE ${OPDEF_SRCS})
        endif()
      else()
        message(FATAL_ERROR "ACLNN TYPE UNSPPORTED, ONLY SUPPORT aclnn/aclnn_inner/aclnn_exclude")
      endif()
    endforeach()
  else()
    file(GLOB OPDEF_SRCS ${SOURCE_DIR}/op_host/*_def*.cpp)
    add_ext_def_sources()
    if(OPDEF_SRCS)
      message(
        FATAL_ERROR
          "Should Manually specify aclnn/aclnn_inner/aclnn_exclude\n"
          "usage: add_all_modules_sources(OPTYPE optypes ACLNNTYPE aclnntypes)\n"
          "example: add_all_modules_sources(OPTYPE add ACLNNTYPE aclnn_exclude)"
        )
    endif()
  endif()

  file(GLOB OP_GRAPH_SRCS ${SOURCE_DIR}/op_graph/*_graph_*.cpp ${SOURCE_DIR}/op_graph/*_fallback.cpp ${SOURCE_DIR}/op_graph/fusion_pass/*_pass.cpp)
  if(OP_GRAPH_SRCS)
    add_op_graph_modules()
    target_sources(${GRAPH_PLUGIN_NAME}_obj PRIVATE ${OP_GRAPH_SRCS})
  endif()

  file(GLOB OP_GRAPH_PROTO_HEADERS ${SOURCE_DIR}/op_graph/*_proto*.h)
  if(OP_GRAPH_PROTO_HEADERS)
    target_sources(${GRAPH_PLUGIN_NAME}_proto_headers INTERFACE ${OP_GRAPH_PROTO_HEADERS})
  endif()

  #添加plugin文件
  file(GLOB ONNX_PLUGIN_SRCS ${SOURCE_DIR}/framework/*_onnx_plugin.cpp)
  if (ONNX_PLUGIN_SRCS)
    add_onnx_plugin_modules()
    target_sources(${ONNX_PLUGIN_NAME}_obj PRIVATE ${ONNX_PLUGIN_SRCS})
  endif()

  # 添加tf_plugin文件
  file(GLOB TF_PLUGIN_SRCS ${SOURCE_DIR}/framework/*_tf_plugin.cpp)
  if (TF_PLUGIN_SRCS)
    if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG AND NOT ENABLE_TEST)
      target_sources(${TF_PLUGIN_NAME}_obj PRIVATE ${TF_PLUGIN_SRCS})
    endif()
  endif()

  # 添加所有的UT
  add_all_ut_sources(UT_TILING_DIR "${TILING_SOC_DIR}" OP_NAME ${OP_NAME})
endmacro()

# usage: add_all_ut_sources()
macro(add_all_ut_sources)
  set(oneValueArgs UT_TILING_DIR OP_NAME)
  cmake_parse_arguments(MODULE "" "${oneValueArgs}" "" ${ARGN})
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  if(UT_TEST_ALL OR OP_API_UT)
    add_modules_ut_sources(UT_NAME ${OP_API_MODULE_NAME} MODE PRIVATE DIR ${SOURCE_DIR}/tests/ut/op_api TILING_DIR ${MODULE_UT_TILING_DIR})
  endif()

  if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${SOURCE_DIR}/tests/ut/op_host TILING_DIR ${MODULE_UT_TILING_DIR})
    add_modules_ut_sources(UT_NAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${SOURCE_DIR}/tests/ut/op_host TILING_DIR ${MODULE_UT_TILING_DIR})
  endif()

  if(UT_TEST_ALL OR OP_GRAPH_UT)
    add_modules_ut_sources(UT_NAME ${OP_GRAPH_MODULE_NAME} MODE PRIVATE DIR ${SOURCE_DIR}/tests/ut/op_graph TILING_DIR ${MODULE_UT_TILING_DIR})
  endif()

  if(UT_TEST_ALL OR OP_KERNEL_AICPU_UT)
    add_aicpu_op_test_case(${MODULE_OP_NAME})
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/CMakeLists.txt")
    add_subdirectory(${SOURCE_DIR}/op_kernel)
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/ut/op_kernel/CMakeLists.txt")
    add_subdirectory(${SOURCE_DIR}/tests/ut/op_kernel)
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/ut/op_kernel_aicpu/CMakeLists.txt")
    add_subdirectory(${SOURCE_DIR}/tests/ut/op_kernel_aicpu)
  endif()
endmacro()

# ######################################################################################################################
# get op_type from *_def.cpp
# ######################################################################################################################
function(get_op_type_from_op_name OP_NAME OP_TYPE)
  execute_process(
    COMMAND
      find ${OP_DIR} -name ${OP_NAME}_def.cpp -exec grep OP_ADD {} \;
    OUTPUT_VARIABLE op_type
    )
  if(NOT op_type)
    set(op_type "")
  else()
    string(REGEX REPLACE "[\t ]*OP_ADD\\([\t ]*" "" op_type ${op_type})
    string(REGEX REPLACE "[\t ]*\\).*$" "" op_type ${op_type})
  endif()
  set(${OP_TYPE}
      ${op_type}
      PARENT_SCOPE
    )
endfunction()

# usage: add_kernel_sources(OPTYPE)
# 1.调用asc_opc 工具 编译二进制kernel时， --simplified_key_mode/--impl_mode 选项中填写的值。
# 2.调用asc_opc工具 算子名.py 中auto_sync与compile_option中的配置项。
# 格式如下所示：
# [KERNEL_SRC op_name.cpp]          算子入口文件，缺省为算子名.cpp
# [COMPUTE_UNITS ascendxx]          soc版本，支持配置多个unit，缺省为配置所有的soc（支持单独配置soc，单独配置优先级更高）
# [SIMPLIFIED_KEY 0/None]           缺省为0，最终的编译参数则是--simplified_key_mode=0，若设置为None，则不会携带该参数
# [AUTO_SYNC false]                 同步选项，缺省为true
# [IMPL_MODE high_performance]      高性能模式，缺省为high_performance[,optional] [optional]是可选的 
# [OPTIONS "option1" "option2"]     其他编译选项
function(add_kernel_sources)
  set(oneValueArgs KERNEL_SRC SIMPLIFIED_KEY AUTO_SYNC IMPL_MODE)
  set(multiValueArgs COMPUTE_UNITS OPTIONS)
  cmake_parse_arguments(MODULE "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
  get_filename_component(PARENT_DIR ${SOURCE_DIR} DIRECTORY)
  get_filename_component(OP_NAME ${PARENT_DIR} NAME)
  set(OP_DIR ${PARENT_DIR})
  set(op_type "")
  get_op_type_from_op_name("${OP_NAME}" op_type)

  set(_kernel_src_provided FALSE)
  if(NOT MODULE_KERNEL_SRC)
    set(MODULE_KERNEL_SRC "${OP_NAME}")
  else()
    string(REGEX REPLACE "\\.cpp$" "" MODULE_KERNEL_SRC "${MODULE_KERNEL_SRC}")
    set(_kernel_src_provided TRUE)
  endif()

  set(simplified_key ${MODULE_SIMPLIFIED_KEY})
  set(auto_sync ${MODULE_AUTO_SYNC})
  set(impl_mode ${MODULE_IMPL_MODE})
  set(options ${MODULE_OPTIONS})

  if(NOT simplified_key)
    set(simplified_key "0")
  else()
    string(TOLOWER "${simplified_key}" simplified_key_lower)
    if(simplified_key_lower STREQUAL "none")
      set(simplified_key "None")
    endif()
  endif()

  if(NOT impl_mode)
    set(impl_mode "high_performance,optional")
  endif()

  set(auto_sync_option "")
  string(TOLOWER "${auto_sync}" auto_sync)
  if(auto_sync STREQUAL "false")
    set(auto_sync_option ${auto_sync})
  endif()

  string(JOIN "," unit_options ${options})

  if(NOT MODULE_COMPUTE_UNITS OR MODULE_COMPUTE_UNITS STREQUAL "")
    set(target_compute_units "ALL")
  else()
    set(target_compute_units "${ASCEND_COMPUTE_UNIT}")
  endif()

  if(NOT MODULE_COMPUTE_UNITS OR "${target_compute_units}" IN_LIST MODULE_COMPUTE_UNITS)
    if(_kernel_src_provided)
      list(APPEND KERNEL_SRC_LIST "${op_type} ${target_compute_units} ${MODULE_KERNEL_SRC}")
      set(KERNEL_SRC_LIST ${KERNEL_SRC_LIST} CACHE INTERNAL "kernel_src")
    endif()
    list(APPEND SIMPLIFIED_KEY_LIST "${op_type} ${target_compute_units} ${simplified_key}")
    list(APPEND IMPL_MODE_LIST "${op_type} ${target_compute_units} ${impl_mode}")
    set(SIMPLIFIED_KEY_LIST ${SIMPLIFIED_KEY_LIST} CACHE INTERNAL "simplified_key")
    set(IMPL_MODE_LIST ${IMPL_MODE_LIST} CACHE INTERNAL "impl_mode")

    if(auto_sync_option STREQUAL "false")
      list(APPEND AUTO_SYNC_LIST "${op_type} ${target_compute_units} ${auto_sync_option}")
      set(AUTO_SYNC_LIST ${AUTO_SYNC_LIST} CACHE INTERNAL "auto_sync")
    endif()
    if(unit_options)
      list(APPEND OPTION_LIST "${op_type} ${target_compute_units} ${unit_options}")
      set(OPTION_LIST ${OPTION_LIST} CACHE INTERNAL "options")
    endif()
  endif()
endfunction()

# usage: add_graph_plugin_sources()
macro(add_graph_plugin_sources)
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  # 获取算子层级目录名称，判断是否编译该算子
  get_filename_component(PARENT_DIR ${SOURCE_DIR} DIRECTORY)
  get_filename_component(OP_NAME ${PARENT_DIR} NAME)
  if(NEED_COMPILE_OPS AND (NOT ${OP_NAME} IN_LIST NEED_COMPILE_OPS))
    return()
  endif()

  file(GLOB OP_GRAPH_SRCS ${SOURCE_DIR}/*_graph_*.cpp ${SOURCE_DIR}/*_fallback.cpp ${SOURCE_DIR}/fusion_pass/*_pass.cpp)

  if(OP_GRAPH_SRCS)
    add_op_graph_modules()
    target_sources(${GRAPH_PLUGIN_NAME}_obj PRIVATE ${OP_GRAPH_SRCS})
  endif()

  file(GLOB OP_GRAPH_PROTO_HEADERS ${SOURCE_DIR}/*_proto*.h)
  if(OP_GRAPH_PROTO_HEADERS)
    target_sources(${GRAPH_PLUGIN_NAME}_proto_headers INTERFACE ${OP_GRAPH_PROTO_HEADERS})
  endif()
endmacro()

# ######################################################################################################################
# get operating system info
# ######################################################################################################################
function(get_system_info SYSTEM_INFO)
  if(UNIX)
    execute_process(
      COMMAND
        grep -i ^id= /etc/os-release
      OUTPUT_VARIABLE TEMP
      )
    string(REGEX REPLACE "\n|id=|ID=|\"" "" SYSTEM_NAME ${TEMP})
    set(${SYSTEM_INFO}
        ${SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}
        PARENT_SCOPE
      )
  elseif(WIN32)
    message(STATUS "System is Windows. Only for pre-build.")
  else()
    message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} not support.")
  endif()
endfunction()

# ######################################################################################################################
# add compile options, e.g.: -g -O0
# ######################################################################################################################
function(add_ops_compile_options OP_TYPE)
  cmake_parse_arguments(OP_COMPILE "" "OP_TYPE" "COMPUTE_UNIT;OPTIONS" ${ARGN})
  execute_process(
    COMMAND
      ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/util/ascendc_gen_options.py
      ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS} ${OP_TYPE} ${OP_COMPILE_COMPUTE_UNIT} ${OP_COMPILE_OPTIONS}
    RESULT_VARIABLE EXEC_RESULT
    OUTPUT_VARIABLE EXEC_INFO
    ERROR_VARIABLE EXEC_ERROR
    )
  if(${EXEC_RESULT})
    message("add ops compile options info: ${EXEC_INFO}")
    message("add ops compile options error: ${EXEC_ERROR}")
    message(FATAL_ERROR "add ops compile options failed!")
  endif()
endfunction()

# ######################################################################################################################
# check whether the compiled operators meet expectations
# ######################################################################################################################
function(check_compiled_ops)
  message(STATUS "Ops for this compilation contains: ${COMPILED_OPS}")
  if(COMPILED_OPS STREQUAL "")
    message(FATAL_ERROR "Specified ops not found in this depository, please check --ops paramater")
  endif()

  # 未指定算子，全部编译
  if(NOT NEED_COMPILE_OPS)
    return()
  endif()

  # 指定了但未参与编译的算子，即为无效算子名，应该报错
  set(not_compiled_ops)
  foreach(op_name IN LISTS NEED_COMPILE_OPS)
    if(NOT op_name IN_LIST COMPILED_OPS)
      list(APPEND not_compiled_ops ${op_name})
    endif()
  endforeach()

  if(NOT not_compiled_ops)
    return()
  endif()

  list(JOIN not_compiled_ops "," not_compiled_ops_str)
  if(ENABLE_EXPERIMENTAL)
    message(FATAL_ERROR
      "Specified ops(${not_compiled_ops_str}) not found in experimental, please check --ops paramater"
    )
  else()
    message(FATAL_ERROR
      "Specified ops(${not_compiled_ops_str}) not found in this depository, please check --ops paramater"
    )
  endif()
endfunction()

function(protobuf_generate_external comp c_var h_var)
  if (NOT ARGN)
    message(SEND_ERROR "Error: protobuf_generate() called without any proto files")
    return()
  endif()

  set(${c_var})
  set(${h_var})
  set(_add_target FALSE)

  set(extra_option "")
  foreach(arg ${ARGN})
    if ("${arg}" MATCHES "--proto_path")
      set(extra_option ${arg})
    endif()
  endforeach()

  foreach(file ${ARGN})
    if ("${file}" STREQUAL "TARGET")
      set(_add_target TRUE)
      continue()
    endif()

    if ("${file}" MATCHES "--proto_path")
      continue()
    endif()

    get_filename_component(abs_file ${file} ABSOLUTE)
    get_filename_component(file_name ${file} NAME_WE)
    get_filename_component(file_dir ${abs_file} PATH)
    get_filename_component(parent_subdir ${file_dir} NAME)

    if ("${parent_subdir}" STREQUAL "proto")
      set(proto_output_path ${CMAKE_BINARY_DIR}/proto/${comp}/proto)
    else()
      set(proto_output_path ${CMAKE_BINARY_DIR}/proto/${comp}/proto/${parent_subdir})
    endif()
    list(APPEND ${c_var} "${proto_output_path}/${file_name}.pb.cc")
    list(APPEND ${h_var} "${proto_output_path}/${file_name}.pb.h")

    add_custom_command(
      OUTPUT "${proto_output_path}/${file_name}.pb.cc" "${proto_output_path}/${file_name}.pb.h"
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      COMMAND ${CMAKE_COMMAND} -E make_directory "${proto_output_path}"
      COMMAND ${CMAKE_COMMAND} -E echo "generate proto cpp_out ${comp} by ${abs_file}"
      COMMAND ${Protobuf_PROTOC_EXECUTABLE} -I${file_dir} ${extra_option} --cpp_out=${proto_output_path} ${abs_file}
      DEPENDS ${abs_file} host_protoc
      COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM)
  endforeach()

    if (_add_target)
      add_custom_target(
        ${comp} DEPENDS ${${c_var}} ${${h_var}}) 
    endif()

    set_source_files_properties(${${c_var}} ${${h_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)
endfunction()


function(get_op_type_from_binary_json BINARY_JSON OP_TYPE)
  execute_process(COMMAND grep -w op_type ${BINARY_JSON} OUTPUT_VARIABLE op_type)
  string(REGEX REPLACE "\"op_type\"" "" op_type ${op_type})
  string(REGEX MATCH "\".+\"" op_type ${op_type})
  string(REGEX REPLACE "\"" "" op_type ${op_type})

  set(${OP_TYPE} ${op_type} PARENT_SCOPE)
endfunction()

macro(add_onnx_plugin_sources)
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  file(GLOB ONNX_PLUGIN_SRCS ${SOURCE_DIR}/*_onnx_plugin.cpp)
  if (ONNX_PLUGIN_SRCS)
    add_onnx_plugin_modules()
    target_sources(${ONNX_PLUGIN_NAME}_obj PRIVATE ${ONNX_PLUGIN_SRCS})
  else()
    message(WARNING "No onnx plugin source files found in ${SOURCE_DIR}")
  endif()
endmacro()

# TF plugin 初始化函数（顶层调用一次，内部条件检查）
function(init_tf_plugin_modules)
  # 条件不满足，直接返回，不创建 OBJECT 库（零浪费）
  if(NOT BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG OR ENABLE_TEST)
    return()
  endif()

  # 条件满足，生成 TF protobuf  
  set(tf_proto_srcs
    ${ASCEND_DIR}/include/proto/ge_ir.proto
  )

  protobuf_generate_external(tf tf_proto_cc tf_proto_h ${tf_proto_srcs})

  # 创建 OBJECT 库（只创建一次）
  add_library(${TF_PLUGIN_NAME}_obj OBJECT ${tf_proto_h})

  # 配置编译选项
  set_target_properties(${TF_PLUGIN_NAME}_obj PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )

  target_include_directories(${TF_PLUGIN_NAME}_obj
    PRIVATE
    ${OP_PROTO_INCLUDE}
    ${HOST_PROTOC_SRC}
    ${HOST_PROTOC_PATH}
    ${PROTOBUF_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}/proto
    ${TF_PLUGIN_COMMON_INCLUDE}
  )

  target_compile_options(${TF_PLUGIN_NAME}_obj
    PRIVATE -Dgoogle=ascend_private
            -fvisibility=hidden
            -Wno-shadow
            -Wno-unused-parameter
  )

  target_link_libraries(${TF_PLUGIN_NAME}_obj
    PRIVATE $<BUILD_INTERFACE:intf_pub_cxx17>
            $<BUILD_INTERFACE:dlog_headers>
            json
  )
endfunction()

# TF plugin 源文件收集宏（各目录调用，内部条件检查）
macro(add_tf_plugin_sources)
  # 条件不满足，直接返回，不添加源文件（零浪费）
  if(NOT BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG OR ENABLE_TEST)
    return()
  endif()

  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  # 扫描源文件
  file(GLOB TF_PLUGIN_SRCS ${SOURCE_DIR}/*_tf_plugin.cpp)
  if (TF_PLUGIN_SRCS)
    # 直接添加源文件，无需检查 TARGET 是否存在（已在阶段1创建）
    target_sources(${TF_PLUGIN_NAME}_obj PRIVATE ${TF_PLUGIN_SRCS})
  else()
    message(WARNING "No TF plugin source files found in ${SOURCE_DIR}")
  endif()
endmacro()


# 当MODULE_EXT存在时，扫描新增的def文件
macro(add_ext_def_sources)
  if(NOT ${MODULE_EXT} STREQUAL "")
    get_filename_component(OP_PARENT_DIR "${SOURCE_DIR}/.." ABSOLUTE)
    get_filename_component(OP_PARENT_NAME ${OP_PARENT_DIR} NAME)
    file(GLOB OPDEF_SRCS_EXT ${MODULE_EXT}/${OP_PARENT_NAME}/${OpType}/op_host/${OpType}_*_def*.cpp )
    list(APPEND OPDEF_SRCS ${OPDEF_SRCS_EXT})
    message(STATUS "OPDEF_SRCS: '${OPDEF_SRCS}'")
  endif()
endmacro()
