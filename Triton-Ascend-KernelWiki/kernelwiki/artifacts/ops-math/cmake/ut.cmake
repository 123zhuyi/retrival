# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
include_guard(GLOBAL)
add_cann_third_party(gtest)

set(OP_TILING_MODULE_NAME
    ${PKG_NAME}_op_tiling_ut
    CACHE STRING "op_tiling ut module name" FORCE)
set(OP_INFERSHAPE_MODULE_NAME
    ${PKG_NAME}_op_infershape_ut
    CACHE STRING "op_infershape ut module name" FORCE)

set(OP_API_MODULE_NAME
    ${PKG_NAME}_op_api_ut
    CACHE STRING "op_api ut module name" FORCE)

set(OP_KERNEL_MODULE_NAME
    ${PKG_NAME}_op_kernel_ut
    CACHE STRING "op_kernel ut module name" FORCE)

set(fastOpTestSocVersions
    ""
    CACHE STRING "fastOp Test SocVersions")

set(AICPU_OP_KERNEL_MODULE_NAME
    ${PKG_NAME}_aicpu_op_kernel_ut
    CACHE STRING "aicpu_op_kernel ut module name" FORCE)

set(OP_GRAPH_MODULE_NAME
    ${PKG_NAME}_op_graph_ut
    CACHE STRING "op_graph ut module name" FORCE)

function(add_optiling_ut_modules OP_TILING_MODULE_NAME)
  # add optiling ut common object: math_op_tiling_ut_common_obj
  add_library(${OP_TILING_MODULE_NAME}_common_obj OBJECT)
  file(GLOB OP_TILING_UT_COMMON_SRC ${UT_COMMON_INC}/tiling_context_faker.cpp
       ${UT_COMMON_INC}/tiling_case_executor.cpp)
  target_sources(${OP_TILING_MODULE_NAME}_common_obj
                 PRIVATE ${OP_TILING_UT_COMMON_SRC})
  target_compile_definitions(${OP_TILING_MODULE_NAME}_common_obj
                             PRIVATE BUILD_SOC_VERSION=${ASCEND_COMPUTE_UNIT})

  target_include_directories(
    ${OP_TILING_MODULE_NAME}_common_obj
    PRIVATE ${JSON_INCLUDE_DIR} ${OP_TILING_INCLUDE} ${ASCEND_DIR}/pkg_inc
            ${ASCEND_DIR}/include/base/context_builder)
  target_link_libraries(
    ${OP_TILING_MODULE_NAME}_common_obj
    PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> json GTest::gtest c_sec)

  # add optiling ut cases object: math_op_tiling_ut_cases_obj
  if(NOT TARGET ${OP_TILING_MODULE_NAME}_cases_obj)
    add_library(${OP_TILING_MODULE_NAME}_cases_obj OBJECT ${UT_PATH}/empty.cpp)
  endif()
  target_include_directories(
    ${OP_TILING_MODULE_NAME}_cases_obj
    PRIVATE ${UT_COMMON_INC} ${GTEST_INCLUDE} ${OP_TILING_INCLUDE}
            ${dlog_MATH_INCLUDE_DIR} ${ASCEND_DIR}/include/base/context_builder)
  target_link_libraries(
    ${OP_TILING_MODULE_NAME}_cases_obj
    PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> GTest::gtest)

  # add op tiling ut cases static lib: libmath_op_tiling_ut_cases.a
  add_library(${OP_TILING_MODULE_NAME}_cases STATIC)
  target_link_libraries(
    ${OP_TILING_MODULE_NAME}_cases PRIVATE ${OP_TILING_MODULE_NAME}_common_obj
                                           ${OP_TILING_MODULE_NAME}_cases_obj)
endfunction()

function(add_infershape_ut_modules OP_INFERSHAPE_MODULE_NAME)
  # add opinfershape ut common object: math_op_infershape_ut_common_obj
  add_library(${OP_INFERSHAPE_MODULE_NAME}_common_obj OBJECT)
  file(GLOB OP_INFERSHAPE_UT_COMMON_SRC
       ${UT_COMMON_INC}/infershape_context_faker.cpp
       ${UT_COMMON_INC}/infershape_case_executor.cpp)
  target_sources(${OP_INFERSHAPE_MODULE_NAME}_common_obj
                 PRIVATE ${OP_INFERSHAPE_UT_COMMON_SRC})
  target_include_directories(
    ${OP_INFERSHAPE_MODULE_NAME}_common_obj
    PRIVATE ${ASCEND_DIR}/include/base/context_builder ${ASCEND_DIR}/pkg_inc)
  target_link_libraries(
    ${OP_INFERSHAPE_MODULE_NAME}_common_obj
    PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> GTest::gtest c_sec)

  # add opinfershape ut cases object: math_op_infershape_ut_cases_obj
  if(NOT TARGET ${OP_INFERSHAPE_MODULE_NAME}_cases_obj)
    add_library(${OP_INFERSHAPE_MODULE_NAME}_cases_obj OBJECT
                ${UT_PATH}/empty.cpp)
  endif()
  target_include_directories(
    ${OP_INFERSHAPE_MODULE_NAME}_cases_obj
    PRIVATE ${UT_COMMON_INC} ${GTEST_INCLUDE} ${OP_PROTO_INCLUDE}
            ${ASCEND_DIR}/pkg_inc
            ${ASCEND_DIR}/include/base/context_builder)
  target_link_libraries(
    ${OP_INFERSHAPE_MODULE_NAME}_cases_obj
    PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> GTest::gtest)

  # add op infershape ut cases static lib: libmath_op_infershape_ut_cases.a
  add_library(${OP_INFERSHAPE_MODULE_NAME}_cases STATIC)
  target_link_libraries(
    ${OP_INFERSHAPE_MODULE_NAME}_cases
    PRIVATE ${OP_INFERSHAPE_MODULE_NAME}_common_obj
            ${OP_INFERSHAPE_MODULE_NAME}_cases_obj)
endfunction()

function(add_opapi_ut_modules OP_API_MODULE_NAME)
  # add opapi ut L2 obj
  if(NOT TARGET ${OP_API_MODULE_NAME}_cases_obj)
    add_library(${OP_API_MODULE_NAME}_cases_obj OBJECT)
  endif()
  target_sources(${OP_API_MODULE_NAME}_cases_obj
                 PRIVATE ${UT_PATH}/op_api/stub/opdev/platform.cpp)
  target_include_directories(
    ${OP_API_MODULE_NAME}_cases_obj
    PRIVATE ${JSON_INCLUDE_DIR}
            ${HI_PYTHON_INC_TEMP}
            ${UT_PATH}/op_api/stub
            ${OP_API_UT_COMMON_INC}
            ${ASCEND_DIR}/include
            ${ASCEND_DIR}/include/aclnn
            ${OPAPI_INCLUDE})
  target_link_libraries(
    ${OP_API_MODULE_NAME}_cases_obj
    PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> json GTest::gtest)
endfunction()

function(add_op_graph_ut_modules OP_GRAPH_MODULE_NAME)
  if(TARGET ${GRAPH_PLUGIN_NAME}_obj)
    add_dependencies(${GRAPH_PLUGIN_NAME}_obj build_es_math)
    target_link_libraries(${GRAPH_PLUGIN_NAME}_obj PRIVATE es_math)
  endif()

  add_library(${OP_GRAPH_MODULE_NAME}_cases_obj OBJECT)
  
  target_include_directories(
    ${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE
    ${UT_COMMON_INC}
    ${OP_PROTO_INCLUDE}
    ${CMAKE_BINARY_DIR}/es_packages/include/es_math/
    )
  target_link_libraries(
    ${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
    $<BUILD_INTERFACE:dlog_headers>
    -Wl,--no-as-needed
    metadef
    graph
    GTest::gtest
    register
    ge_compiler
    ge_common
    )

  target_compile_options(
    ${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE
    -fno-access-control
  )

  # add infershape ut static lib
  add_library(${OP_GRAPH_MODULE_NAME}_static_lib STATIC)
  target_link_libraries(
    ${OP_GRAPH_MODULE_NAME}_static_lib PRIVATE
    ${OP_GRAPH_MODULE_NAME}_cases_obj
    ${GRAPH_PLUGIN_NAME}_obj
    es_math
  )
  add_dependencies(${OP_GRAPH_MODULE_NAME}_static_lib build_es_math)
  target_link_libraries(${OP_GRAPH_MODULE_NAME}_static_lib PRIVATE es_math)
endfunction()

function(add_aicpu_opkernel_ut_modules AICPU_OP_KERNEL_MODULE_NAME)
  # add opkernel ut common object: math_aicpu_op_kernel_ut_common_obj
  add_library(${AICPU_OP_KERNEL_MODULE_NAME}_common_obj OBJECT)
  file(GLOB OP_KERNEL_UT_COMMON_SRC ./stub/*.cpp)
  file(GLOB OP_KERNEL_AICPU_UT_UTILS_SRC utils/*.cpp)
  target_sources(${AICPU_OP_KERNEL_MODULE_NAME}_common_obj
                 PRIVATE ${OP_KERNEL_UT_COMMON_SRC})
  target_include_directories(${AICPU_OP_KERNEL_MODULE_NAME}_common_obj
                             PRIVATE ${dlog_MATH_INCLUDE_DIR})
  target_compile_definitions(${AICPU_OP_KERNEL_MODULE_NAME}_common_obj
                             PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
  target_link_libraries(
    ${AICPU_OP_KERNEL_MODULE_NAME}_common_obj
    PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> GTest::gtest c_sec
            Eigen3::Eigen)

  # add opkernel ut cases object: math_aicpu_op_kernel_ut_cases_obj
  if(NOT TARGET ${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj)
    add_library(${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj OBJECT
                ${UT_PATH}/empty.cpp)
  endif()
  target_link_libraries(${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj
                        PRIVATE Eigen3::Eigen gcov -ldl)
  target_sources(${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj
                 PRIVATE ${OP_KERNEL_AICPU_UT_UTILS_SRC})

  # add opkernel ut cases shared lib: libmath_aicpu_op_kernel_ut_cases.so
  add_library(
    ${AICPU_OP_KERNEL_MODULE_NAME}_cases SHARED
    $<TARGET_OBJECTS:${AICPU_OP_KERNEL_MODULE_NAME}_common_obj>
    $<TARGET_OBJECTS:${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj>)

  # 链接静态库时使用 whole-archive，保证 RegistCpuKernel 被拉入
  target_link_libraries(
    ${AICPU_OP_KERNEL_MODULE_NAME}_cases
    PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
            GTest::gtest
            c_sec
            -ldl
            -Wl,--whole-archive
            ${ASCEND_DIR}/lib64/libaicpu_context_host.a
            ${ASCEND_DIR}/lib64/libaicpu_nodedef_host.a
            ${ASCEND_DIR}/lib64/libhost_ascend_protobuf.a
            -Wl,--no-whole-archive
            -Wl,-Bsymbolic
            -Wl,--exclude-libs=libhost_ascend_protobuf.a
            Eigen3::Eigen
            ${AICPU_OP_KERNEL_MODULE_NAME}_common_obj
            ${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj)
endfunction()

function(add_opkernel_ut_modules OP_KERNEL_MODULE_NAME)
  # add opkernel ut common object: math_op_kernel_ut_common_obj
  add_library(${OP_KERNEL_MODULE_NAME}_common_obj OBJECT)
  file(GLOB OP_KERNEL_UT_COMMON_SRC ${UT_COMMON_INC}/tiling_context_faker.cpp
       ${UT_COMMON_INC}/tiling_case_executor.cpp
       ${PROJECT_SOURCE_DIR}/tests/ut/op_kernel/data_utils.cpp)
  target_sources(${OP_KERNEL_MODULE_NAME}_common_obj
                 PRIVATE ${OP_KERNEL_UT_COMMON_SRC})
  target_include_directories(
    ${OP_KERNEL_MODULE_NAME}_common_obj
    PRIVATE ${JSON_INCLUDE_DIR} ${ASCEND_DIR}/include/base/context_builder
            ${ASCEND_DIR}/pkg_inc)
  target_link_libraries(
    ${OP_KERNEL_MODULE_NAME}_common_obj
    PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> json GTest::gtest c_sec)

  foreach(socVersion ${fastOpTestSocVersions})
    # add op kernel ut cases obj: math_op_tiling_ut_${socVersion}_cases
    if(NOT TARGET ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj)
      add_library(${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj OBJECT)
    endif()
    target_link_libraries(${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj
                          PRIVATE gcov)

    # add op kernel ut cases dynamic lib:
    # libmath_op_tiling_ut_${socVersion}_cases.so
    add_library(
      ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases SHARED
      $<TARGET_OBJECTS:${OP_KERNEL_MODULE_NAME}_common_obj>
      $<TARGET_OBJECTS:${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj>)
    target_link_libraries(
      ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases
      PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
              ${OP_KERNEL_MODULE_NAME}_common_obj
              ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj)
  endforeach()
endfunction()

function(add_modules_ut_sources)
  set(options OPTION_RESERVED)
  set(oneValueArgs UT_NAME MODE DIR TILING_DIR)
  set(multiValueArgs MULIT_RESERVED)
  cmake_parse_arguments(MODULE "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  if("${MODULE_UT_NAME}" STREQUAL "${OP_TILING_MODULE_NAME}")
    get_filename_component(UT_DIR ${MODULE_DIR} DIRECTORY)
    get_filename_component(TESTS_DIR ${UT_DIR} DIRECTORY)
    get_filename_component(OP_NAME_DIR ${TESTS_DIR} DIRECTORY)
    get_filename_component(OP_NAME ${OP_NAME_DIR} NAME)
    list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
    # if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL,
    # include all.
    if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
      return()
    endif()

    file(GLOB OPHOST_TILING_CASES_SRC ${MODULE_DIR}/test_*_tiling.cpp
         ${MODULE_DIR}/${MODULE_TILING_DIR}/test_*_tiling*.cpp)
    if(NOT OPHOST_TILING_CASES_SRC)
      return()
    endif()

    if(NOT TARGET ${MODULE_UT_NAME}_cases_obj)
      add_library(${MODULE_UT_NAME}_cases_obj OBJECT)
    endif()
    target_sources(${MODULE_UT_NAME}_cases_obj ${MODULE_MODE}
                                               ${OPHOST_TILING_CASES_SRC})
  endif()

  if("${MODULE_UT_NAME}" STREQUAL "${OP_INFERSHAPE_MODULE_NAME}")
    get_filename_component(UT_DIR ${MODULE_DIR} DIRECTORY)
    get_filename_component(TESTS_DIR ${UT_DIR} DIRECTORY)
    get_filename_component(OP_NAME_DIR ${TESTS_DIR} DIRECTORY)
    get_filename_component(OP_NAME ${OP_NAME_DIR} NAME)
    list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
    # if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL,
    # include all.
    if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
      return()
    endif()

    file(GLOB OPHOST_INFERSHAPE_CASES_SRC ${MODULE_DIR}/test_*_infershape.cpp)
    if(NOT OPHOST_INFERSHAPE_CASES_SRC)
      return()
    endif()

    if(NOT TARGET ${MODULE_UT_NAME}_cases_obj)
      add_library(${MODULE_UT_NAME}_cases_obj OBJECT)
    endif()
    target_sources(${MODULE_UT_NAME}_cases_obj ${MODULE_MODE}
                                               ${OPHOST_INFERSHAPE_CASES_SRC})
  endif()

  if("${MODULE_UT_NAME}" STREQUAL "${OP_API_MODULE_NAME}")
    get_filename_component(OP_HOST_DIR ${MODULE_DIR} DIRECTORY)
    get_filename_component(OP_HOST_NAME ${OP_HOST_DIR} NAME)
    if("${OP_HOST_NAME}" STREQUAL "op_host")
      get_filename_component(UT_DIR ${OP_HOST_DIR} DIRECTORY)
    else()
      get_filename_component(UT_DIR ${MODULE_DIR} DIRECTORY)
    endif()
    get_filename_component(TESTS_DIR ${UT_DIR} DIRECTORY)
    get_filename_component(OP_NAME_DIR ${TESTS_DIR} DIRECTORY)
    get_filename_component(OP_NAME ${OP_NAME_DIR} NAME)
    list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
    # if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL,
    # include all.
    if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
      return()
    endif()

    file(GLOB OPAPI_CASES_SRC ${MODULE_DIR}/test_aclnn_*.cpp)
    if(NOT OPAPI_CASES_SRC)
      return()
    endif()

    if(NOT TARGET ${MODULE_UT_NAME}_cases_obj)
      add_library(${MODULE_UT_NAME}_cases_obj OBJECT)
    endif()
    target_sources(${MODULE_UT_NAME}_cases_obj ${MODULE_MODE}
                                               ${OPAPI_CASES_SRC})
  endif()

  if("${MODULE_UT_NAME}" STREQUAL "${OP_GRAPH_MODULE_NAME}")
    get_filename_component(UT_DIR ${MODULE_DIR} DIRECTORY)
    get_filename_component(TESTS_DIR ${UT_DIR} DIRECTORY)
    get_filename_component(OP_NAME_DIR ${TESTS_DIR} DIRECTORY)
    get_filename_component(OP_NAME ${OP_NAME_DIR} NAME)
    list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
    # if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL, include all.
    if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
      return()
    endif()

    file(GLOB OPGRAPH_CASES_SRC ${MODULE_DIR}/test_*_pass.cpp)
    if(NOT OPGRAPH_CASES_SRC)
      return()
    endif()

    if(NOT TARGET ${MODULE_UT_NAME}_cases_obj)
      add_op_graph_ut_modules(${OP_GRAPH_MODULE_NAME})
    endif()
    target_sources(${MODULE_UT_NAME}_cases_obj ${MODULE_MODE} ${OPGRAPH_CASES_SRC})
  endif()

endfunction()

function(AddOpsTestCase)
  set(options OPTION_RESERVED)
  set(oneValueArgs OP_NAME OTHER_COMPILE_OPTIONS)
  set(multiValueArgs SOC_VERSION TILING_SRC_FILES UT_SRC_FILES)
  cmake_parse_arguments(MODULE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  # check if the case should be executed according to soc
  if(NOT ${ASCEND_COMPUTE_UNIT} IN_LIST MODULE_SOC_VERSION)
    return()
  endif()

  list(FIND ASCEND_OP_NAME ${MODULE_OP_NAME} INDEX)
  # if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL,
  # include all.
  if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
    return()
  endif()

  # find kernel file
  file(
    GLOB KernelFile
    "${PROJECT_SOURCE_DIR}/*/${MODULE_OP_NAME}/op_kernel/${MODULE_OP_NAME}.cpp")

  # standardize opType
  set(opType "")
  string(REPLACE "_" ";" opTypeTemp "${MODULE_OP_NAME}")
  foreach(word IN LISTS opTypeTemp)
    string(SUBSTRING "${word}" 0 1 firstLetter)
    string(SUBSTRING "${word}" 1 -1 restOfWord)
    string(TOUPPER "${firstLetter}" firstLetter)
    string(TOLOWER "${restOfWord}" restOfWord)
    set(opType "${opType}${firstLetter}${restOfWord}")
  endforeach()

  # check KERNEL_SRC_LIST for custom kernel entry file
  set(kernel_src_from_cmake "")
  if(DEFINED KERNEL_SRC_LIST)
    foreach(item IN LISTS KERNEL_SRC_LIST)
      string(REGEX MATCH "^${opType} ${ASCEND_COMPUTE_UNIT} (.+)" matched "${item}")
      if(matched AND CMAKE_MATCH_1)
        set(kernel_src_from_cmake "${CMAKE_MATCH_1}")
        break()
      endif()
      string(REGEX MATCH "^${opType} ALL (.+)" matched "${item}")
      if(matched AND CMAKE_MATCH_1)
        set(kernel_src_from_cmake "${CMAKE_MATCH_1}")
        break()
      endif()
    endforeach()
  endif()

  if(kernel_src_from_cmake)
    file(
      GLOB KernelFile
      "${PROJECT_SOURCE_DIR}/*/${MODULE_OP_NAME}/op_kernel/**/${kernel_src_from_cmake}.cpp")
  endif()

  # standardize tiling files
  string(REPLACE "," ";" tilingSrc "${MODULE_TILING_SRC_FILES}")

  foreach(shortSocVersion ${MODULE_SOC_VERSION})
    # standardize socVersion
    string(TOLOWER ${shortSocVersion} lowerShortSocVersion)
    find_value_by_key("${SHORT_NAME_LIST}" "${FULL_NAME_LIST}"
                      "${lowerShortSocVersion}" oriSocVersion)
    string(REPLACE "ascend" "Ascend" socVersion "${oriSocVersion}")

    # add tiling tmp so: ${MODULE_OP_NAME}_${socVersion}_tiling_tmp.so
    add_library(${MODULE_OP_NAME}_${socVersion}_tiling_tmp SHARED
                ${tilingSrc} $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
                $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
                $<$<TARGET_EXISTS:opbase_tiling_objs>:$<TARGET_OBJECTS:opbase_tiling_objs>>)
    target_include_directories(
      ${MODULE_OP_NAME}_${socVersion}_tiling_tmp
      PRIVATE ${OP_TILING_INCLUDE} ${dlog_MATH_INCLUDE_DIR})
    target_compile_definitions(${MODULE_OP_NAME}_${socVersion}_tiling_tmp
                               PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
    target_link_libraries(
      ${MODULE_OP_NAME}_${socVersion}_tiling_tmp
      PRIVATE -Wl,--whole-archive
              tiling_api
              rt2_registry_static
              -Wl,--no-whole-archive
              register
              opp_registry
              $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
              ${OP_KERNEL_MODULE_NAME}_common_obj)

    # gen ascendc tiling head files
    set(tilingFile ${CMAKE_CURRENT_BINARY_DIR}/${MODULE_OP_NAME}_tiling_data.h)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${MODULE_OP_NAME}_tiling.h")
      set(compileOptions
          -include "${CMAKE_CURRENT_SOURCE_DIR}/${MODULE_OP_NAME}_tiling.h")
    else()
      set(compileOptions -include ${tilingFile})
    endif()
    set(CUSTOM_TILING_DATA_KEYS "")
    string(REGEX MATCH "-DUT_CUSTOM_TILING_DATA_KEYS=([^ ]+)" matchedPart
                 "${MODULE_OTHER_COMPILE_OPTIONS}")
    if(CMAKE_MATCH_1)
      set(CUSTOM_TILING_DATA_KEYS ${CMAKE_MATCH_1})
      string(REGEX REPLACE "-DUT_CUSTOM_TILING_DATA_KEYS=[^ ]+" ""
                           modifiedString ${MODULE_OTHER_COMPILE_OPTIONS})
      set(MODULE_OTHER_COMPILE_OPTIONS ${modifiedString})
    endif()
    string(REPLACE " " ";" options "${MODULE_OTHER_COMPILE_OPTIONS}")
    foreach(option IN LISTS options)
      set(compileOptions ${compileOptions} ${option})
    endforeach()
    message("compileOptions: ${compileOptions}")
    set(gen_tiling_head_file
        ${OPS_MATH_DIR}/tests/ut/op_kernel/scripts/gen_tiling_head_file.sh)
    set(gen_tiling_so_path
        ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_OP_NAME}_${socVersion}_tiling_tmp.so
    )
    set(gen_tiling_head_tag ${MODULE_OP_NAME}_${socVersion}_gen_head)
    set(gen_cmd
        "bash ${gen_tiling_head_file} ${opType} ${MODULE_OP_NAME} ${gen_tiling_so_path} ${CUSTOM_TILING_DATA_KEYS}"
    )
    message("gen tiling head file to ${tilingFile}, command:")
    message("${gen_cmd}")
    add_custom_command(
      OUTPUT ${tilingFile}
      COMMAND rm -f ${tilingFile}
      COMMAND bash -c ${gen_cmd}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS ${MODULE_OP_NAME}_${socVersion}_tiling_tmp)
    add_custom_target(${gen_tiling_head_tag} ALL DEPENDS ${tilingFile})

    # add object: ${MODULE_OP_NAME}_${socVersion}_cases_obj
    if(MODULE_UT_SRC_FILES)
      set(OPKERNEL_CASES_SRC ${MODULE_UT_SRC_FILES})
    else()
      file(GLOB OPKERNEL_CASES_SRC
           ${CMAKE_CURRENT_SOURCE_DIR}/test_${MODULE_OP_NAME}*.cpp)
    endif()
    add_library(${MODULE_OP_NAME}_${socVersion}_cases_obj OBJECT
                ${KernelFile} ${OPKERNEL_CASES_SRC})
    add_dependencies(${MODULE_OP_NAME}_${socVersion}_cases_obj
                     ${gen_tiling_head_tag})
    target_compile_options(
      ${MODULE_OP_NAME}_${socVersion}_cases_obj
      PRIVATE -g ${compileOptions} -DUT_SOC_VERSION="${socVersion}")
    target_include_directories(
      ${MODULE_OP_NAME}_${socVersion}_cases_obj
      PRIVATE
        ${GTEST_INCLUDE}
        ${ASCEND_DIR}/include/base/context_builder
        ${PROJECT_SOURCE_DIR}/tests/ut/op_kernel
        ${PROJECT_SOURCE_DIR}/tests/ut/common
        ${OP_TILING_INCLUDE}
        ${dlog_MATH_INCLUDE_DIR}
        ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/
        ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/impl/basic_api/
        ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include/
        ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include/basic_api/)
    target_link_libraries(
      ${MODULE_OP_NAME}_${socVersion}_cases_obj
      PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
              tikicpulib::${socVersion} GTest::gtest)

    # add object: math_op_kernel_ut_${oriSocVersion}_cases_obj
    if(NOT TARGET ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj)
      add_library(${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj OBJECT
                  $<TARGET_OBJECTS:${MODULE_OP_NAME}_${socVersion}_cases_obj>)
    else()
      target_sources(
        ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj
        PRIVATE $<TARGET_OBJECTS:${MODULE_OP_NAME}_${socVersion}_cases_obj>)
    endif()
    target_link_libraries(
      ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj
      PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
              $<TARGET_OBJECTS:${MODULE_OP_NAME}_${socVersion}_cases_obj>)

    list(FIND fastOpTestSocVersions "${socVersion}" index)
    if(index EQUAL -1)
      set(fastOpTestSocVersions ${fastOpTestSocVersions} ${socVersion}
          CACHE STRING "fastOp Test SocVersions" FORCE)
    endif()
  endforeach()
endfunction()

function(AddOpTestCase opName shortSocVersion otherCompileOptions
         tilingSrcFiles)
  AddOpsTestCase(
    OP_NAME
    ${opName}
    SOC_VERSION
    ${shortSocVersion}
    OTHER_COMPILE_OPTIONS
    "${otherCompileOptions}"
    TILING_SRC_FILES
    ${tilingSrcFiles})
endfunction()

function(add_aicpu_op_test_case opName)
  get_filename_component(UT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

  # find kernel file
  file(GLOB KernelFile
       "${PROJECT_SOURCE_DIR}/*/${opName}/op_kernel_aicpu/${opName}_aicpu.cpp")

  # add object: ${opName}_cases_obj
  file(GLOB OPKERNEL_CASES_SRC
       ${UT_DIR}/${opName}/tests/ut/op_kernel_aicpu/test_${opName}*.cpp)
  message(
    STATUS "aicpu kernel info: ${opName}, ${KernelFile}, ${OPKERNEL_CASES_SRC}")
  if(NOT KernelFile OR NOT OPKERNEL_CASES_SRC)
    return()
  endif()

  add_library(${opName}_cases_obj OBJECT ${KernelFile} ${OPKERNEL_CASES_SRC})
  target_compile_options(${opName}_cases_obj PRIVATE -g)
  # add op_kernel_aicpu test header file search path, so that header files can
  # be referenced based on relative path
  target_include_directories(
    ${opName}_cases_obj
    PRIVATE ${GTEST_INCLUDE} ${AICPU_INCLUDE} ${OPBASE_INC_DIRS}
            ${AICPU_INC_DIRS} ${PROJECT_SOURCE_DIR}/tests/ut/op_kernel_aicpu
            ${dlog_MATH_INCLUDE_DIR})
  target_link_libraries(
    ${opName}_cases_obj PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> -ldl
                                GTest::gtest c_sec Eigen3::Eigen)

  # add object: math_op_kernel_ut_cases_obj
  if(NOT TARGET ${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj)
    add_library(${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj OBJECT
                $<TARGET_OBJECTS:${opName}_cases_obj>)
  else()
    target_sources(${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj
                   PRIVATE $<TARGET_OBJECTS:${opName}_cases_obj>)
  endif()

  target_link_libraries(
    ${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj
    PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> -ldl
            $<TARGET_OBJECTS:${opName}_cases_obj> GTest::gtest c_sec Eigen3::Eigen)
endfunction()
