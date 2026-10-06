cmake_minimum_required(VERSION 3.20)
project(rll_policy LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CycloneDDS REQUIRED)
find_package(Torch REQUIRED)
find_package(yaml-cpp REQUIRED)
find_program(IDLC_EXECUTABLE idlc REQUIRED)
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
add_compile_options(${TORCH_CXX_FLAGS_LIST})

set(IDL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../src/robot_learning_lab_sim_infer/dds/idl/robot_v1.idl")
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GENERATED_DIR}")
add_custom_command(OUTPUT "${GENERATED_DIR}/robot_v1.c" "${GENERATED_DIR}/robot_v1.h"
  COMMAND "${CMAKE_COMMAND}" -E chdir "${GENERATED_DIR}" "${IDLC_EXECUTABLE}" "${IDL_FILE}"
  DEPENDS "${IDL_FILE}" VERBATIM)
add_library(robot_v1_types STATIC "${GENERATED_DIR}/robot_v1.c")
target_include_directories(robot_v1_types PUBLIC "${GENERATED_DIR}")
target_link_libraries(robot_v1_types PUBLIC CycloneDDS::ddsc)

add_library(rll_policy_messages STATIC src/messages.cpp)
target_include_directories(rll_policy_messages PUBLIC include PRIVATE "${GENERATED_DIR}")
target_link_libraries(rll_policy_messages PRIVATE robot_v1_types)

add_library(rll_policy_plugin_loader STATIC src/plugin_loader.cpp)
target_include_directories(rll_policy_plugin_loader PUBLIC include ${TORCH_INCLUDE_DIRS})
target_link_libraries(rll_policy_plugin_loader PUBLIC rll_policy_messages ${TORCH_LIBRARIES} ${CMAKE_DL_LIBS})

add_executable(rll-policy src/main.cpp)
target_link_libraries(rll-policy PRIVATE rll_policy_messages)

include(CTest)
if(BUILD_TESTING)
  add_executable(dds_contract_test tests/dds_contract_test.cpp)
  target_include_directories(dds_contract_test PRIVATE "${GENERATED_DIR}")
  target_link_libraries(dds_contract_test PRIVATE rll_policy_messages)
  add_test(NAME dds_contract_test COMMAND dds_contract_test)

  add_library(test_robot_plugin SHARED tests/test_robot_plugin.cpp)
  target_include_directories(test_robot_plugin PRIVATE include ${TORCH_INCLUDE_DIRS})
  target_compile_definitions(test_robot_plugin PRIVATE RLL_PLUGIN_ABI=1)
  target_link_libraries(test_robot_plugin PRIVATE rll_policy_messages ${TORCH_LIBRARIES})
  add_library(wrong_abi_plugin SHARED tests/test_robot_plugin.cpp)
  target_include_directories(wrong_abi_plugin PRIVATE include ${TORCH_INCLUDE_DIRS})
  target_compile_definitions(wrong_abi_plugin PRIVATE RLL_PLUGIN_ABI=77)
  target_link_libraries(wrong_abi_plugin PRIVATE rll_policy_messages ${TORCH_LIBRARIES})
  add_library(missing_symbols_plugin SHARED tests/missing_symbols_plugin.cpp)
  add_executable(plugin_loader_test tests/plugin_loader_test.cpp)
  target_include_directories(plugin_loader_test PRIVATE include)
  target_link_libraries(plugin_loader_test PRIVATE rll_policy_plugin_loader ${TORCH_LIBRARIES})
  add_test(NAME plugin_loader_test COMMAND plugin_loader_test
    "$<TARGET_FILE:test_robot_plugin>" "$<TARGET_FILE:missing_symbols_plugin>" "$<TARGET_FILE:wrong_abi_plugin>"
    "${CMAKE_CURRENT_BINARY_DIR}/does-not-exist.so")
endif()
