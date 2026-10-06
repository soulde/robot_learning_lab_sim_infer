cmake_minimum_required(VERSION 3.20)
project(rll_policy LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CycloneDDS REQUIRED)
find_package(Torch REQUIRED)
find_package(yaml-cpp REQUIRED)
find_program(IDLC_EXECUTABLE idlc REQUIRED)

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
target_include_directories(rll_policy_messages PUBLIC include "${GENERATED_DIR}")
target_link_libraries(rll_policy_messages PUBLIC robot_v1_types)

add_executable(rll-policy src/main.cpp)
target_link_libraries(rll-policy PRIVATE rll_policy_messages)

include(CTest)
if(BUILD_TESTING)
  add_executable(dds_contract_test tests/dds_contract_test.cpp)
  target_link_libraries(dds_contract_test PRIVATE rll_policy_messages)
  add_test(NAME dds_contract_test COMMAND dds_contract_test)
endif()
