cmake_minimum_required(VERSION 3.20)
project(rll_policy LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CycloneDDS REQUIRED)
find_package(Torch REQUIRED)
find_package(yaml-cpp REQUIRED)
find_program(IDLC_EXECUTABLE idlc REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
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

add_library(rll_policy_inference STATIC src/inference_engine.cpp)
target_include_directories(rll_policy_inference PUBLIC include ${TORCH_INCLUDE_DIRS})
target_link_libraries(rll_policy_inference PUBLIC ${TORCH_LIBRARIES})

add_library(rll_policy_runtime STATIC src/config.cpp src/runtime_utils.cpp src/dds_transport.cpp)
target_include_directories(rll_policy_runtime PUBLIC include PRIVATE "${GENERATED_DIR}")
target_link_libraries(rll_policy_runtime PUBLIC rll_policy_inference rll_policy_messages yaml-cpp CycloneDDS::ddsc)

add_executable(rll-policy src/main.cpp)
target_link_libraries(rll-policy PRIVATE rll_policy_plugin_loader rll_policy_runtime)

add_library(chocolate_robot_plugin SHARED plugins/chocolate/chocolate_plugin.cpp)
target_include_directories(chocolate_robot_plugin PRIVATE include ${TORCH_INCLUDE_DIRS})
target_link_libraries(chocolate_robot_plugin PRIVATE rll_policy_messages yaml-cpp ${TORCH_LIBRARIES})
install(TARGETS chocolate_robot_plugin LIBRARY DESTINATION lib/rll-policy/plugins)

# Torch is commonly provided by the active Isaac Lab environment rather than
# the system loader path. Preserve that library location for installed users.
get_filename_component(TORCH_PACKAGE_ROOT "${Torch_DIR}/../../.." ABSOLUTE)
set_property(TARGET rll-policy APPEND PROPERTY INSTALL_RPATH "${TORCH_PACKAGE_ROOT}/lib")
set_property(TARGET chocolate_robot_plugin APPEND PROPERTY INSTALL_RPATH "${TORCH_PACKAGE_ROOT}/lib")
install(TARGETS rll-policy RUNTIME DESTINATION bin)
install(FILES config/policy_node.yaml DESTINATION share/rll-policy/config)
install(FILES include/rll_policy/messages.hpp include/rll_policy/processors.hpp
              include/rll_policy/plugin_api.hpp DESTINATION include/rll_policy)

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

  set(FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/fixtures")
  set(FIXTURE_FILES
    "${FIXTURE_DIR}/velocity.pt" "${FIXTURE_DIR}/tracking.pt" "${FIXTURE_DIR}/tuple.pt"
    "${FIXTURE_DIR}/wrong_shape.pt" "${FIXTURE_DIR}/wrong_dtype.pt" "${FIXTURE_DIR}/non_finite.pt"
    "${FIXTURE_DIR}/corrupt.pt")
  add_custom_command(OUTPUT ${FIXTURE_FILES}
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/make_torchscript_fixture.py" "${FIXTURE_DIR}"
    DEPENDS tests/fixtures/make_torchscript_fixture.py VERBATIM)
  add_custom_target(inference_fixtures DEPENDS ${FIXTURE_FILES})
  add_executable(inference_engine_test tests/inference_engine_test.cpp)
  target_include_directories(inference_engine_test PRIVATE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(inference_engine_test PRIVATE rll_policy_inference)
  add_dependencies(inference_engine_test inference_fixtures)
  add_test(NAME inference_engine_test COMMAND inference_engine_test "${FIXTURE_DIR}")

  add_executable(policy_runtime_test tests/policy_runtime_test.cpp)
  target_link_libraries(policy_runtime_test PRIVATE rll_policy_runtime)
  add_test(NAME policy_runtime_test COMMAND policy_runtime_test "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/runtime.yaml")
  add_test(NAME native_sim_interop COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_sim_interop_test.py"
    --binary "$<TARGET_FILE:rll-policy>" --python "${Python3_EXECUTABLE}")
  add_test(NAME install_contract_test COMMAND "${CMAKE_COMMAND}"
    -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    -DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/install_contract_test.cmake")

  add_executable(chocolate_plugin_test tests/chocolate_plugin_test.cpp)
  target_include_directories(chocolate_plugin_test PRIVATE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(chocolate_plugin_test PRIVATE rll_policy_plugin_loader ${TORCH_LIBRARIES})
  add_dependencies(chocolate_plugin_test chocolate_robot_plugin)
  add_test(NAME chocolate_plugin_test COMMAND chocolate_plugin_test "$<TARGET_FILE:chocolate_robot_plugin>")
endif()
