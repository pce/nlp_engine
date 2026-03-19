cmake_minimum_required(VERSION 3.25)
project(nlp_engine VERSION 1.0.0 LANGUAGES CXX)

# -----------------------------------------------------------------------------
# C++ Standard Configuration
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------------
# Dependencies (via FetchContent)
# -----------------------------------------------------------------------------
include(FetchContent)

# nlohmann-json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(BUILD_TESTING "Build the test suite" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" OFF)

# CppUTest (Unit Testing & Memory Leak Detection)
if(BUILD_TESTING)
    FetchContent_Declare(
        cpputest
        URL https://github.com/cpputest/cpputest/archive/refs/tags/v4.0.tar.gz
    )
    # Disable CppUTest's own internal testing and extensions we don't need to speed up build
    set(TESTS OFF CACHE BOOL "Disable CppUTest internal tests" FORCE)
    set(EXTENSIONS ON CACHE BOOL "Enable CppUTest extensions" FORCE)
    FetchContent_MakeAvailable(cpputest)
endif()

# -----------------------------------------------------------------------------
# Library: nlp_engine
# -----------------------------------------------------------------------------
add_library(nlp_engine
    nlp/nlp_engine.cpp
    nlp/nlp_engine.hh
    nlp/nlp_engine_async.cpp
    nlp/nlp_engine_async.hh
)

target_include_directories(nlp_engine PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/nlp>
)

# Link JSON
target_link_libraries(nlp_engine PUBLIC nlohmann_json::nlohmann_json)

# -----------------------------------------------------------------------------
# Python Bindings (Optional)
# -----------------------------------------------------------------------------
if(BUILD_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)

    FetchContent_Declare(
        pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.1.tar.gz
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(nlp_engine_python nlp/nlp_engine_wrapper.cpp nlp/nlp_engine_async.cpp)
    target_link_libraries(nlp_engine_python PRIVATE nlp_engine)
    set_target_properties(nlp_engine_python PROPERTIES OUTPUT_NAME nlp_engine)

    message(STATUS "Python bindings: ENABLED")
else()
    message(STATUS "Python bindings: DISABLED")
endif()

# -----------------------------------------------------------------------------
# Resource Management: Shared Data Target
# To avoid race conditions when multiple targets copy the 'data' directory
# simultaneously during parallel builds, we use a single sync target.
# -----------------------------------------------------------------------------
add_custom_target(prepare_data ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/data"
        "${CMAKE_CURRENT_BINARY_DIR}/data"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/data/training"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/data/models"
    COMMENT "Synchronizing linguistic resources and preparing model directories"
)

# -----------------------------------------------------------------------------
# Executables
# -----------------------------------------------------------------------------

# Example: nlp_example_simple
add_executable(nlp_example_simple examples/simple/main.cpp)
target_link_libraries(nlp_example_simple PRIVATE nlp_engine)
add_dependencies(nlp_example_simple prepare_data)

# Trainer CLI: nlp_trainer
add_executable(nlp_trainer training/nlp_trainer.cpp)
target_link_libraries(nlp_trainer PRIVATE nlp_engine)
add_dependencies(nlp_trainer prepare_data)

# Example: nlp_example_train
add_executable(nlp_example_train examples/train/main.cpp)
target_link_libraries(nlp_example_train PRIVATE nlp_engine)
add_dependencies(nlp_example_train prepare_data)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
if(BUILD_TESTING)
    enable_testing()

    # Standard Tests (now using CppUTest)
    add_executable(nlp_tests tests/test_nlp.cpp)
    target_link_libraries(nlp_tests PRIVATE nlp_engine CppUTest CppUTestExt)
    add_dependencies(nlp_tests prepare_data)
    add_test(NAME nlp_engine_test COMMAND nlp_tests)

    # Async & Integration Tests (now using CppUTest)
    add_executable(nlp_tests_async tests/test_nlp_async.cpp)
    target_link_libraries(nlp_tests_async PRIVATE nlp_engine CppUTest CppUTestExt)
    add_dependencies(nlp_tests_async prepare_data)
    add_test(NAME nlp_engine_async_test COMMAND nlp_tests_async)

    # Markov Addon Tests (Professional Suite with CppUTest)
    add_executable(nlp_tests_markov tests/test_nlp_markov.cpp)
    target_link_libraries(nlp_tests_markov PRIVATE nlp_engine CppUTest CppUTestExt)
    target_compile_definitions(nlp_tests_markov PRIVATE CPPUTEST_MEM_LEAK_DETECTION_DISABLED=0)
    add_dependencies(nlp_tests_markov prepare_data)
    add_test(NAME nlp_engine_markov_test COMMAND nlp_tests_markov)
endif()

# -----------------------------------------------------------------------------
# Documentation (Doxygen)
# -----------------------------------------------------------------------------
find_package(Doxygen)

if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    set(DOXYGEN_HTML_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/html)

    if(NOT EXISTS ${DOXYGEN_IN})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs)
        file(WRITE ${DOXYGEN_IN}
            "PROJECT_NAME           = \"@PROJECT_NAME@\"\n"
            "PROJECT_NUMBER         = @PROJECT_VERSION@\n"
            "OUTPUT_DIRECTORY       = @CMAKE_CURRENT_BINARY_DIR@/docs\n"
            "INPUT                  = @CMAKE_CURRENT_SOURCE_DIR@/nlp @CMAKE_CURRENT_SOURCE_DIR@/readme.md\n"
            "USE_MDFILE_AS_MAINPAGE = @CMAKE_CURRENT_SOURCE_DIR@/readme.md\n"
            "RECURSIVE              = YES\n"
            "GENERATE_LATEX         = NO\n"
            "GENERATE_HTML          = YES\n"
            "HTML_OUTPUT            = html\n"
            "EXTRACT_ALL            = YES\n"
            "EXTRACT_PRIVATE        = YES\n"
            "EXTRACT_STATIC         = YES\n"
            "CLASS_DIAGRAMS         = YES\n"
        )
    endif()

    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        COMMAND ${CMAKE_COMMAND} -E touch ${DOXYGEN_HTML_DIR}/.nojekyll
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation (Output in build/docs/html)"
        VERBATIM
    )
endif()

# -----------------------------------------------------------------------------
# Build Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "NLP Engine Build Configuration")
message(STATUS "==============================")
message(STATUS "Project Name:    ${PROJECT_NAME}")
message(STATUS "C++ Standard:    ${CMAKE_CXX_STANDARD}")
message(STATUS "Doxygen Found:   ${DOXYGEN_FOUND}")
message(STATUS "Build Testing:   ${BUILD_TESTING}")
message(STATUS "Python Bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "")
