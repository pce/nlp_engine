cmake_minimum_required(VERSION 3.25)
project(nlp_engine VERSION 0.9.0 LANGUAGES CXX)

# -----------------------------------------------------------------------------
# Generate vcpkg.json from template to match project version
# -----------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vcpkg.json.in")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vcpkg.json.in"
        "${CMAKE_CURRENT_BINARY_DIR}/vcpkg.json"
        @ONLY
    )
    message(STATUS "Generated vcpkg.json for version ${PROJECT_VERSION}")
endif()

# -----------------------------------------------------------------------------
# Version Header Generation
# -----------------------------------------------------------------------------
set(VERSION_H_IN "${CMAKE_CURRENT_SOURCE_DIR}/nlp/version.hh.in")
set(VERSION_H_OUT "${CMAKE_CURRENT_BINARY_DIR}/generated/include/nlp/version.hh")

if(NOT EXISTS "${VERSION_H_IN}")
    file(WRITE "${VERSION_H_IN}"
        "#pragma once\n\n"
        "#define NLP_ENGINE_VERSION \"@PROJECT_VERSION@\"\n"
        "#define NLP_ENGINE_VERSION_MAJOR @PROJECT_VERSION_MAJOR@\n"
        "#define NLP_ENGINE_VERSION_MINOR @PROJECT_VERSION_MINOR@\n"
        "#define NLP_ENGINE_VERSION_PATCH @PROJECT_VERSION_PATCH@\n"
    )
endif()

configure_file("${VERSION_H_IN}" "${VERSION_H_OUT}" @ONLY)

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

# simdutf (Fast Unicode Validation & Transcoding)
FetchContent_Declare(
    simdutf
    URL https://github.com/simdutf/simdutf/archive/refs/tags/v5.6.4.tar.gz
)
FetchContent_MakeAvailable(simdutf)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(BUILD_TESTING "Build the test suite" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" OFF)

# Catch2 (Modern C++23 Unit Testing & Memory Safety)
if(BUILD_TESTING)
    FetchContent_Declare(
        Catch2
        URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz
    )
    # Force static linkage for Catch2 to avoid runtime relocation segfaults on macOS with LLVM 22
    set(CATCH_BUILD_STATIC_LIBRARY ON CACHE BOOL "Build Catch2 as a static library" FORCE)
    FetchContent_MakeAvailable(Catch2)
endif()

# -----------------------------------------------------------------------------
# Library: nlp_engine
# -----------------------------------------------------------------------------
add_library(nlp_engine
    nlp/nlp_engine.cpp
    nlp/nlp_engine.hh
    nlp/nlp_engine_async.cpp
    nlp/nlp_engine_async.hh
    nlp/addons/graph_addon.hh
)

target_include_directories(nlp_engine PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/nlp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated/include>
)

# Link JSON and simdutf
# We use BUILD_INTERFACE to ensure that dependencies are only required during the build,
# preventing export errors since the dependencies are managed via FetchContent.
target_link_libraries(nlp_engine PUBLIC
    $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
    $<BUILD_INTERFACE:simdutf::simdutf>
)

# -----------------------------------------------------------------------------
# Python Bindings (Optional)
# -----------------------------------------------------------------------------
if(BUILD_PYTHON_BINDINGS)
    # PkgConfig is often missing on macOS/Windows; handle gracefully
    find_package(PkgConfig QUIET)
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

# Example: nlp_example_train (Offline Training)
add_executable(nlp_example_train examples/train/train.cpp)
target_link_libraries(nlp_example_train PRIVATE nlp_engine)
add_dependencies(nlp_example_train prepare_data)

# Example: nlp_example_textgen (Inference & Benchmarking)
add_executable(nlp_example_textgen examples/train/textgen.cpp)
target_link_libraries(nlp_example_textgen PRIVATE nlp_engine)
add_dependencies(nlp_example_textgen prepare_data)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
if(BUILD_TESTING)
    enable_testing()

    # Standard Tests (Catch2)
    add_executable(nlp_tests tests/test_nlp.cpp)
    target_link_libraries(nlp_tests PRIVATE nlp_engine Catch2::Catch2WithMain)
    add_dependencies(nlp_tests prepare_data)
    add_test(NAME nlp_engine_test COMMAND nlp_tests)

    # Linguistic Precision Tests (Catch2)
    add_executable(nlp_tests_linguistics tests/test_nlp_linguistics.cpp)
    target_link_libraries(nlp_tests_linguistics PRIVATE nlp_engine Catch2::Catch2WithMain)
    add_dependencies(nlp_tests_linguistics prepare_data)
    add_test(NAME nlp_engine_linguistics_test COMMAND nlp_tests_linguistics)

    # Async & Integration Tests (Catch2)
    add_executable(nlp_tests_async tests/test_nlp_async.cpp)
    target_link_libraries(nlp_tests_async PRIVATE nlp_engine Catch2::Catch2WithMain)
    add_dependencies(nlp_tests_async prepare_data)
    add_test(NAME nlp_engine_async_test COMMAND nlp_tests_async)

    # Markov Addon Tests (Catch2)
    add_executable(nlp_tests_markov tests/test_nlp_markov.cpp)
    target_link_libraries(nlp_tests_markov PRIVATE nlp_engine Catch2::Catch2WithMain)
    add_dependencies(nlp_tests_markov prepare_data)
    add_test(NAME nlp_engine_markov_test COMMAND nlp_tests_markov)

    # Graph Addon Tests (Catch2)
    add_executable(nlp_tests_graph tests/test_nlp_graph.cpp)
    target_link_libraries(nlp_tests_graph PRIVATE nlp_engine Catch2::Catch2WithMain)
    add_dependencies(nlp_tests_graph prepare_data)
    add_test(NAME nlp_engine_graph_test COMMAND nlp_tests_graph)

    # Unicode & Multilingual Tests (Catch2)
    add_executable(nlp_tests_unicode tests/test_nlp_unicode.cpp)
    target_link_libraries(nlp_tests_unicode PRIVATE nlp_engine Catch2::Catch2WithMain)
    add_dependencies(nlp_tests_unicode prepare_data)
    add_test(NAME nlp_engine_unicode_test COMMAND nlp_tests_unicode)
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
# Code Style & Linting (clang-format & bun)
# -----------------------------------------------------------------------------
find_program(CLANG_FORMAT_EXE clang-format)
find_program(BUN_EXE bun)

file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
    nlp/*.cpp nlp/*.hh
    tests/*.cpp
    examples/*.cpp
    training/*.cpp
)

if(CLANG_FORMAT_EXE)
    add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXE} -i ${ALL_CXX_SOURCE_FILES}
        COMMENT "Formatting C++ source files with clang-format"
        VERBATIM
    )

    add_custom_target(lint-check
        COMMAND ${CLANG_FORMAT_EXE} --dry-run -Werror ${ALL_CXX_SOURCE_FILES}
        COMMENT "Checking C++ code style with clang-format"
        VERBATIM
    )
else()
    add_custom_target(format
        COMMAND ${CMAKE_COMMAND} -E echo "clang-format not found, skipping format."
    )
    add_custom_target(lint-check
        COMMAND ${CMAKE_COMMAND} -E echo "clang-format not found, skipping lint check."
    )
endif()

if(BUN_EXE)
    # Check if a frontend or bun-based directory exists (e.g., in project root or subdirs)
    # Assuming bun files are in the project or a typical 'frontend' location if applicable
    add_custom_target(format-bun
        COMMAND ${BUN_EXE} format --write .
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Formatting JS/TS/JSON files with bun"
        VERBATIM
    )

    add_custom_target(lint-bun
        COMMAND ${BUN_EXE} lint .
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Linting JS/TS files with bun"
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
message(STATUS "Clang-Format:    ${CLANG_FORMAT_EXE}")
message(STATUS "Bun:             ${BUN_EXE}")
message(STATUS "")
