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
# Library: nlp_engine
# -----------------------------------------------------------------------------
add_library(nlp_engine
    nlp/nlp_engine.cpp
    nlp/nlp_engine.h
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
option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" OFF)

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
# Example: nlp_example_simple
# -----------------------------------------------------------------------------
add_executable(nlp_example_simple examples/simple/main.cpp)
target_link_libraries(nlp_example_simple PRIVATE nlp_engine)

# Copy dictionaries to build directory
add_custom_command(TARGET nlp_example_simple POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/data"
        "$<TARGET_FILE_DIR:nlp_example_simple>/data"
    COMMENT "Copying dictionaries to build directory"
)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
option(BUILD_TESTING "Build the test suite" ON)

if(BUILD_TESTING)
    enable_testing()
    # Standard Tests
    add_executable(nlp_tests tests/test_nlp.cpp)
    target_link_libraries(nlp_tests PRIVATE nlp_engine)
    add_test(NAME nlp_engine_test COMMAND nlp_tests)

    # Async & Integration Tests
    add_executable(nlp_tests_async tests/test_nlp_async.cpp)
    target_link_libraries(nlp_tests_async PRIVATE nlp_engine)
    add_test(NAME nlp_engine_async_test COMMAND nlp_tests_async)

    # Copy data for both test targets
    foreach(test_target nlp_tests nlp_tests_async)
        add_custom_command(TARGET ${test_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/data"
                "$<TARGET_FILE_DIR:${test_target}>/data"
            COMMENT "Copying dictionaries for ${test_target}"
        )
    endforeach()
endif()

# -----------------------------------------------------------------------------
# Documentation (Doxygen)
# -----------------------------------------------------------------------------
find_package(Doxygen)

if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    set(DOXYGEN_HTML_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/html)

    # Create a template Doxyfile.in if it doesn't exist
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
        # CRITICAL: Disable Jekyll for GitHub Pages to show files with underscores
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
message(STATUS "")
