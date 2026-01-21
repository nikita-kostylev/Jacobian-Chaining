# **************************************************************************** #
# This file is part of the JCDP build system. It provides the main CMake
# entry point.
# **************************************************************************** #

cmake_minimum_required(VERSION 3.25.0)
cmake_policy(VERSION 3.25.0)

# **************************************************************************** #
# Project information
# **************************************************************************** #

# Project description
set(JCDP_PROJECT_DESCRIPTION "Jacobian Chaining via Dynamic Programming")

# Project version
set(JCDP_VER_MAJOR 0)
set(JCDP_VER_MINOR 1)
set(JCDP_VER_PATCH 0)
set(JCDP_VERSION "${JCDP_VER_MAJOR}.${JCDP_VER_MINOR}.${JCDP_VER_PATCH}")

project(JCDP
  LANGUAGES CXX
  DESCRIPTION ${JCDP_PROJECT_DESCRIPTION}
  VERSION ${JCDP_VERSION})

# **************************************************************************** #
# General setup
# **************************************************************************** #
# Require at least C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Library / Archive output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib/")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib/")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin/")

# JCDP folders & files
set(JCDP_CMAKE_DIR "${PROJECT_SOURCE_DIR}/additionals/cmake")
set(JCDP_IWYU_DIR "${PROJECT_SOURCE_DIR}/additionals/iwyu")
set(JCDP_CONFIG_DIR "${PROJECT_SOURCE_DIR}/additionals/configs")
set(JCDP_SCRIPT_DIR "${PROJECT_SOURCE_DIR}/additionals/scripts")

# Include CMake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${JCDP_CMAKE_DIR}")

# Print title
include(print_helper)
print_title("${JCDP_PROJECT_DESCRIPTION}" EMPTY_BEFORE EMPTY_AFTER)

# **************************************************************************** #
# CMake options
# **************************************************************************** #
option(JCDP_USE_OPENMP
  "Whether to use OpenMP." ON)
# option(JCDP_BUILD_DOXYGEN
#   "Whether to build the Doxygen documentation." OFF)
option(JCDP_OPENMP_GPU
  "Whether to offload OpenMP to GPU." ON)

# **************************************************************************** #
# Include some modules (intrinsics, PIC, sanitation, etc.)
# **************************************************************************** #
include(build_types)
include(cxx_standard_library)
include(enable_intrinsics)
include(enable_position_independent_code)
include(enable_sanitation)
include(compiler_flags)
include(include_what_you_use)
include(cpplint)
include(configure_compiler_warnings)
include(openmp_backend)

# **************************************************************************** #
# Find necessary packages
# **************************************************************************** #
# if(JCDP_BUILD_DOXYGEN)
#   find_package(Doxygen REQUIRED dot OPTIONAL_COMPONENTS dia mscgen)
# endif()

# **************************************************************************** #
# JCDP include directories
# **************************************************************************** #
set(JCDP_include_dirs "${PROJECT_SOURCE_DIR}/include")

# Collect IWYU flags
set(JCDP_IWYU_FLAGS "--mapping_file=${JCDP_IWYU_DIR}/jcdp.imp" "--cxx17ns")
set(JCDP_IWYU_COMPILER_FLAGS "--std=c++23")

add_subdirectory(include)

# **************************************************************************** #
# Compile sources
# **************************************************************************** #
add_subdirectory(src)

# **************************************************************************** #
# Install targets
# **************************************************************************** #
install(DIRECTORY ${JCDP_CONFIG_DIR} DESTINATION .)
install(FILES ${JCDP_SCRIPT_DIR}/generate_plots.py DESTINATION .)

# **************************************************************************** #
# Summary
# **************************************************************************** #
print_subtitle("Configuration summary" EMPTY_BEFORE)
include(JCDP_summary)
print_divide(EMPTY_AFTER)
