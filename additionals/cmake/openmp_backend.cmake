cmake_minimum_required(VERSION 3.13)
include_guard(GLOBAL)

include(compiler_flags)

# Custom OpenMP runtime library and include directory
set(JCDP_OPENMP_RUNTIME "" CACHE PATH "Custom OpenMP runtime library.")
set(JCDP_OPENMP_INCLUDE_DIR "" CACHE PATH "Custom OpenMP include directory.")

# Check if the include directory exists if it is defined
if(JCDP_OPENMP_INCLUDE_DIR)
  if(NOT EXISTS ${JCDP_OPENMP_INCLUDE_DIR})
    print_error(
      "Custom OpenMP include directory '${JCDP_OPENMP_INCLUDE_DIR}' not found")
  endif()

  set(CMAKE_INCLUDE_PATH ${JCDP_OPENMP_INCLUDE_DIR} ${CMAKE_INCLUDE_PATH})
endif()

# Check if the OpenMP runtime library exists if it is defined
if(JCDP_OPENMP_RUNTIME)
  if(NOT EXISTS ${JCDP_OPENMP_RUNTIME})
    print_error(
      "Custom OpenMP runtime library '${JCDP_OPENMP_RUNTIME}' not found")
  endif()

  get_filename_component(_omp_lib_dir ${JCDP_OPENMP_RUNTIME} DIRECTORY)
  get_filename_component(_omp_lib ${JCDP_OPENMP_RUNTIME} NAME_WE)
  string(REGEX REPLACE "^lib" "" _omp_lib ${_omp_lib})

  # AppleClang doesn't come with an OpenMP header so use the custom one
  if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(CMAKE_LIBRARY_PATH ${_omp_lib_dir} ${CMAKE_LIBRARY_PATH})
  endif()
endif()

# Use llvm libomp with msvc for OpenMP tasks
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(OpenMP_RUNTIME_MSVC llvm)
endif()

# Find OpenMP
find_package(OpenMP COMPONENTS CXX REQUIRED)

# Override OpenMP_CXX_LIBRARIES and set OpenMP_CXX_LIBRARY_DIR if necessary
if(JCDP_OPENMP_RUNTIME)
  set(OpenMP_CXX_LIBRARIES ${_omp_lib})
  set(OpenMP_CXX_LIBRARY_DIR ${_omp_lib_dir})
endif()

# Separate multiple flags
string(REPLACE " " ";" OpenMP_CXX_FLAGS "${OpenMP_CXX_FLAGS}")

# Specific for nvidia gpus...
if (JCDP_OPENMP_GPU)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda")
endif()

# **************************************************************************** #
# Cleanup
# **************************************************************************** #
if(JCDP_OPENMP_RUNTIME)
  unset(_omp_lib_dir)
  unset(_omp_lib)
endif()

# **************************************************************************** #
# Helper function to apply the OpenMP flags to a target
# **************************************************************************** #
function(jcdp_compile_with_openmp visibility targets)
  set(targets ${targets} ${ARGN})

  # Check visibility
  if(NOT ${visibility} STREQUAL "PRIVATE" AND
     NOT ${visibility} STREQUAL "PUBLIC" AND
     NOT ${visibility} STREQUAL "INTERFACE" AND
     NOT ${visibility} STREQUAL "NONE")
    print_error(
      "First argument must be either "
      "'PRIVATE', 'PUBLIC', 'INTERFACE' or 'NONE'.")
  endif()

  if(${visibility} STREQUAL "NONE")
    set(visibility "")
  endif()

  foreach(tgt ${targets})
    if(NOT TARGET ${tgt})
      print_error("OpenMP compile flags: Target ${tgt} doesn't exist.")
    endif()

    if(OpenMP_CXX_INCLUDE_DIR)
      target_include_directories(
        ${tgt} SYSTEM ${visibility} ${OpenMP_CXX_INCLUDE_DIR})
    endif()

    if(JCDP_USE_OPENMP)
      target_compile_options(${tgt} ${visibility} ${OpenMP_CXX_FLAGS})
    else()
      add_cxx_flag("-Wno-unknown-pragmas" WNO_UNKNOWN_PRAGMAS ${tgt})
    endif()
  endforeach()
endfunction()

# **************************************************************************** #
# Helper function to link the OpenMP runtime against a target
# **************************************************************************** #
function(jcdp_link_openmp_runtime visibility targets)
  set(targets ${targets} ${ARGN})

  # Check visibility
  if(NOT ${visibility} STREQUAL "PRIVATE" AND
     NOT ${visibility} STREQUAL "PUBLIC" AND
     NOT ${visibility} STREQUAL "INTERFACE" AND
     NOT ${visibility} STREQUAL "NONE")
    print_error(
      "First argument must be either "
      "'PRIVATE', 'PUBLIC', 'INTERFACE' or 'NONE'.")
  endif()

  if(${visibility} STREQUAL "NONE")
    set(visibility "")
  endif()

  foreach(tgt ${targets})
    if(NOT TARGET ${tgt})
      print_error("OpenMP runtime linking: Target ${tgt} doesn't exist.")
    endif()

    if(JCDP_USE_OPENMP)
      target_link_libraries(${tgt} ${visibility} ${OpenMP_CXX_LIBRARIES})
      if(OpenMP_CXX_LIBRARY_DIR)
        target_link_directories(${tgt} PRIVATE ${OpenMP_CXX_LIBRARY_DIR})
      endif()
    endif()
  endforeach()
endfunction()
