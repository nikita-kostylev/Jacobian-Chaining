cmake_minimum_required(VERSION 3.25)

# =====================================================================
#  Sources
# =====================================================================

set(JCDP_OFFLOAD_TU branch_and_bound_gpu_offload.cpp)

add_executable(jcdp
  jcdp.cpp
  ${JCDP_OFFLOAD_TU}
)

add_executable(jcdp_batch
  jcdp_batch.cpp
  ${JCDP_OFFLOAD_TU}
)

# =====================================================================
#  Includes (use your project's variable if present, else fallback)
# =====================================================================

if(DEFINED JCDP_include_dirs)
  target_include_directories(jcdp PRIVATE ${JCDP_include_dirs})
  target_include_directories(jcdp_batch PRIVATE ${JCDP_include_dirs})
else()
  target_include_directories(jcdp PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(jcdp_batch PRIVATE ${CMAKE_SOURCE_DIR}/include)
endif()

# =====================================================================
#  Optional project hooks (only if these macros exist in your project)
# =====================================================================

if(COMMAND check_with_iwyu)
  check_with_iwyu(jcdp IWYU_FLAGS ${JCDP_IWYU_FLAGS})
  check_with_iwyu(jcdp_batch IWYU_FLAGS ${JCDP_IWYU_FLAGS})
endif()

if(COMMAND check_with_cpplint)
  check_with_cpplint(jcdp IWYU_FLAGS ${JCDP_IWYU_FLAGS})
  check_with_cpplint(jcdp_batch IWYU_FLAGS ${JCDP_IWYU_FLAGS})
endif()

# =====================================================================
#  OpenMP (host + runtime)
# =====================================================================

# Prefer your project's OpenMP helpers if they exist.
if(COMMAND jcdp_compile_with_openmp)
  jcdp_compile_with_openmp(PRIVATE jcdp jcdp_batch)
else()
  target_compile_options(jcdp PRIVATE -fopenmp)
  target_compile_options(jcdp_batch PRIVATE -fopenmp)
endif()

if(COMMAND jcdp_link_openmp_runtime)
  jcdp_link_openmp_runtime(PRIVATE jcdp jcdp_batch)
else()
  target_link_options(jcdp PRIVATE -fopenmp)
  target_link_options(jcdp_batch PRIVATE -fopenmp)
endif()

# =====================================================================
#  OpenMP GPU Offload (Clang/LLVM -> NVPTX)
# =====================================================================

# H100 => sm_90
# You can override at configure-time:
#   cmake -S . -B build -DGPU_ARCHS="sm_90"
# Or multiple:
#   cmake -S . -B build -DGPU_ARCHS="sm_90;sm_80"
set(GPU_ARCHS "sm_90" CACHE STRING "NVIDIA GPU SM arch list (e.g. sm_90 or sm_90;sm_80)")

# Base offload flags
set(OMP_OFFLOAD_FLAGS
  -fopenmp
  -fopenmp-targets=nvptx64-nvidia-cuda
)

# Add one --offload-arch=... per arch in the list
foreach(a IN LISTS GPU_ARCHS)
  list(APPEND OMP_OFFLOAD_FLAGS --offload-arch=${a})
endforeach()

# IMPORTANT:
# Apply the offload flags BOTH:
#  1) when compiling the TU that contains the target regions
#  2) when linking the final executable (so clang runs the offload wrapper)
set_source_files_properties(${JCDP_OFFLOAD_TU} PROPERTIES
  COMPILE_OPTIONS "${OMP_OFFLOAD_FLAGS};-O3;-fno-exceptions;-fno-rtti"
)

target_link_options(jcdp PRIVATE ${OMP_OFFLOAD_FLAGS})
target_link_options(jcdp_batch PRIVATE ${OMP_OFFLOAD_FLAGS})

# =====================================================================
#  Install
# =====================================================================

install(TARGETS jcdp jcdp_batch DESTINATION .)
