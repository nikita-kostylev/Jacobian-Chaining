# **************************************************************************** #
# This file is part of the JCDP build system. It builds the two main
# executable (jcdp and jcdp_batch).
# **************************************************************************** #

cmake_minimum_required(VERSION 3.25.0)
cmake_policy(VERSION 3.25.0)

add_executable(jcdp 
"jcdp.cpp"
"branch_and_bound_gpu_offload.cpp"
)

target_include_directories(jcdp PRIVATE ${JCDP_include_dirs})
check_with_iwyu(jcdp IWYU_FLAGS ${JCDP_IWYU_FLAGS})
check_with_cpplint(jcdp IWYU_FLAGS ${JCDP_IWYU_FLAGS})

add_executable(jcdp_batch "jcdp_batch.cpp")
target_include_directories(jcdp_batch PRIVATE ${JCDP_include_dirs})
check_with_iwyu(jcdp_batch IWYU_FLAGS ${JCDP_IWYU_FLAGS})
check_with_cpplint(jcdp_batch IWYU_FLAGS ${JCDP_IWYU_FLAGS})



# OpenMP
jcdp_compile_with_openmp(PRIVATE jcdp jcdp_batch)
jcdp_link_openmp_runtime(PRIVATE jcdp jcdp_batch)

# --- OpenMP GPU offload: only compile this TU for nvptx ---
set(GPU_ARCH "sm_80" CACHE STRING "NVIDIA GPU arch (e.g. sm_86, sm_90)")

set_source_files_properties(branch_and_bound_gpu_offload.cpp PROPERTIES
  COMPILE_OPTIONS "-fopenmp;-fopenmp-targets=nvptx64-nvidia-cuda;--offload-arch=${GPU_ARCH}"
)

# The final link step must also include offload flags (Clang bundles device image at link).
target_link_options(jcdp PRIVATE
  -fopenmp
  -fopenmp-targets=nvptx64-nvidia-cuda
  --offload-arch=${GPU_ARCH}
)



if(WIN32)
  add_cxx_flag("/EHsc" EHSC jcdp jcdp_batch)
endif()

install(TARGETS jcdp jcdp_batch DESTINATION .)
