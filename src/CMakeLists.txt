# **************************************************************************** #
# This file is part of the JCDP build system. It builds the two main
# executable (jcdp and jcdp_batch).
# **************************************************************************** #

cmake_minimum_required(VERSION 3.25.0)
cmake_policy(VERSION 3.25.0)

add_executable(jcdp "jcdp.cpp")
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

if(WIN32)
  add_cxx_flag("/EHsc" EHSC jcdp jcdp_batch)
endif()

install(TARGETS jcdp jcdp_batch DESTINATION .)
