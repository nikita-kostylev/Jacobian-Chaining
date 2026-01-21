# **************************************************************************** #
# This file is part of the JCDP build system. It generates some IWYU and
# Cpplint for the header files.
# **************************************************************************** #

# Collect local headers
set(_local_headers
  ${CMAKE_CURRENT_SOURCE_DIR}/branch_and_bound.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/branch_and_bound_gpu.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/priority_list.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scheduler.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bnb_block.hpp)

# Setup header-only IWYU target
header_only_iwyu_targets("jcdp_scheduler"
  HEADERS ${_local_headers}
  COMPILER_FLAGS ${JCDP_IWYU_COMPILER_FLAGS}
  INCLUDE_DIRS ${JCDP_include_dirs}
  IWYU_FLAGS ${JCDP_IWYU_FLAGS})

# Setup header-only Cpplint targets
header_only_cpplint_targets("jcdp_scheduler"
  HEADERS ${_local_headers})

# **************************************************************************** #
# Cleanup
# **************************************************************************** #
unset(_local_headers)
