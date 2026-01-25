cmake_minimum_required(VERSION 3.10)
include_guard(GLOBAL)

# Enable IN_LIST if() operator
cmake_policy(SET CMP0057 NEW)

include(util/check_flags)
include(util/print)

# **************************************************************************** #

# Add a C compiler flag
function(add_c_flag flag test_var)
  get_property(_enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  if("C" IN_LIST _enabled_languages)
    _add_flag("C" "${flag}" "${test_var}" ${ARGN})
  else()
    _print_warning(
      "Tried to add C flag '${flag}' but the C language is not enabled.")
  endif()
endfunction()

# Add a C++ compiler flag
function(add_cxx_flag flag test_var)
  get_property(_enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  if("CXX" IN_LIST _enabled_languages)
    _add_flag("CXX" "${flag}" "${test_var}" ${ARGN})
  else()
    _print_warning(
      "Tried to add C++ flag '${flag}' but the C++ language is not enabled.")
  endif()
endfunction()

# Add a Fortran compiler flag
function(add_fortran_flag flag test_var)
  get_property(_enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  if("Fortran" IN_LIST _enabled_languages)
    _add_flag("Fortran" "${flag}" "${test_var}" ${ARGN})
  else()
    _print_warning(
      "Tried to add Fortran flag '${flag}' but Fortran is not enabled.")
  endif()
endfunction()

# Add a compiler flag to every enabled language
function(add_flag flag test_var)
  get_property(_enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  foreach(_language "C" "CXX" "Fortran")
    if(${_language} IN_LIST _enabled_languages)
      _add_flag(${_language} "${flag}" "${test_var}" ${ARGN})
    endif()
  endforeach()
endfunction()

# **************************************************************************** #

# Add a C compiler definition
function(add_c_definition def)
  get_property(_enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  if("C" IN_LIST _enabled_languages)
    _add_definition("C" "${def}" ${ARGN})
  else()
    _print_warning(
      "Tried to add C definition '${def}' but the C is not enabled.")
  endif()
endfunction()

# Add a C++ compiler definition
function(add_cxx_definition def)
  get_property(_enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  if("CXX" IN_LIST _enabled_languages)
    _add_definition("CXX" "${def}" ${ARGN})
  else()
    _print_warning(
      "Tried to add C++ definition '${def}' but the C++ is not enabled.")
  endif()
endfunction()

# Add a Fortran compiler definition
function(add_fortran_definition def)
  get_property(_enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  if("Fortran" IN_LIST _enabled_languages)
    _add_definition("Fortran" "${def}" ${ARGN})
  else()
    _print_warning(
      "Tried to add Fortran definition '${def}' but Fortran is not enabled.")
  endif()
endfunction()

# Add a compiler definition to every enabled language
function(add_definition def)
  get_property(_enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  foreach(_language "C" "CXX" "Fortran")
    if(${_language} IN_LIST _enabled_languages)
      _add_definition(${_language} "${def}" ${ARGN})
    endif()
  endforeach()
endfunction()

if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
  add_cxx_flag("-O3" O3)
  add_cxx_flag("-march=native" MARCH_NATIVE)
  add_cxx_flag("-ffast-math" FAST_MATH)
  add_cxx_flag("-funroll-loops" UNROLL)
  add_cxx_flag("-fno-exceptions" NO_EXC)
  add_cxx_flag("-fno-rtti" NO_RTTI)
endif()
