cmake_minimum_required(VERSION 3.10)
include_guard(GLOBAL)

# Enable IN_LIST if() operator
cmake_policy(SET CMP0057 NEW)

include(util/check_flags)
include(util/print)

# Enable intrinsics option
option(ENABLE_NATIVE_INTRINSICS "Enable all host specific intrinsics." ON)

macro(print_intrinsics_status)
  _print_status("Native intrinsics: '${ENABLE_NATIVE_INTRINSICS}'")
endmacro()

# **************************************************************************** #

# Searches for the best architecture flags
function(_search_for_architecture_flags language)
  if(WIN32)
    if(CMAKE_${language}_COMPILER_ID MATCHES "MSVC")
      # Check for AVX 512
      _add_flag(${language} "/arch:AVX512" AVX512)
      if(_current_flags)
        set(_current_flags ${_current_flags} PARENT_SCOPE)
        return()
      endif()

      # Check for AVX2
      _add_flag(${language} "/arch:AVX2" AVX2)
      if(_current_flags)
        set(_current_flags ${_current_flags} PARENT_SCOPE)
        return()
      endif()

      # Check for AVX
      _add_flag(${language} "/arch:AVX" AVX)
      if(_current_flags)
        set(_current_flags ${_current_flags} PARENT_SCOPE)
        return()
      endif()

      if(CMAKE_SIZEOF_VOID_P EQUAL 4)
        # Check for SSE2
        _add_flag(${language} "/arch:SSE2" SSE2)
        if(_current_flags)
          set(_current_flags ${_current_flags} PARENT_SCOPE)
          return()
        endif()

        # Check for SSE
        _add_flag(${language} "/arch:SSE" SSE)
        if(_current_flags)
          set(_current_flags ${_current_flags} PARENT_SCOPE)
          return()
        endif()

        # Check for IA32
        _add_flag(${language} "/arch:IA32" IA32)
      endif()
    elseif(CMAKE_${language}_COMPILER_ID MATCHES "Intel")
      # Check for native architecture flags for the Intel compiler on Windows
      _add_flag(${language} "/QxHost" XHOST)
    endif()
  else()
    if(CMAKE_${language}_COMPILER_ID MATCHES "Intel")
      # Check for native architecture flags for the Intel compiler on Unix
      _add_flag(${language} "-xHost" XHOST)
    else()
      # Check for native architecture flags on Unix
      _add_flag(${language} "-march=native" MARCH_NATIVE)
    endif()
  endif()

  set(_current_flags ${_current_flags} PARENT_SCOPE)
endfunction()

# **************************************************************************** #

# If ENABLE_NATIVE_INTRINSICS is set, check for native architecture flags.
if(ENABLE_NATIVE_INTRINSICS)
  get_property(_enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  foreach(_language "C" "CXX" "Fortran")
    if("${_language}" IN_LIST _enabled_languages)
      # Search for architecture flag for the current language
      set(_current_flags "")
      _search_for_architecture_flags(${_language})

      if(_current_flags)
        _print_list(
          "${_language} architecture flags:" ${_current_flags})
      else()
        _print_status("No ${_language} architecture flags found.")
      endif()
    endif()
  endforeach()
endif()

# **************************************************************************** #

# Cleanup
unset(_enabled_languages)
unset(_current_flags)
unset(_valid_flags)
unset(_invalid_flags)
