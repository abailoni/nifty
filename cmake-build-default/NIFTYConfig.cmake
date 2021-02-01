# NIFTY cmake module
# This module sets the following variables in your project::
#
#   NIFTY_FOUND - true if NIFTY found on the system
#   NIFTY_INCLUDE_DIR  - the directory containing NIFTY headers
#   NIFTY_INCLUDE_DIRS - the directory containing NIFTY headers
#   NIFTY_LIBRARY - EMPTY
#   NIFTY_LIBRARIES - EMPTY

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was NIFTYConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(PN NIFTY)
set_and_check(${PN}_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include")
set_and_check(${PN}_INCLUDE_DIRS ${${PN}_INCLUDE_DIR})
set_and_check(${PN}_LIBRARY      "${PACKAGE_PREFIX_DIR}/lib"/qpbo)
set(${PN}_LIBRARY "")
set(${PN}_LIBRARIES ${${PN}_LIBRARY})
check_required_components(${PN})
