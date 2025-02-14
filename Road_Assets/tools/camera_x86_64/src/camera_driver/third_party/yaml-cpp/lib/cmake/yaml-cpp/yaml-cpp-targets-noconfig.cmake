#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "yaml-cpp::yaml-cpp" for configuration ""
set_property(TARGET yaml-cpp::yaml-cpp APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(yaml-cpp::yaml-cpp PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libyaml-cpp.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS yaml-cpp::yaml-cpp )
list(APPEND _IMPORT_CHECK_FILES_FOR_yaml-cpp::yaml-cpp "${_IMPORT_PREFIX}/lib/libyaml-cpp.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
