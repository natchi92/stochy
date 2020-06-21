cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 14)

# generate compile_commands.json, useful for language servers
set(CMAKE_EXPORT_COMPILE_COMMANDS true)

# group output targets by type (executables and libraries)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# set RPATH relative to location of executable for install directories
set(CMAKE_INSTALL_RPATH "\$ORIGIN/../lib")

add_subdirectory(src)

#-------------------------------------------------------------------------------
# THIRD-PARTY LIBS
#-------------------------------------------------------------------------------

# overwrite install() command with a dummy macro that is a nop.
# this prevents third-parties from polluting install directory
macro (install)
endmacro ()

# only third-party libs that support cmake should be added here
option (NLOPT_PYTHON "build python bindings" OFF)
option (NLOPT_OCTAVE "build octave bindings" OFF)
option (NLOPT_MATLAB "build matlab bindings" OFF)
option (NLOPT_GUILE "build guile bindings" OFF)
option (NLOPT_SWIG "use SWIG to build bindings" OFF)
add_subdirectory(lib/nlopt)
add_subdirectory(lib/cubature)

# restore install() command
macro (install)
  _install(${ARGV})
endmacro(install)

#-------------------------------------------------------------------------------
# CPACK
#-------------------------------------------------------------------------------

set(CPACK_GENERATOR "ZIP")

set(CPACK_BINARY_7Z "OFF")
set(CPACK_BINARY_BUNDLE "OFF")
set(CPACK_BINARY_CYGWIN "OFF")
set(CPACK_BINARY_DEB "OFF")
set(CPACK_BINARY_DRAGNDROP "OFF")
set(CPACK_BINARY_FREEBSD "OFF")
set(CPACK_BINARY_IFW "OFF")
set(CPACK_BINARY_NSIS "OFF")
set(CPACK_BINARY_NUGET "OFF")
set(CPACK_BINARY_OSXX11 "OFF")
set(CPACK_BINARY_TBZ2 "OFF")
set(CPACK_BINARY_TGZ "OFF")
set(CPACK_BINARY_TXZ "OFF")
set(CPACK_BINARY_TZ "OFF")
set(CPACK_BINARY_ZIP "ON")

set(CPACK_PACKAGE_NAME "stochy")

set(CPACK_PACKAGE_FILE_NAME "stochy-1.0.0")
set(CPACK_PACKAGE_VENDOR "Nathalie Cauchi")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")

include(CPack)
