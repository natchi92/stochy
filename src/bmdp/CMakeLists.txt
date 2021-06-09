file(GLOB_RECURSE sources "*.cpp" "*.h")

add_library(bmdp ${sources})

# find SYSTEM libraries
find_library(ARMA_LIB armadillo)
find_library(GINAC_LIB ginac)

# setup scaffolding for third-party non-cmake libraries
set(BMDP_LIB_PATH "${CMAKE_SOURCE_DIR}/lib/BMDP-synthesis")

add_custom_target(BMDP_LIB_TARGET)
add_custom_command(
  TARGET BMDP_LIB_TARGET
  COMMAND make lib
  WORKING_DIRECTORY "${BMDP_LIB_PATH}"
  COMMENT "Building BMDP library"
  )

# 1.0: create an library target out of the library compilation result
add_library(BMDP_LIB STATIC IMPORTED GLOBAL)
add_dependencies(BMDP_LIB BMDP_LIB_TARGET)

# 2.0: specify where the library is and where to find the headers
set_target_properties(BMDP_LIB
    PROPERTIES
    IMPORTED_LOCATION "${BMDP_LIB_PATH}/lib/libsynth.a"
    INTERFACE_INCLUDE_DIRECTORIES "${BMDP_LIB_PATH}")

# 3.0: finally, link
add_dependencies(bmdp BMDP_LIB)

# link with third-party SYSTEM libraries
target_link_libraries(bmdp 
  PUBLIC common
  PUBLIC ${ARMA_LIB} 
  PUBLIC ${GINAC_LIB}
  PUBLIC nlopt
  PUBLIC task_parallelisation
  PUBLIC BMDP_LIB)

add_custom_command(
 TARGET bmdp POST_BUILD
 COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/phi2.txt" ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
 COMMENT "copying CS1.mat => ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/phi2.txt" DESTINATION bin)

generic_includes(bmdp)
