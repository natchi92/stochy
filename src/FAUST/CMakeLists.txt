file(GLOB_RECURSE sources "*.cpp" "*.h")

add_library(FAUST ${sources})

find_library(ARMA_LIB armadillo)
find_library(GINAC_LIB ginac)

#TODO: not sure why this is needed.
target_include_directories(FAUST PUBLIC
 $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib/cubature>
 )

target_link_libraries(FAUST 
  PUBLIC ${ARMA_LIB} 
  PUBLIC ${GINAC_LIB}
  PUBLIC shs
  PRIVATE nlopt 
  PRIVATE cubature)


install(FILES "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libcubature.so" DESTINATION lib)
install(FILES "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libnlopt.so.0.10.0" 
        RENAME "libnlopt.so.0"
        DESTINATION lib)


generic_includes(FAUST)
