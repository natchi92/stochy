file(GLOB_RECURSE sources "*.cpp" "*.h")

add_library(common ${sources})

# if we only have header files cmake might not be able to detect that this is 
# c++, so let's explicitly tell it
set_target_properties(common PROPERTIES LINKER_LANGUAGE CXX)

# find SYSTEM libraries
find_library(ARMA_LIB armadillo)

# link with third-party SYSTEM libraries
target_link_libraries(common 
  PUBLIC ${ARMA_LIB}
  PUBLIC shs
  )

generic_includes(common)
