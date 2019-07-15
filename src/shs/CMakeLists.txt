file(GLOB_RECURSE sources "*.cpp" "*.h")

add_library(shs ${sources})

find_library(ARMA_LIB armadillo)
find_library(MATIO_LIB matio)
find_library(GINAC_LIB ginac)

target_link_libraries(shs 
  PUBLIC ${ARMA_LIB} 
  PUBLIC ${GINAC_LIB} 
  PRIVATE ${MATIO_LIB} 
  )

generic_includes(shs)
