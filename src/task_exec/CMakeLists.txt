file(GLOB_RECURSE sources "*.cpp" "*.h")

add_library(task_exec ${sources})

# if we only have header files cmake might not be able to detect that this is 
# c++, so let's explicitly tell it
set_target_properties(task_exec PROPERTIES LINKER_LANGUAGE CXX)

# find SYSTEM libraries
find_library(ARMA_LIB armadillo)

# link with third-party SYSTEM libraries
target_link_libraries(task_exec 
  PUBLIC FAUST
  PUBLIC bmdp
  PUBLIC common
  PUBLIC shs
  PUBLIC ${ARMA_LIB}
  )

generic_includes(task_exec)
