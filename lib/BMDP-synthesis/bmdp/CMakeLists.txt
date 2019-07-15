file(GLOB_RECURSE BMDP_SOURCE_CODE *.cpp)
add_library(bmdp SHARED ${BMDP_SOURCE_CODE})
target_link_libraries(bmdp ${SUPERLU_LIBRARIES} ${OMPL_LIBRARY})
