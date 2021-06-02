cmake_minimum_required(VERSION 3.0.0)
project(task_parallelisation VERSION 1.0.0)

include_directories(    
    ${CMAKE_CURRENT_SOURCE_DIR}/include/task_parallelisation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")

add_library(task_parallelisation SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/include/task_parallelisation/bag.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/task_parallelisation/task_manager.h
)

target_include_directories(task_parallelisation 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(task_parallelisation PROPERTIES LINKER_LANGUAGE CXX)