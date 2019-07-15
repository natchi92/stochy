project(StocHy)

macro(generic_includes name)
    target_include_directories(${name}
        PUBLIC
        ${STOCHY_BINARY_DIR}
        ${STOCHY_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endmacro(generic_includes)

add_subdirectory(stochy)
add_subdirectory(case_studies)
add_subdirectory(common)
add_subdirectory(shs)
add_subdirectory(FAUST)
add_subdirectory(bmdp)
add_subdirectory(task_exec)
