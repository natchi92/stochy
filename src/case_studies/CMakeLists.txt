file(GLOB_RECURSE sources "*.cpp" "*.h")

add_library(case_studies ${sources})

target_link_libraries(case_studies common task_exec)

add_custom_command(
 TARGET case_studies POST_BUILD
 COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/CS1.mat" ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
 COMMENT "copying CS1.mat => ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

add_custom_command(
 TARGET case_studies POST_BUILD
 COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/CS4.mat" ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
 COMMENT "copying CS1.mat => ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

add_custom_command(
 TARGET case_studies POST_BUILD
 COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/u.txt" ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
 COMMENT "copying u.txt => ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/u.txt" DESTINATION bin)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/CS1.mat" DESTINATION bin)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/CS4.mat" DESTINATION bin)

generic_includes(case_studies)
