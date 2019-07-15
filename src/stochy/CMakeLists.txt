add_executable(stochy stochy_main.cpp)

add_executable(stochy_bas MainBAS.cpp)
add_executable(stochy_cs1 MainCs1.cpp)
add_executable(stochy_cs2 MainCs2.cpp)
add_executable(stochy_cs3 MainCs3.cpp)
add_executable(stochy_cs4 MainCs4.cpp)
add_executable(stochy_2d_eg Main_2D_eg.cpp)
add_executable(stochy_ane_ra Main_Ane_Ra.cpp)
add_executable(stochy_ane_safety Main_Ane_Safety.cpp)
add_executable(stochy_binary Main_Binary.cpp)

target_link_libraries(stochy 
  PRIVATE case_studies 
  PRIVATE common)

target_link_libraries(stochy_bas
  PRIVATE common
  PRIVATE task_exec)

target_link_libraries(stochy_cs1
  PRIVATE common
  PRIVATE task_exec)

target_link_libraries(stochy_cs2
  PRIVATE common
  PRIVATE task_exec)

target_link_libraries(stochy_cs3
  PRIVATE common
  PRIVATE task_exec)

target_link_libraries(stochy_cs4
  PRIVATE common
  PRIVATE task_exec)

target_link_libraries(stochy_2d_eg
  PRIVATE common
  PRIVATE task_exec)

target_link_libraries(stochy_ane_ra
  PRIVATE common
  PRIVATE task_exec)

target_link_libraries(stochy_ane_safety
  PRIVATE common
  PRIVATE task_exec)

target_link_libraries(stochy_binary
  PRIVATE common
  PRIVATE task_exec)

install(TARGETS stochy DESTINATION bin)
install(FILES "${CMAKE_SOURCE_DIR}/README.md" DESTINATION .)
install(FILES "${CMAKE_SOURCE_DIR}/get_dep.dist.sh" DESTINATION .)

