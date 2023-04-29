add_executable(stochy stochy_main.cpp)

#add_executable(stochy_bas MainBAS.cpp)
#add_executable(stochy_cs1 MainCs1.cpp)
#add_executable(stochy_cs2 MainCs2.cpp)
#add_executable(stochy_cs3 MainCs3.cpp)
#add_executable(stochy_cs4 MainCs4.cpp)
#add_executable(stochy_2d_eg Main_2D_eg.cpp)
#add_executable(stochy_ane_ra Main_Ane_Ra.cpp)
#add_executable(stochy_ane_safety Main_Ane_Safety.cpp)
#add_executable(stochy_binary Main_Binary.cpp)
#add_executable(stochy_arch Main_ARCH.cpp)

target_link_libraries(stochy 
  PRIVATE case_studies 
  PRIVATE common)



install(TARGETS stochy DESTINATION bin)
install(FILES "${CMAKE_SOURCE_DIR}/README.md" DESTINATION .)
install(FILES "${CMAKE_SOURCE_DIR}/get_dep.dist.sh" DESTINATION .)

