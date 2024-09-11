set(erf_exe_name LandSurfaceModel_SLMERF)

add_executable(${erf_exe_name} "")
target_sources(${erf_exe_name}
   PRIVATE
     prob.cpp
)

target_include_directories(${erf_exe_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set(TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/inputs"
               "${CMAKE_CURRENT_SOURCE_DIR}/inputs_sq")
file(COPY ${TEST_FILES} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/")

include(${CMAKE_SOURCE_DIR}/CMake/BuildERFExe.cmake)
build_erf_exe(${erf_exe_name})
