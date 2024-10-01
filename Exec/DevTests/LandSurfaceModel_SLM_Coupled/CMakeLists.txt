set(erf_exe_name LandSurfaceModel_SLMERF)

add_executable(${erf_exe_name} "")
target_sources(${erf_exe_name}
   PRIVATE
     ERF_prob.cpp
)

target_include_directories(${erf_exe_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set(TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/inputs"
               "${CMAKE_CURRENT_SOURCE_DIR}/inputs_cass_slm"
               "${CMAKE_CURRENT_SOURCE_DIR}/input_sounding"
               "${CMAKE_CURRENT_SOURCE_DIR}/lsf_cass"
               "${CMAKE_CURRENT_SOURCE_DIR}/snd_cass"
               "${CMAKE_CURRENT_SOURCE_DIR}/sounding_cass"
               "${CMAKE_CURRENT_SOURCE_DIR}/sounding_cass_2")
file(COPY ${TEST_FILES} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/")

include(${CMAKE_SOURCE_DIR}/CMake/BuildERFExe.cmake)
build_erf_exe(${erf_exe_name})
