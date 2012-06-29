cmake_minimum_required(VERSION 2.6)

Project(PatchProjection)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")

include_directories(/media/portable/src/Eigen)

FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})

FIND_PACKAGE(VTK REQUIRED)
INCLUDE(${VTK_USE_FILE})

if(NOT TARGET ITKHelpers)
  add_subdirectory(ITKHelpers)
endif()

if(NOT TARGET EigenHelpers)
  add_subdirectory(EigenHelpers)
endif()

SET(PatchProjection_BuildExample OFF CACHE BOOL "PatchProjection_BuildExample")
if(PatchProjection_BuildExample)
  add_executable(PatchProjectionExample Example.cpp)
  target_link_libraries(PatchProjectionExample ${ITK_LIBRARIES} ITKHelpers EigenHelpers)
endif()


# add_library(PatchProjection)
# target_link_libraries(PatchProjection ${ITK_LIBRARIES})
