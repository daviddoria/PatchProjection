cmake_minimum_required(VERSION 2.6)

Project(PatchProjection)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")

include_directories(/media/portable/src/Eigen)

FIND_PACKAGE(ITK REQUIRED ITKCommon ITKIOImageBase
ITKImageIntensity ITKImageFeature ITKMathematicalMorphology ITKBinaryMathematicalMorphology ITKDistanceMap)
INCLUDE(${ITK_USE_FILE})

FIND_PACKAGE(VTK REQUIRED)
INCLUDE(${VTK_USE_FILE})

if(NOT TARGET ITKHelpers)
  add_subdirectory(ITKHelpers)
  include_directories(${ITKHelpers_includes})
endif()

if(NOT TARGET EigenHelpers)
  add_subdirectory(EigenHelpers)
  include_directories(${EigenHelpers_includes})
endif()

if(NOT ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  set(PatchProjection_includes ${ITKHelpers_includes} ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
endif()

SET(PatchProjection_BuildExample OFF CACHE BOOL "PatchProjection_BuildExample")
if(PatchProjection_BuildExample)
  add_executable(PatchProjectionExample Example.cpp)
  target_link_libraries(PatchProjectionExample ${ITK_LIBRARIES} ITKHelpers EigenHelpers)
endif()

# Do this even though it is not necessary (this is a template library (header-only)). We use a
# target with the same name as the submodule to check if the submodule has already been included.
add_library(PatchProjection)
target_link_libraries(PatchProjection ${ITK_LIBRARIES})

SET(PatchProjection_BuildTests OFF CACHE BOOL "PatchProjection_BuildTests")
if(PatchProjection_BuildTests)
  add_subdirectory(Tests)
endif()
