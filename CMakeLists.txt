cmake_minimum_required(VERSION 2.6)

Project(PatchProjection)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")

include_directories(/media/portable/src/Eigen)

FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})

add_executable(PatchProjectionExample Example.cpp)
target_link_libraries(PatchProjectionExample ${ITK_LIBRARIES})

# add_library(PatchProjection)
# target_link_libraries(PatchProjection ${ITK_LIBRARIES})
