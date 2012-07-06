# Allow headers in tests to be included like
# #include "PatchProjection.h" rather than needing
# #include "PatchProjection/PatchProjection.h"
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_executable(TestPatchProjection TestPatchProjection.cpp)
target_link_libraries(TestPatchProjection ITKHelpers ${ITK_LIBRARIES})

add_executable(ProjectionMatrixTiming ProjectionMatrixTiming.cpp)
target_link_libraries(ProjectionMatrixTiming ITKHelpers ${ITK_LIBRARIES})
