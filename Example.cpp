#include <iostream>
#include <vector>

#include "PatchProjection.h"

int main()
{
  std::vector<itk::ImageRegion<2> > allPoints;

  PatchProjection::VectorizePatch();
  return EXIT_SUCCESS;
}
