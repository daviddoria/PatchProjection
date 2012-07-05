#include "PatchProjection.h"

// ITK
#include "itkImage.h"

// STL
#include <vector>

int main( int argc, char ** argv )
{
  typedef itk::Image<unsigned char, 2> ImageType;
  ImageType::Pointer image = ImageType::New();

  itk::Index<2> corner = {{0,0}};
  itk::Size<2> size = {{10,10}};
  itk::ImageRegion<2> region(corner, size);

  image->SetRegions(region);
  image->Allocate();

  itk::ImageRegionIterator<ImageType> imageIterator(image, image->GetLargestPossibleRegion());

  while(!imageIterator.IsAtEnd())
    {
    imageIterator.Set(rand() % 255);
    ++imageIterator;
    }

  typedef Eigen::MatrixXf MatrixType;
  typedef Eigen::VectorXf VectorType;

  VectorType meanVector;
  std::vector<VectorType::Scalar> sortedEigenvalues;
  
  MatrixType projectionMatrix = PatchProjection<MatrixType, VectorType>::ComputeProjectionMatrixFromImage
                                (image.GetPointer(), 3, meanVector, sortedEigenvalues);
  
  return 0;
}
