#include "PatchProjection.h"

// ITK
#include "itkImage.h"

// STL
#include <vector>

int main( int argc, char ** argv )
{
  typedef itk::VectorImage<unsigned char, 2> ImageType;
  ImageType::Pointer image = ImageType::New();

  itk::Index<2> corner = {{0,0}};
  const unsigned int sideLength = 200;
  itk::Size<2> size = {{sideLength, sideLength}};
  itk::ImageRegion<2> region(corner, size);

  image->SetRegions(region);
  image->SetNumberOfComponentsPerPixel(3);
  image->Allocate();

  itk::ImageRegionIterator<ImageType> imageIterator(image, image->GetLargestPossibleRegion());

  while(!imageIterator.IsAtEnd())
    {
    ImageType::PixelType pixel(image->GetNumberOfComponentsPerPixel());
    for(unsigned int i = 0; i < image->GetNumberOfComponentsPerPixel(); ++i)
    {
      pixel[i] = rand() % 255;
    }
    imageIterator.Set(pixel);
    ++imageIterator;
    }

  typedef Eigen::MatrixXf MatrixType;
  typedef Eigen::VectorXf VectorType;

  //const unsigned int patchRadius = 15;
  const unsigned int patchRadius = 7;

  VectorType meanVector;
  std::vector<VectorType::Scalar> sortedEigenvalues;

//#define FastMethod

#ifdef FastMethod

  MatrixType projectionMatrixFromFeatureMatrix = PatchProjection<MatrixType, VectorType>::ComputeProjectionMatrix_CovarianceEigen
                                (image.GetPointer(), patchRadius, meanVector, sortedEigenvalues);
// 
//   std::cout << "projectionMatrixDirect meanVector: " << std::endl << meanVector << std::endl;
//   std::cout << "projectionMatrixDirect: " << std::endl << projectionMatrixDirect << std::endl;
#else
//     MatrixType projectionMatrixDirect = PatchProjection<MatrixType, VectorType>::ComputeProjectionMatrixFromImageElementWise
//                                 (image.GetPointer(), patchRadius, meanVector, sortedEigenvalues);

    MatrixType projectionMatrixDirect = PatchProjection<MatrixType, VectorType>::ComputeProjectionMatrixFromImageOuterProduct
                              (image.GetPointer(), patchRadius, meanVector, sortedEigenvalues);
//   std::cout << "projectionMatrixFromFeatureMatrix meanVector: " << std::endl << meanVector << std::endl;
//   std::cout << "projectionMatrixFromFeatureMatrix: " << std::endl << projectionMatrixFromFeatureMatrix << std::endl;
#endif

  return 0;
}
