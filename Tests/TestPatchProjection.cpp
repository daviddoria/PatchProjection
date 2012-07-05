#include "PatchProjection.h"

// ITK
#include "itkImage.h"

// STL
#include <vector>

static void Small();
static void Large();

int main( int argc, char ** argv )
{
  Small();
  //Large();
  return 0;
}

void Small()
{
  typedef itk::Image<unsigned char, 2> ImageType;
  ImageType::Pointer image = ImageType::New();

  itk::Index<2> corner = {{0,0}};
  itk::Size<2> size = {{4,4}};
  itk::ImageRegion<2> region(corner, size);

  image->SetRegions(region);
  image->Allocate();

  itk::Index<2> index;
  /* The following image generates this feature matrix:
   * octave:
   X=[199 135 228 196;
      135  31 196  66;
      31  31  66  66;
      228 196 166 246;
      196  66 246 204;
      66  66 204 204;
      166 246 135 135;
      246 204 135 135;
      204 204 135 135]
    */

  index[0] = 0; index[1] = 0;
  image->SetPixel(index, 199);

  index[0] = 0; index[1] = 1;
  image->SetPixel(index, 228);

  index[0] = 0; index[1] = 2;
  image->SetPixel(index, 166);

  index[0] = 0; index[1] = 3;
  image->SetPixel(index, 135);
  
  index[0] = 1; index[1] = 0;
  image->SetPixel(index, 135);

  index[0] = 1; index[1] = 1;
  image->SetPixel(index, 196);

  index[0] = 1; index[1] = 2;
  image->SetPixel(index, 246);

  index[0] = 1; index[1] = 3;
  image->SetPixel(index, 135);
  
  index[0] = 2; index[1] = 0;
  image->SetPixel(index, 31);

  index[0] = 2; index[1] = 1;
  image->SetPixel(index, 66);

  index[0] = 2; index[1] = 2;
  image->SetPixel(index, 204);

  index[0] = 2; index[1] = 3;
  image->SetPixel(index, 135);

  index[0] = 3; index[1] = 0;
  image->SetPixel(index, 31);

  index[0] = 3; index[1] = 1;
  image->SetPixel(index, 66);

  index[0] = 3; index[1] = 2;
  image->SetPixel(index, 204);

  index[0] = 3; index[1] = 3;
  image->SetPixel(index, 135);


  typedef Eigen::MatrixXf MatrixType;
  typedef Eigen::VectorXf VectorType;
  const unsigned int patchRadius = 1;

  // Get the feature matrix just so that we can compare the computed covariance matrices to the octave equivalent
  MatrixType featureMatrix = PatchProjection<MatrixType, VectorType>::VectorizeImage(image.GetPointer(), patchRadius);
  std::cout << "featureMatrix: " << std::endl << featureMatrix << std::endl;

  // This is the covariance matrix that should be generated from the featureMatrix
  /* octave:
   * cov(transpose(X)) = 

  1528.333   2522.667    525.000   -175.333   3020.667   2070.000  -1918.333   -902.000 -1035.000
   2522.667   5387.333    560.000  -1274.667   4667.333   2208.000  -2522.667   -712.000 -1104.000
    525.000    560.000    408.333    -70.000   1096.667   1610.000   -828.333  -1050.000 -805.000
   -175.333  -1274.667    -70.000   1249.333    -54.667   -276.000   -284.667    404.000 138.000
   3020.667   4667.333   1096.667    -54.667   6056.000   4324.000  -3958.000  -1910.000 -2162.000
   2070.000   2208.000   1610.000   -276.000   4324.000   6348.000  -3266.000  -4140.000 -3174.000
  -1918.333  -2522.667   -828.333   -284.667  -3958.000  -3266.000   2747.000   1570.000 1633.000
   -902.000   -712.000  -1050.000    404.000  -1910.000  -4140.000   1570.000   2994.000 2070.000
  -1035.000  -1104.000   -805.000    138.000  -2162.000  -3174.000   1633.000   2070.000 1587.000
  */

  VectorType meanVector;
  std::vector<VectorType::Scalar> sortedEigenvalues;

  MatrixType projectionMatrixDirect = PatchProjection<MatrixType, VectorType>::ComputeProjectionMatrixFromImage
                                (image.GetPointer(), patchRadius, meanVector, sortedEigenvalues);

  std::cout << "projectionMatrixDirect meanVector: " << std::endl << meanVector << std::endl;
  std::cout << "projectionMatrixDirect: " << std::endl << projectionMatrixDirect << std::endl;

  MatrixType projectionMatrixFromFeatureMatrix = PatchProjection<MatrixType, VectorType>::ComputeProjectionMatrix_CovarianceEigen
                                (image.GetPointer(), patchRadius, meanVector, sortedEigenvalues);

  std::cout << "projectionMatrixFromFeatureMatrix meanVector: " << std::endl << meanVector << std::endl;
  std::cout << "projectionMatrixFromFeatureMatrix: " << std::endl << projectionMatrixFromFeatureMatrix << std::endl;
}

void Large()
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

  const unsigned int patchRadius = 1;

  MatrixType projectionMatrixDirect = PatchProjection<MatrixType, VectorType>::ComputeProjectionMatrixFromImage
                                (image.GetPointer(), patchRadius, meanVector, sortedEigenvalues);

  std::cout << "projectionMatrixDirect meanVector: " << std::endl << meanVector << std::endl;
  std::cout << "projectionMatrixDirect: " << std::endl << projectionMatrixDirect << std::endl;

  //MatrixType featureMatrix = PatchProjection<MatrixType, VectorType>::VectorizeImage(image.GetPointer(), patchRadius);
  MatrixType projectionMatrixFromFeatureMatrix = PatchProjection<MatrixType, VectorType>::ComputeProjectionMatrix_CovarianceEigen
                                (image.GetPointer(), patchRadius, meanVector, sortedEigenvalues);

  std::cout << "projectionMatrixFromFeatureMatrix meanVector: " << std::endl << meanVector << std::endl;
  std::cout << "projectionMatrixFromFeatureMatrix: " << std::endl << projectionMatrixFromFeatureMatrix << std::endl;
}

