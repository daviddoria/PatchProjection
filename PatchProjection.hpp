#ifndef PatchProjection_hpp
#define PatchProjection_hpp

#include "PatchProjection.h" // appease syntax parser

// ITK
#include "itkImageRegionConstIterator.h"

// Submodules
#include "ITKHelpers/ITKHelpers.h"
#include "EigenHelpers/EigenHelpers.h"
#include "ITKHelpers/Helpers/ParallelSort.h"

template <typename TImage>
Eigen::VectorXf PatchProjection::VectorizePatch(const TImage* const image, const itk::ImageRegion<2>& region)
{
  Eigen::VectorXf vectorized =
       Eigen::VectorXf::Zero(image->GetNumberOfComponentsPerPixel() * region.GetNumberOfPixels());

  itk::ImageRegionConstIterator<TImage> imageIterator(image, region);

  unsigned int numberOfComponentsPerPixel = image->GetNumberOfComponentsPerPixel();
  unsigned int pixelCounter = 0;
  while(!imageIterator.IsAtEnd())
    {
    for(unsigned int component = 0; component < numberOfComponentsPerPixel; ++component)
      {
      vectorized[numberOfComponentsPerPixel * pixelCounter + component] = imageIterator.Get()[component];
      }
    pixelCounter++;
    ++imageIterator;
    }
  return vectorized;
}

template <typename TImage>
void PatchProjection::UnvectorizePatch(const Eigen::VectorXf& vectorized, TImage* const image, const unsigned int channels)
{
  // This function assumes the patch is square
  image->SetNumberOfComponentsPerPixel(channels);

  unsigned int numberOfPixels = vectorized.size()/channels;
  unsigned int sideLength = sqrt(numberOfPixels);

  itk::Size<2> size;
  size.Fill(sideLength);

  itk::ImageRegion<2> region = ITKHelpers::CornerRegion(size);
  image->SetRegions(region);
  image->Allocate();

  itk::ImageRegionIterator<TImage> imageIterator(image, region);
  unsigned int pixelCounter = 0;
  while(!imageIterator.IsAtEnd())
  {
    typename TImage::PixelType pixel(channels);

    for(unsigned int component = 0; component < channels; ++component)
    {
      float value = vectorized[channels * pixelCounter + component];
      // Make sure that the float value is within the range of the image pixel type. For example, we can't
      // convert -23.4 to uchar.
      if(value < std::numeric_limits<typename TImage::InternalPixelType>::min() ||
         value > std::numeric_limits<typename TImage::InternalPixelType>::max())
      {
        std::stringstream ss;
        ss << "PatchProjection::UnvectorizePatch: Value " << value
           << " cannot be converted to a component of the type of the image pixels";
        //throw std::runtime_error(ss.str());
      }
      pixel[component] = value;
    } // end for
    imageIterator.Set(pixel);
    pixelCounter++;
    ++imageIterator;
  } // end while
}

template <typename TImage>
Eigen::MatrixXf PatchProjection::VectorizeImage(const TImage* const image, const unsigned int patchRadius)
{
  // The matrix constructed by this has each vectorized patch as a column.

  unsigned int numberOfComponentsPerPixel = image->GetNumberOfComponentsPerPixel();
  unsigned int pixelsPerPatch = (patchRadius * 2 + 1) * (patchRadius * 2 + 1);
  unsigned int featureLength = numberOfComponentsPerPixel * pixelsPerPatch;

  // This is how many patches fit entirely inside the image.
  // For a 572x516 image and patch radius 7, we get 280116 patches.
  itk::Size<2> imageSize = image->GetLargestPossibleRegion().GetSize();
  unsigned int numberOfPatches = (imageSize[0] - patchRadius*2) * (imageSize[1] - patchRadius*2);

  std::cout << "Allocating feature matrix " << featureLength << " x " << numberOfPatches << std::endl;
  Eigen::MatrixXf featureMatrix(featureLength, numberOfPatches);
  std::cout << "Allocated feature matrix " << featureMatrix.rows() << " x " << featureMatrix.cols() << std::endl;
  itk::ImageRegionConstIterator<TImage> imageIterator(image, image->GetLargestPossibleRegion());

  std::vector<itk::ImageRegion<2> > allPatches = ITKHelpers::GetAllPatches(image->GetLargestPossibleRegion(), patchRadius);
  std::cout << "There are " << allPatches.size() << " patches." << std::endl;
  for(unsigned int patchId = 0; patchId < allPatches.size(); ++patchId)
  {
    featureMatrix.col(patchId) = VectorizePatch(image, allPatches[patchId]);
  }

  return featureMatrix;
}

template <typename TImage>
Eigen::MatrixXf PatchProjection::GetDummyProjectionMatrix(const TImage* const image, const unsigned int patchRadius,
                                                          Eigen::VectorXf& meanVector, Eigen::VectorXf& standardDeviationVector)
{
  unsigned int numberOfComponentsPerPixel = image->GetNumberOfComponentsPerPixel();
  unsigned int pixelsPerPatch = (patchRadius * 2 + 1) * (patchRadius * 2 + 1);
  unsigned int featureLength = numberOfComponentsPerPixel * pixelsPerPatch;

  Eigen::MatrixXf dummyProjectionMatrix(featureLength, featureLength);
  dummyProjectionMatrix.setIdentity();

  meanVector.resize(featureLength);
  meanVector.setZero();

  standardDeviationVector.resize(featureLength);
  standardDeviationVector.setOnes();

  return dummyProjectionMatrix;
}

template <typename TImage>
Eigen::MatrixXf PatchProjection::GetDummyProjectionMatrix(const TImage* const image, const unsigned int patchRadius)
{
  Eigen::VectorXf meanVector;
  Eigen::VectorXf standardDeviationVector;
  return GetDummyProjectionMatrix(image, patchRadius, meanVector, standardDeviationVector);
}

template <typename TImage>
Eigen::MatrixXf PatchProjection::ComputeProjectionMatrix(const TImage* const image, const unsigned int patchRadius)
{
  Eigen::VectorXf meanVector;
  Eigen::VectorXf standardDeviationVector;
  return ComputeProjectionMatrix_CovarianceEigen(image, patchRadius, meanVector, standardDeviationVector);
}

template <typename TImage>
Eigen::MatrixXf PatchProjection::ComputeProjectionMatrix_CovarianceEigen(const TImage* const image, const unsigned int patchRadius,
                                                         Eigen::VectorXf& meanVector, Eigen::VectorXf& standardDeviationVector)
{
  throw std::runtime_error("Not yet implemented.");
}

template <typename TImage>
Eigen::MatrixXf PatchProjection::ComputeProjectionMatrix_CovarianceEigen(const TImage* const image, const unsigned int patchRadius,
                                                         Eigen::VectorXf& meanVector)
{
  std::vector<itk::ImageRegion<2> > allPatches = ITKHelpers::GetAllPatches(image->GetLargestPossibleRegion(), patchRadius);
  Eigen::MatrixXf featureMatrix = PatchProjection::VectorizeImage(image, patchRadius);

  // Standardize the vectorized patches, and store the meanVector
  // used to do so for later un-standardization
  meanVector = featureMatrix.rowwise().mean();
  // Subtract the mean vector from every column
  featureMatrix.colwise() -= meanVector;

  Eigen::MatrixXf covarianceMatrix = EigenHelpers::ConstructCovarianceMatrixFromFeatureMatrix(featureMatrix);

  std::cout << "Done computing covariance matrix (" << covarianceMatrix.rows() << " x " << covarianceMatrix.cols() << ")" << std::endl;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigensolver(covarianceMatrix);
  if (eigensolver.info() != Eigen::Success)
  {
    throw std::runtime_error("Eigen decomposition of the covariance matrix failed!");
  }

  // Sort eigenvectors by increasing eigenvalue magnitude.

  std::vector<float> eigenvalueMagnitudes(eigensolver.eigenvalues().size());
  for(int i = 0; i < eigensolver.eigenvalues().size(); ++i)
  {
    eigenvalueMagnitudes[i] = fabs(eigensolver.eigenvalues()[i]);
  }

  // Sort the eigenvalues from largest magnitude to smallest
  std::vector<ParallelSort::IndexedValue<float> > sorted = ParallelSort::ParallelSortDescending<float>(eigenvalueMagnitudes);

  // Reorder the eigenvectors
  Eigen::MatrixXf sortedEigenVectors(eigensolver.eigenvectors().rows(), eigensolver.eigenvectors().cols());
  for(size_t i = 0; i < sorted.size(); ++i)
  {
    sortedEigenVectors.col(i) = eigensolver.eigenvectors().col(sorted[i].index);
  }

  return sortedEigenVectors;
}


template <typename TImage>
Eigen::MatrixXf PatchProjection::ComputeProjectionMatrix_SVD(const TImage* const image, const unsigned int patchRadius,
                                                         Eigen::VectorXf& meanVector, Eigen::VectorXf& standardDeviationVector)
{
  std::vector<itk::ImageRegion<2> > allPatches = ITKHelpers::GetAllPatches(image->GetLargestPossibleRegion(), patchRadius);
  Eigen::MatrixXf featureMatrix = PatchProjection::VectorizeImage(image, patchRadius);

  // Standardize the vectorized patches, and store the meanVector and standardDeviationVector
  // used to do so for later un-standardization
  meanVector = featureMatrix.rowwise().mean();
  // Subtract the mean vector from every column
  featureMatrix.colwise() -= meanVector;

  // The variance is computed as 1/N \sum (x_i - x_mean)^2 . Since we have zero mean, this is just the square of the components
  Eigen::MatrixXf squaredMean0FeatureMatrix = featureMatrix.array().pow(2); // Square all components
  Eigen::VectorXf variance = squaredMean0FeatureMatrix.rowwise().mean();
  standardDeviationVector = variance.array().sqrt(); // Take the square root of all components

  // Divide by the standard devation
  // featureMatrix.colwise() /= standardDeviation; // this does not yet work in Eigen
  featureMatrix = standardDeviationVector.matrix().asDiagonal().inverse() * featureMatrix;

  typedef Eigen::JacobiSVD<Eigen::MatrixXf> SVDType;
  //SVDType svd(featureMatrix, Eigen::ComputeFullU);
  SVDType svd(featureMatrix, Eigen::ComputeThinU);
  return svd.matrixU();
}

#endif
