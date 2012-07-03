#ifndef PatchProjection_hpp
#define PatchProjection_hpp

#include "PatchProjection.h" // appease syntax parser

// ITK
#include "itkImageRegionConstIterator.h"

// Submodules
#include "ITKHelpers/ITKHelpers.h"
#include "EigenHelpers/EigenHelpers.h"
#include "ITKHelpers/Helpers/ParallelSort.h"

template <typename TMatrixType, typename TVectorType>
template <typename TImage>
TVectorType PatchProjection<TMatrixType, TVectorType>::VectorizePatch(const TImage* const image, const itk::ImageRegion<2>& region)
{
  TVectorType vectorized =
       TVectorType::Zero(image->GetNumberOfComponentsPerPixel() * region.GetNumberOfPixels());

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

template <typename TMatrixType, typename TVectorType>
template <typename TImage>
void PatchProjection<TMatrixType, TVectorType>::UnvectorizePatch(const TVectorType& vectorized,
                                                                 TImage* const image,
                                                                 const unsigned int channels)
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
      typename TVectorType::Scalar value = vectorized[channels * pixelCounter + component];
      // Make sure that the scalar value is within the range of the image pixel type. For example, we can't
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

template <typename TMatrixType, typename TVectorType>
template <typename TImage>
TMatrixType PatchProjection<TMatrixType, TVectorType>::VectorizeImage(const TImage* const image, const unsigned int patchRadius)
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
  TMatrixType featureMatrix(featureLength, numberOfPatches);
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

template <typename TMatrixType, typename TVectorType>
template <typename TImage>
TMatrixType PatchProjection<TMatrixType, TVectorType>::GetDummyProjectionMatrix(const TImage* const image, const unsigned int patchRadius,
                                                          TVectorType& meanVector, TVectorType& standardDeviationVector)
{
  unsigned int numberOfComponentsPerPixel = image->GetNumberOfComponentsPerPixel();
  unsigned int pixelsPerPatch = (patchRadius * 2 + 1) * (patchRadius * 2 + 1);
  unsigned int featureLength = numberOfComponentsPerPixel * pixelsPerPatch;

  TMatrixType dummyProjectionMatrix(featureLength, featureLength);
  dummyProjectionMatrix.setIdentity();

  meanVector.resize(featureLength);
  meanVector.setZero();

  standardDeviationVector.resize(featureLength);
  standardDeviationVector.setOnes();

  return dummyProjectionMatrix;
}

template <typename TMatrixType, typename TVectorType>
template <typename TImage>
TMatrixType PatchProjection<TMatrixType, TVectorType>::GetDummyProjectionMatrix(const TImage* const image, const unsigned int patchRadius)
{
  TVectorType meanVector;
  TVectorType standardDeviationVector;
  return GetDummyProjectionMatrix(image, patchRadius, meanVector, standardDeviationVector);
}

template <typename TMatrixType, typename TVectorType>
template <typename TImage>
TMatrixType PatchProjection<TMatrixType, TVectorType>::GetDummyProjectionMatrix(const TImage* const image, const unsigned int patchRadius,
                                            TVectorType& meanVector, std::vector<typename TVectorType::Scalar>& sortedEigenvalues)
{
  unsigned int numberOfComponentsPerPixel = image->GetNumberOfComponentsPerPixel();
  unsigned int pixelsPerPatch = (patchRadius * 2 + 1) * (patchRadius * 2 + 1);
  unsigned int featureLength = numberOfComponentsPerPixel * pixelsPerPatch;

  TMatrixType dummyProjectionMatrix(featureLength, featureLength);
  dummyProjectionMatrix.setIdentity();

  meanVector.resize(featureLength);
  meanVector.setZero();

  sortedEigenvalues.resize(featureLength, 1.0f);

  return dummyProjectionMatrix;
}

template <typename TMatrixType, typename TVectorType>
template <typename TImage>
TMatrixType PatchProjection<TMatrixType, TVectorType>::ComputeProjectionMatrix(const TImage* const image, const unsigned int patchRadius)
{
  TVectorType meanVector;
  TVectorType standardDeviationVector;
  return ComputeProjectionMatrix_CovarianceEigen(image, patchRadius, meanVector, standardDeviationVector);
}

template <typename TMatrixType, typename TVectorType>
template <typename TImage>
TMatrixType PatchProjection<TMatrixType, TVectorType>::ComputeProjectionMatrix_CovarianceEigen(const TImage* const image, const unsigned int patchRadius,
                                                         TVectorType& meanVector, TVectorType& standardDeviationVector)
{
  throw std::runtime_error("Not yet implemented.");
}

template <typename TMatrixType, typename TVectorType>
template <typename TImage>
TMatrixType PatchProjection<TMatrixType, TVectorType>::ComputeProjectionMatrix_CovarianceEigen
                                                        (const TImage* const image,
                                                         const unsigned int patchRadius,
                                                         TVectorType& meanVector,
                                                         std::vector<typename TVectorType::Scalar>& sortedEigenvalues)
{
  std::vector<itk::ImageRegion<2> > allPatches = ITKHelpers::GetAllPatches(image->GetLargestPossibleRegion(), patchRadius);
  TMatrixType featureMatrix = PatchProjection::VectorizeImage(image, patchRadius);

  // Standardize the vectorized patches, and store the meanVector
  // used to do so for later un-standardization
  meanVector = featureMatrix.rowwise().mean();

  std::cout << "meanVector: " << meanVector << std::endl;
  
  // Subtract the mean vector from every column
  featureMatrix.colwise() -= meanVector;

  TMatrixType covarianceMatrix = EigenHelpers::ConstructCovarianceMatrixFromFeatureMatrix(featureMatrix);

  std::cout << "Done computing covariance matrix (" << covarianceMatrix.rows() << " x " << covarianceMatrix.cols() << ")" << std::endl;

  Eigen::SelfAdjointEigenSolver<TMatrixType> eigensolver(covarianceMatrix);
  if (eigensolver.info() != Eigen::Success)
  {
    throw std::runtime_error("Eigen decomposition of the covariance matrix failed!");
  }

  // Sort eigenvectors by increasing eigenvalue magnitude.

  std::vector<float> eigenvalueMagnitudes(eigensolver.eigenvalues().size());
  for(int i = 0; i < eigensolver.eigenvalues().size(); ++i)
  {
    if(eigensolver.eigenvalues()[i] < 0)
    {
      std::stringstream ss;
      ss << "Eigenvalue cannot be negative! " << eigensolver.eigenvalues()[i];
      std::runtime_error(ss.str());
    }
    eigenvalueMagnitudes[i] = fabs(eigensolver.eigenvalues()[i]);
  }

  // Sort the eigenvalues from largest magnitude to smallest
  std::vector<ParallelSort::IndexedValue<float> > sorted = ParallelSort::ParallelSortDescending<float>(eigenvalueMagnitudes);

  // Write eigenvalue magnitudes to file (for plotting later)
//   //std::vector<float> sortedEigenvalues(sorted.size());
//   sortedEigenvalues.resize(sorted.size());
//   for(unsigned int i = 0; i < sorted.size(); ++i)
//   {
//     sortedEigenvalues[i] = sorted[i].value;
//   }
//   Helpers::WriteVectorToFile(sortedEigenvalues, "eigenvalues.txt");
//   std::cout << "Wrote eigenvalues." << std::endl;

  sortedEigenvalues.resize(sorted.size());
  std::cout << "Eigenvalues: ";
  for(unsigned int i = 0; i < sorted.size(); ++i)
  {
    sortedEigenvalues[i] = eigensolver.eigenvalues()[sorted[i].index];
    std::cout << sortedEigenvalues[i] << " ";
  }
  std::cout << std::endl;
  
  Helpers::WriteVectorToFile(sortedEigenvalues, "eigenvalues.txt");
  std::cout << "Wrote eigenvalues." << std::endl;

  // Reorder the eigenvectors
  TMatrixType sortedEigenVectors(eigensolver.eigenvectors().rows(), eigensolver.eigenvectors().cols());
  for(size_t i = 0; i < sorted.size(); ++i)
  {
    sortedEigenVectors.col(i) = eigensolver.eigenvectors().col(sorted[i].index);
  }

  return sortedEigenVectors;
}

template <typename TMatrixType, typename TVectorType>
template <typename TImage>
TMatrixType PatchProjection<TMatrixType, TVectorType>::ComputeProjectionMatrix_SVD(const TImage* const image, const unsigned int patchRadius,
                                                         TVectorType& meanVector, TVectorType& standardDeviationVector)
{
  std::vector<itk::ImageRegion<2> > allPatches = ITKHelpers::GetAllPatches(image->GetLargestPossibleRegion(), patchRadius);
  TMatrixType featureMatrix = PatchProjection::VectorizeImage(image, patchRadius);

  // Standardize the vectorized patches, and store the meanVector and standardDeviationVector
  // used to do so for later un-standardization
  meanVector = featureMatrix.rowwise().mean();
  // Subtract the mean vector from every column
  featureMatrix.colwise() -= meanVector;

  // The variance is computed as 1/N \sum (x_i - x_mean)^2 . Since we have zero mean, this is just the square of the components
  TMatrixType squaredMean0FeatureMatrix = featureMatrix.array().pow(2); // Square all components
  TVectorType variance = squaredMean0FeatureMatrix.rowwise().mean();
  standardDeviationVector = variance.array().sqrt(); // Take the square root of all components

  // Divide by the standard devation
  // featureMatrix.colwise() /= standardDeviation; // this does not yet work in Eigen
  featureMatrix = standardDeviationVector.matrix().asDiagonal().inverse() * featureMatrix;

  typedef Eigen::JacobiSVD<TMatrixType> SVDType;
  //SVDType svd(featureMatrix, Eigen::ComputeFullU);
  SVDType svd(featureMatrix, Eigen::ComputeThinU);
  return svd.matrixU();
}

#endif
