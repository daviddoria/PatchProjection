#ifndef PatchProjection_H
#define PatchProjection_H

// ITK
#include "itkImage.h"
#include "itkVectorImage.h"

// Eigen
#include <Eigen/Dense>

template <typename TMatrixType, typename TVectorType>
class PatchProjection
{
public:

  /** Convert every region of an image to a vector.*/
  template <typename TImage>
  static TMatrixType VectorizeImage(const TImage* const image, const unsigned int patchRadius);

  /** Convert a region of a vector image to a vector.*/
  template <typename TPixel>
  static TVectorType VectorizePatch(const itk::VectorImage<TPixel, 2>* const image, const itk::ImageRegion<2>& region);

  /** Convert a region of a scalar image to a vector.*/
  template <typename TPixel>
  static TVectorType VectorizePatch(const itk::Image<TPixel, 2>* const image, const itk::ImageRegion<2>& region);
  
  /** Given a vectorized patch, convert it back to an image.*/
  template <typename TImage>
  static void UnvectorizePatch(const TVectorType& vectorized, TImage* const image,
                               const unsigned int channels);

  /** Vectorize the entire image, construct a feature matrix, perform an SVD on the covariance matrix of the feature matrix,
    * and return the 'U' matrix from the SVD. */
  template <typename TImage>
  static TMatrixType ComputeProjectionMatrix_CovarianceEigen(const TImage* const image, const unsigned int patchRadius,
                                                 TVectorType& meanVector, TVectorType& standardDeviationVector);

  /** Vectorize the entire image, construct a feature matrix, perform an SVD on the covariance matrix of the feature matrix,
    * and return the 'U' matrix from the SVD. Return the 'meanVector' by reference, and returns the eigenvalues by
    * reference in 'sortedEigenvalues'. */
  template <typename TImage>
  static TMatrixType ComputeProjectionMatrix_CovarianceEigen(const TImage* const image, const unsigned int patchRadius,
                                                 TVectorType& meanVector, std::vector<typename TVectorType::Scalar>& sortedEigenvalues);

  /** When the data is too big to compute a feature matrix, we can compute the covariance matrix directly from the image,
    * though it is much slower.
    * Return the 'U' matrix from the eigendecomposition. Return the 'meanVector' by reference, and returns the
    * eigenvalues by reference in 'sortedEigenvalues'. */
  template <typename TImage>
  static TMatrixType ComputeProjectionMatrixFromImage(const TImage* const image, const unsigned int patchRadius,
                                                      TVectorType& meanVector, std::vector<typename TVectorType::Scalar>& sortedEigenvalues);
  
  /** Vectorize the entire image, construct a feature matrix, perform an SVD on the covariance matrix of the feature matrix,
    * and return the 'U' matrix from the SVD. */
  template <typename TImage>
  static TMatrixType ComputeProjectionMatrix_SVD(const TImage* const image, const unsigned int patchRadius,
                                                 TVectorType& meanVector, TVectorType& standardDeviationVector);

  /** If the meanVector and standardDeviationVector are not needed in the caller, this overload provides a cleaner interface. */
  template <typename TImage>
  static TMatrixType ComputeProjectionMatrix(const TImage* const image, const unsigned int patchRadius);

  /** This function exists to return a matrix of the correct dimensions to be used for testing. */
  template <typename TImage>
  static TMatrixType GetDummyProjectionMatrix(const TImage* const image, const unsigned int patchRadius);

  template <typename TImage>
  static TMatrixType GetDummyProjectionMatrix(const TImage* const image, const unsigned int patchRadius,
                                              TVectorType& meanVector, TVectorType& standardDeviationVector);

  template <typename TImage>
  static TMatrixType GetDummyProjectionMatrix(const TImage* const image, const unsigned int patchRadius,
                                              TVectorType& meanVector, std::vector<typename TVectorType::Scalar>& sortedEigenvalues);
};

#include "PatchProjection.hpp"

#endif
