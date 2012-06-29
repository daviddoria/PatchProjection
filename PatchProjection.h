#ifndef PatchProjection_H
#define PatchProjection_H

// ITK
#include "itkVectorImage.h"

// Eigen
#include <Eigen/Dense>

template <typename TMatrixType, typename TVectorType>
class PatchProjection
{
public:

  /** Convert EVERY region of an image to a vector.*/
  template <typename TImage>
  static TMatrixType VectorizeImage(const TImage* const image, const unsigned int patchRadius);

  /** Convert a region of an image to a vector.*/
  template <typename TImage>
  static TVectorType VectorizePatch(const TImage* const image, const itk::ImageRegion<2>& region);

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
    * and return the 'U' matrix from the SVD. */
  template <typename TImage>
  static TMatrixType ComputeProjectionMatrix_CovarianceEigen(const TImage* const image, const unsigned int patchRadius,
                                                 TVectorType& meanVector);

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
};

#include "PatchProjection.hpp"

#endif
