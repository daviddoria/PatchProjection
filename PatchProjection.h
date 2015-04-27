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

  /** Compute the eigendecomposition of the covariance matrix and return the matrix of eigenvectors (as columns)
    * sorted in order of decreasing corresponding eigenvalues.*/
  static TMatrixType SortedEigenDecomposition(const TMatrixType& covarianceMatrix,
                                              std::vector<typename TVectorType::Scalar>& sortedEigenvalues);

  /** Compute a "feature matrix" by convert every region of radius 'patchRadius' of an image
    * inside of 'region' into a vector and storing
    * it as a column of the matrix.*/
  template <typename TImage>
  static TMatrixType VectorizeImage(const TImage* const image, const unsigned int patchRadius,
                                    const itk::ImageRegion<2>& region);

  /** Compute a "feature matrix" by convert every region of radius 'patchRadius' of an image
    * into a vector and storing
    * it as a column of the matrix.*/
  template <typename TImage>
  static TMatrixType VectorizeImage(const TImage* const image, const unsigned int patchRadius);

  /** Convert a region of a vector image to a vector.*/
  template <typename TPixel>
  static TVectorType VectorizePatch(const itk::VectorImage<TPixel, 2>* const image,
                                    const itk::ImageRegion<2>& region);

  /** Convert a region of a vector image to a vector.*/
  template <typename TPixel, unsigned int PixelDimension>
  static TVectorType VectorizePatch(const itk::Image<itk::CovariantVector<TPixel, PixelDimension>, 2>* const image,
                                    const itk::ImageRegion<2>& region);

  /** Convert a region of a scalar image to a vector.*/
  template <typename TPixel>
  static TVectorType VectorizePatch(const itk::Image<TPixel, 2>* const image, const itk::ImageRegion<2>& region);

  /** Given a vectorized patch, convert it back to an image.*/
  template <typename TImage>
  static void UnvectorizePatch(const TVectorType& vectorized, TImage* const image,
                               const unsigned int channels);

  /** Vectorize the entire image, construct a feature matrix, perform an SVD on the
    * covariance matrix of the feature matrix,
    * and return the 'U' matrix from the SVD. */
  template <typename TImage>
  static TMatrixType ComputeProjectionMatrix_CovarianceEigen(const TImage* const image,
                                                             const unsigned int patchRadius,
                                                             TVectorType& meanVector,
                                                             TVectorType& standardDeviationVector);

  /** Perform an SVD on the covariance matrix of the 'featureMatrix',
    * and return the 'U' matrix from the SVD. Return the 'meanVector'
    * by reference, and returns the eigenvalues by
    * reference in 'sortedEigenvalues'. */
  static TMatrixType ProjectionMatrixFromFeatureMatrix(const TMatrixType& featureMatrix,
                                                TVectorType& meanVector,
                                                std::vector<typename TVectorType::Scalar>& sortedEigenvalues);

  /** Perform an SVD on the covariance matrix of the 'featureMatrix',
    * and return the 'U' matrix from the SVD. Return the 'meanVector' by reference. */
  static TMatrixType ProjectionMatrixFromFeatureMatrix(const TMatrixType& featureMatrix,
                                                TVectorType& meanVector);

  /** Perform an SVD on the covariance matrix of the 'featureMatrix',
    * and return the 'U' matrix from the SVD. This function simply calls the function of
    * the same name with dummy arguments. */
  static TMatrixType ProjectionMatrixFromFeatureMatrix(const TMatrixType& featureMatrix);

  /** Vectorize the entire image, construct a feature matrix, perform an SVD on
    * the covariance matrix of the feature matrix,
    * and return the 'U' matrix from the SVD. Return the 'meanVector' by reference,
    * and returns the eigenvalues by reference in 'sortedEigenvalues'. */
  template <typename TImage>
  static TMatrixType ComputeProjectionMatrix_CovarianceEigen(const TImage* const image,
                                                             const unsigned int patchRadius,
                                                 TVectorType& meanVector,
                                                 std::vector<typename TVectorType::Scalar>& sortedEigenvalues);

  /** When the data is too big to compute a feature matrix, we can compute the
   *  covariance matrix directly from the image
    * by computing it one element at a time. It is hundreds of times slower (694m43.748s vs 4s for the
    * full matrix multiplication).
    * Return the 'U' matrix from the eigendecomposition. Return the 'meanVector' by reference, and returns the
    * eigenvalues by reference in 'sortedEigenvalues'. */
  template <typename TImage>
  static TMatrixType ComputeProjectionMatrixFromImageElementWise(const TImage* const image,
                                                                 const unsigned int patchRadius,
                                                      TVectorType& meanVector,
                                                      std::vector<typename TVectorType::Scalar>& sortedEigenvalues);

  /** When the data is too big to compute a feature matrix, we can compute the covariance
    * matrix directly from the image
    * by computing it by summing the outer product of one column at a time.
    * though it is much slower (1m41s vs 3s for the full matrix product technique).
    * Return the 'U' matrix from the eigendecomposition. Return the 'meanVector' by reference, and returns the
    * eigenvalues by reference in 'sortedEigenvalues'. */
  template <typename TImage>
  static TMatrixType ComputeProjectionMatrixFromImageOuterProduct(const TImage* const image,
                                                                  const unsigned int patchRadius,
                                                      TVectorType& meanVector,
                                                      std::vector<typename TVectorType::Scalar>& sortedEigenvalues);

  /** When the data is too big to compute a feature matrix, we can compute the
    * covariance matrix directly from the image
    * by breaking it into blocks and summing the products of the blocks.
    * Return the 'U' matrix from the eigendecomposition. Return the 'meanVector' by reference, and returns the
    * eigenvalues by reference in 'sortedEigenvalues'. */
  template <typename TImage>
  static TMatrixType ComputeProjectionMatrixFromImagePartialMatrix(const TImage* const image,
                                                                   const unsigned int patchRadius,
                                                      TVectorType& meanVector,
                                                      std::vector<typename TVectorType::Scalar>& sortedEigenvalues);

  /** Vectorize the entire image, construct a feature matrix, perform an SVD
    * on the covariance matrix of the feature matrix,
    * and return the 'U' matrix from the SVD. */
  template <typename TImage>
  static TMatrixType ComputeProjectionMatrix_SVD(const TImage* const image, const unsigned int patchRadius,
                                                 TVectorType& meanVector, TVectorType& standardDeviationVector);

  /** If the meanVector and standardDeviationVector are not needed in the caller,
    * this overload provides a cleaner interface. */
  template <typename TImage>
  static TMatrixType ComputeProjectionMatrix(const TImage* const image, const unsigned int patchRadius);

  /** This function exists to return a matrix of the correct dimensions to be used for testing. */
  template <typename TImage>
  static TMatrixType GetDummyProjectionMatrix(const TImage* const image, const unsigned int patchRadius);

  template <typename TImage>
  static TMatrixType GetDummyProjectionMatrix(const TImage* const image, const unsigned int patchRadius,
                                              TVectorType& meanVector);

  template <typename TImage>
  static TMatrixType GetDummyProjectionMatrix(const TImage* const image, const unsigned int patchRadius,
                                              TVectorType& meanVector, TVectorType& standardDeviationVector);

  template <typename TImage>
  static TMatrixType GetDummyProjectionMatrix(const TImage* const image, const unsigned int patchRadius,
                                              TVectorType& meanVector,
                                              std::vector<typename TVectorType::Scalar>& sortedEigenvalues);
};

#include "PatchProjection.hpp"

#endif
