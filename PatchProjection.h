#ifndef PatchProjection_H
#define PatchProjection_H

// ITK
#include "itkVectorImage.h"

// Eigen
#include <Eigen/Dense>

class PatchProjection
{
public:

  /** Convert EVERY region of an image to a vector.*/
  template <typename TImage>
  static Eigen::MatrixXf VectorizeImage(const TImage* const image, const unsigned int patchRadius);

  /** Convert a region of an image to a vector.*/
  template <typename TImage>
  static Eigen::VectorXf VectorizePatch(const TImage* const image, const itk::ImageRegion<2>& region);

  /** Given a vectorized patch, convert it back to an image.*/
  template <typename TImage>
  static void UnvectorizePatch(const Eigen::VectorXf& vectorized, TImage* const image,
                               const unsigned int channels);

  /** Vectorize the entire image, construct a feature matrix, perform an SVD on the covariance matrix of the feature matrix,
    * and return the 'U' matrix from the SVD. */
  template <typename TImage>
  static Eigen::MatrixXf ComputeProjectionMatrix(const TImage* const image, const unsigned int patchRadius,
                                                 Eigen::VectorXf& meanVector, Eigen::VectorXf& standardDeviationVector);

  /** If the meanVector and standardDeviationVector are not needed in the caller, this overload provides a cleaner interface. */
  template <typename TImage>
  static Eigen::MatrixXf ComputeProjectionMatrix(const TImage* const image, const unsigned int patchRadius);

  /** This function exists to return a matrix of the correct dimensions to be used for testing. */
  template <typename TImage>
  static Eigen::MatrixXf GetDummyProjectionMatrix(const TImage* const image, const unsigned int patchRadius);
};

#include "PatchProjection.hpp"

#endif
