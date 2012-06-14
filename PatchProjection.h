#ifndef PatchProjection_H
#define PatchProjection_H

// ITK
#include "itkVectorImage.h"

// Eigen
#include <Eigen/Dense>

class PatchProjection
{
public:

  template <typename TImage>
  static Eigen::MatrixXf VectorizeImage(const TImage* const image, const unsigned int patchRadius);

  template <typename TImage>
  static Eigen::VectorXf VectorizePatch(const TImage* const image, const itk::ImageRegion<2>& region);

  template <typename TImage>
  static void UnvectorizePatch(const Eigen::VectorXf& vectorized, const TImage* const image,
                               const unsigned int channels);

  template <typename TImage>
  static Eigen::MatrixXf ComputeProjectionMatrix(const TImage* const image, const unsigned int patchRadius);

};

#include "PatchProjection.hpp"

#endif
