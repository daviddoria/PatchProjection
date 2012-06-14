#ifndef PatchProjection_hpp
#define PatchProjection_hpp

#include "PatchProjection.h" // appease syntax parser

#include "itkImageRegionConstIterator.h"

#include "ITKHelpers/ITKHelpers.h"

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
void PatchProjection::UnvectorizePatch(const Eigen::VectorXf& vectorized, const TImage* const image, unsigned int channels)
{
  // This function assumes the patch is square
  image->SetNumberOfComponentsPerPixel(channels);

  itk::Size<2> size;
  size.Fill(vectorized.size()/channels);

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
      pixel[component] = vectorized[channels * pixelCounter + component];
      }
    imageIterator.Set(pixel);
    pixelCounter++;
    ++imageIterator;
    }
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

  Eigen::MatrixXf featureMatrix(featureLength, numberOfPatches);

  itk::ImageRegionConstIterator<TImage> imageIterator(image, image->GetLargestPossibleRegion());

  std::vector<itk::ImageRegion<2> > allPatches = ITKHelpers::GetAllPatches(image->GetLargestPossibleRegion(), patchRadius);

  unsigned int patchCounter = 0;
  while(!imageIterator.IsAtEnd())
    {
    featureMatrix.col(patchCounter) = VectorizePatch(image, allPatches[patchCounter]);
    patchCounter++;
    ++imageIterator;
    }
  return featureMatrix;
}

#endif
