#ifndef PatchProjection_hpp
#define PatchProjection_hpp

#include "PatchProjection.h" // appease syntax parser

#include "itkImageRegionConstIterator.h"

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
static void PatchProjection::UnvectorizePatch(const Eigen::VectorXf& vectorized, const TImage* const image, unsigned int channels)
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

#endif
