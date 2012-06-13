#include <iostream>
#include <vector>

#include "PatchProjection.h"

// Submodules
#include "PatchProjection/ITKHelpers/Helpers/Helpers.h"
#include "PatchProjection/ITKHelpers/ITKHelpers.h"
#include "PatchProjection/EigenHelpers/EigenHelpers.h"

int main(int argc, char* argv[])
{
  if(argc < 4)
  {
    std::cerr << "Required arguments: inputFileName patchRadius dimensions" << std::endl;
    return EXIT_FAILURE;
  }

  std::stringstream ss;
  for(int i = 1; i < argc; ++i)
  {
    ss << argv[i] << " ";
  }
  std::cout << ss.str() << std::endl;

  std::string inputFileName;
  unsigned int patchRadius;
  float percentOfSingularWeightToKeep;
  ss >> inputFileName >> patchRadius >> percentOfSingularWeightToKeep;

  std::cout << "Arguments:" << std::endl
            << "Filename: " << inputFileName << std::endl
            << "patchRadius = " << patchRadius << std::endl
            << "percentOfSingularWeightToKeep = " << percentOfSingularWeightToKeep << std::endl;

  //typedef itk::VectorImage<float, 2> ImageType;
  typedef itk::Image<itk::CovariantVector<float, 3>, 2> ImageType;

  typedef itk::ImageFileReader<ImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(inputFileName);
  reader->Update();

  ImageType* image = reader->GetOutput();

  ////////// Project all of the patches using the covariance matrix constructed from the downsampled set /////

  std::vector<itk::ImageRegion<2> > allPatches = ITKHelpers::GetAllPatches(image->GetLargestPossibleRegion(), patchRadius);

  EigenHelpers::VectorOfVectors vectorizedPatches(allPatches.size());

  // Vectorize all of the patches
  for(unsigned int i = 0; i < allPatches.size(); ++i)
  {
    // Vectorize the RGB values
    Eigen::VectorXf vectorized = PatchProjection::VectorizePatch(image, allPatches[i]);
    if(Helpers::ContainsNaN(vectorized))
    {
      throw std::runtime_error("vectorized contains NaNs!");
    }

    vectorizedPatches[i] = vectorized;
  }

  std::cout << "Done vectorizing " << allPatches.size() << " patches." << std::endl;

  // Standardize the vectorized patches, and store the meanVector and standardDeviationVector used to do so for later un-standardization
  Eigen::VectorXf meanVector;
  Eigen::VectorXf standardDeviationVector;
  EigenHelpers::Standardize(vectorizedPatches, meanVector, standardDeviationVector);
  std::cout << "Done standardizing " << allPatches.size() << " patches." << std::endl;

  // Timings are with 280116 patches (radius=7)
  //Eigen::MatrixXf covarianceMatrix = EigenHelpers::ConstructCovarianceMatrixZeroMean(vectorizedPatches); //10m18.109s with printfs
  //Eigen::MatrixXf covarianceMatrix = EigenHelpers::ConstructCovarianceMatrixZeroMean(vectorizedPatches); //9m5.624s without printfs
  Eigen::MatrixXf covarianceMatrix = EigenHelpers::ConstructCovarianceMatrixZeroMeanFast(vectorizedPatches); // Only 0m34.123s ! 20x faster!

  std::cout << "Done computing covariance matrix (" << covarianceMatrix.rows() << " x " << covarianceMatrix.cols() << ")" << std::endl;

  exit(-1);
//   EigenHelpers::VectorOfVectors projectedVectors =
//           EigenHelpers::DimensionalityReduction(vectorizedPatches, covarianceMatrix, dimensions);

  EigenHelpers::VectorOfVectors projectedVectors =
          EigenHelpers::DimensionalityReduction(vectorizedPatches, covarianceMatrix, percentOfSingularWeightToKeep);

  std::cout << "There are " << projectedVectors.size() << " projectedVectors with "
            << projectedVectors[0].size() << " components each." << std::endl;
  covarianceMatrix.resize(0,0); // Free the memory

  /////////////////////
  itk::CovariantVector<float, 3> zeroVector;
  zeroVector.Fill(0);

  typedef itk::Image<itk::CovariantVector<float, 3>, 2> OutputImageType; // (x, y, score)

  OutputImageType::Pointer locationField = OutputImageType::New();
  locationField->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
  locationField->Allocate();
  locationField->FillBuffer(zeroVector);

  OutputImageType::Pointer offsetField = OutputImageType::New();
  offsetField->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
  offsetField->Allocate();
  zeroVector.Fill(0);
  offsetField->FillBuffer(zeroVector);

  float distance = 0.0f;

  for(unsigned int i = 0; i < projectedVectors.size(); ++i)
  {
    //std::cout << i << " of " << projectedVectors.size() << std::endl;
    printf ("%d of %d\n", i, projectedVectors.size());
    float minDistance = std::numeric_limits<float>::max();
    unsigned int bestId = 0;

    for(unsigned int j = 0; j < projectedVectors.size(); ++j)
    {
      // Don't compare a patch to itself
      if(i == j)
      {
        continue;
      }

      distance = (projectedVectors[i] - projectedVectors[j]).squaredNorm();

      if(distance < minDistance)
      {
        minDistance = distance;
        bestId = j;
      }

    } // end loop j


    itk::Index<2> patchCenter = ITKHelpers::GetRegionCenter(allPatches[i]);
    itk::Index<2> bestMatchCenter = ITKHelpers::GetRegionCenter(allPatches[bestId]);

    // Location
    itk::CovariantVector<float, 3> locationPixel;
    locationPixel[0] = bestMatchCenter[0];
    locationPixel[1] = bestMatchCenter[1];
    locationPixel[2] = minDistance;

    locationField->SetPixel(patchCenter, locationPixel);

    // Offset
    itk::Offset<2> offset = bestMatchCenter - patchCenter;

    itk::CovariantVector<float, 3> offsetPixel;
    offsetPixel[0] = offset[0];
    offsetPixel[1] = offset[1];
    offsetPixel[2] = minDistance;

    offsetField->SetPixel(patchCenter, offsetPixel);
  } // end loop i

  std::stringstream ssLocation;
  ssLocation << "Projected_Location_" << patchRadius << "_" << percentOfSingularWeightToKeep << ".mha";
  ITKHelpers::WriteImage(locationField.GetPointer(), ssLocation.str());

  std::stringstream ssOffset;
  ssOffset << "Projected_Offset_" << patchRadius << "_" << percentOfSingularWeightToKeep << ".mha";
  ITKHelpers::WriteImage(offsetField.GetPointer(), ssOffset.str());

  return EXIT_SUCCESS;
}

#if 0
void Downsample()
{

  //////////// Compute the covariance matrix from a downsampled set of patches ////////////////////

  // This shouldn't actually speed up that much, because the covariance matrix (and hence SVD)
  // is based on the dimensionality of the vector, not the number of vectors used.
  //unsigned int downsampleFactor = 10;
  //unsigned int downsampleFactor = 5;
  unsigned int downsampleFactor = patchRadius;
  std::vector<itk::Index<2> > downsampledIndices =
         ITKHelpers::GetDownsampledIndicesInRegion(image->GetLargestPossibleRegion(), downsampleFactor);
  std::vector<itk::ImageRegion<2> > downsampledPatches =
         ITKHelpers::GetValidPatchesCenteredAtIndices(downsampledIndices,
                                                      image->GetLargestPossibleRegion(), patchRadius);

  //std::vector<itk::ImageRegion<2> > allPatches =
            //ITKHelpers::GetAllPatches(reader->GetOutput()->GetLargestPossibleRegion(), patchRadius);
  std::cout << "There are " << downsampledPatches.size() << " downsampled patches." << std::endl;

  EigenHelpers::VectorOfVectors vectorizedDownsampledPatches(downsampledPatches.size());

  unsigned int numberOfHistogramBins = 10;

  // Vectorized a subset of the patches
  for(unsigned int i = 0; i < downsampledPatches.size(); ++i)
  {
    // Vectorize the RGB values
    Eigen::VectorXf vectorized = PatchProjection::VectorizePatch(image, downsampledPatches[i]);

    if(Helpers::ContainsNaN(vectorized))
    {
      throw std::runtime_error("vectorized contains NaNs!");
    }

    vectorizedDownsampledPatches[i] = vectorized;
  }

  std::cout << "There are " << vectorizedDownsampledPatches.size() << " vectorizedDownsampledPatches." << std::endl;

  std::cout << "Each vector has " << vectorizedDownsampledPatches[0].size() << " components." << std::endl;

}

#endif
