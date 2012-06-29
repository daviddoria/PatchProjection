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
    std::cerr << "Provided:" << std::endl;
    for(int i = 1; i < argc; ++i)
      {
      std::cerr << argv[i] << " ";
      }
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

  Eigen::MatrixXf projectionMatrix = PatchProjection<Eigen::MatrixXf, Eigen::VectorXf>::ComputeProjectionMatrix(image, patchRadius);

  itk::Index<2> corner = {{0,0}};
  itk::Size<2> size = {{patchRadius*2 + 1, patchRadius*2 + 1}};
  itk::ImageRegion<2> cornerRegion(corner, size);
  Eigen::VectorXf patch = PatchProjection<Eigen::MatrixXf, Eigen::VectorXf>::VectorizePatch(image, cornerRegion);

  //unsigned int numberOfColumnsToKeep = 10;
  unsigned int numberOfColumnsToKeep = projectionMatrix.rows(); // for now, keep the entire amount of information

  Eigen::VectorXf projectedVector =
          EigenHelpers::DimensionalityReduction(patch, projectionMatrix, numberOfColumnsToKeep);

  // This is the matrix that produced the projection:
  Eigen::MatrixXf truncatedU = EigenHelpers::TruncateColumns(projectionMatrix, numberOfColumnsToKeep);

  Eigen::MatrixXf inverseProjection = EigenHelpers::PseudoInverse(truncatedU);

  Eigen::VectorXf unprojectedVector = inverseProjection * projectedVector;

  std::cout << "original: " << patch << std::endl;
  std::cout << "unprojectedVector: " << unprojectedVector << std::endl;

  {
//   Eigen::VectorXf projectedVectorAtLeastEnergy =
//           EigenHelpers::DimensionalityReduction(v, svd.matrixU(), svd.singularValues(), 0.5);
  }

  return EXIT_SUCCESS;
}
