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

  std::vector<itk::ImageRegion<2> > allPatches = ITKHelpers::GetAllPatches(image->GetLargestPossibleRegion(), patchRadius);
  Eigen::MatrixXf featureMatrix = PatchProjection::VectorizeImage(image, patchRadius);

  // Standardize the vectorized patches, and store the meanVector and standardDeviationVector
  // used to do so for later un-standardization
  Eigen::VectorXf meanVector = featureMatrix.rowwise().mean();
  // Subtract the mean vector from every column
  featureMatrix.colwise() -= meanVector;

  Eigen::MatrixXf squaredMean0FeatureMatrix = featureMatrix.array().pow(2);
  Eigen::VectorXf variance = squaredMean0FeatureMatrix.rowwise().mean();
  Eigen::VectorXf standardDeviation = variance.array().sqrt();

  // Divide by the standard devation
  // featureMatrix.colwise() /= standardDeviation; // this does not yet work in Eigen
  featureMatrix = standardDeviation.matrix().asDiagonal().inverse() * featureMatrix;

  Eigen::MatrixXf covarianceMatrix = EigenHelpers::ConstructCovarianceMatrixFromFeatureMatrix(featureMatrix);

  std::cout << "Done computing covariance matrix (" << covarianceMatrix.rows() << " x " << covarianceMatrix.cols() << ")" << std::endl;

  // Use the first vector for testing
  Eigen::VectorXf v = featureMatrix.col(0);

  typedef Eigen::JacobiSVD<Eigen::MatrixXf> SVDType;
  //SVDType svd(covarianceMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  SVDType svd(covarianceMatrix, Eigen::ComputeFullU);

  //unsigned int numberOfColumnsToKeep = 10;
  unsigned int numberOfColumnsToKeep = featureMatrix.rows(); // for now, keep the entire amount of information

  Eigen::VectorXf projectedVector =
          EigenHelpers::DimensionalityReduction(v, svd.matrixU(), numberOfColumnsToKeep);

  // This is the matrix that produced the projection:
  Eigen::MatrixXf truncatedU = EigenHelpers::TruncateColumns(svd.matrixU(), numberOfColumnsToKeep);

  Eigen::MatrixXf inverseProjection = EigenHelpers::PseudoInverse(truncatedU);

  Eigen::VectorXf unprojectedVector = inverseProjection * projectedVector;

  std::cout << "original: " << v << std::endl;
  std::cout << "unprojectedVector: " << unprojectedVector << std::endl;

  {
//   Eigen::VectorXf projectedVectorAtLeastEnergy =
//           EigenHelpers::DimensionalityReduction(v, svd.matrixU(), svd.singularValues(), 0.5);
  }

  return EXIT_SUCCESS;
}
