#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef _WIN32
#define CERES_MSVC_USE_UNDERSCORE_PREFIXED_BESSEL_FUNCTIONS
#pragma warning(push)
#pragma warning(disable : 4100)
#pragma warning(disable : 4996)
#pragma warning(disable : 4127)
#endif
#include <ceres/ceres.h>
#ifdef _WIN32
#pragma warning(pop)
#endif

struct KannalaBrandtParams {
  double fx, fy, cx, cy;
  double k1, k2, k3, k4;

  KannalaBrandtParams(double fx, double fy, double cx, double cy, double k1,
                      double k2, double k3, double k4)
      : fx(fx), fy(fy), cx(cx), cy(cy), k1(k1), k2(k2), k3(k3), k4(k4) {}
};

struct OdenCoefficients {
  double k[5]; // Oden polynomial coefficients
  double maxTheta;

  OdenCoefficients() : maxTheta(0.0) {
    for (int i = 0; i < 5; ++i) {
      k[i] = 0.0;
    }
  }
};

struct OdenCoefficientError {
  const double targetRadius;
  const double theta;
  const double imageHeight;

  OdenCoefficientError(double targetRadiusIn, double thetaIn,
                       double imageHeightIn)
      : targetRadius(targetRadiusIn), theta(thetaIn),
        imageHeight(imageHeightIn) {}

  static ceres::CostFunction *Create(double targetRadius, double theta,
                                     double imageHeight) {
    return (new ceres::AutoDiffCostFunction<OdenCoefficientError, 1, 5>(
        new OdenCoefficientError(targetRadius, theta, imageHeight)));
  }

  template <typename T>
  bool operator()(const T *const odenCoeffs, T *residuals) const {
    T theta_T = T(theta);
    T target_T = T(targetRadius);
    T height_T = T(imageHeight);

    // Oden model: r = (k[0]*θ + k[1]*θ² + k[2]*θ³ + k[3]*θ⁴ + k[4]*θ⁵) *
    // height/2
    T oden_normalized =
        odenCoeffs[0] * theta_T + odenCoeffs[1] * theta_T * theta_T +
        odenCoeffs[2] * theta_T * theta_T * theta_T +
        odenCoeffs[3] * theta_T * theta_T * theta_T * theta_T +
        odenCoeffs[4] * theta_T * theta_T * theta_T * theta_T * theta_T;

    T oden_radius = oden_normalized * height_T / T(2.0);

    residuals[0] = target_T - oden_radius;

    return true;
  }
};

double kannalaBrandtToRadius(const KannalaBrandtParams &params, double theta) {
  // Kannala-Brandt model: θ_d = θ + k1*θ³ + k2*θ⁵ + k3*θ⁷ + k4*θ⁹
  double theta_d =
      theta + params.k1 * std::pow(theta, 3) + params.k2 * std::pow(theta, 5) +
      params.k3 * std::pow(theta, 7) + params.k4 * std::pow(theta, 9);

  // Convert to radius using focal length (assuming fx and fy are similar)
  double f = (params.fx + params.fy) / 2.0;
  return f * theta_d;
}

OdenCoefficients convertKannalaBrandtToOden(const KannalaBrandtParams &kbParams,
                                            double imageHeight) {
  ceres::Solver::Options options;
  options.max_num_iterations = 500000;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;

  ceres::Problem problem;

  OdenCoefficients oden;

  // Initialize with reasonable values
  oden.k[0] = 1.0; // Linear term
  oden.k[1] = 0.0; // Quadratic term
  oden.k[2] = 0.0; // Cubic term
  oden.k[3] = 0.0; // Quartic term
  oden.k[4] = 0.0; // Quintic term

  // Generate target data points from Kannala-Brandt model
  double maxTheta = 1.5; // Maximum angle to fit (radians)
  for (double theta = 0.01; theta < maxTheta; theta += 0.01) {
    double targetRadius = kannalaBrandtToRadius(kbParams, theta);

    ceres::CostFunction *costFunction =
        OdenCoefficientError::Create(targetRadius, theta, imageHeight);

    problem.AddResidualBlock(costFunction, nullptr, oden.k);
  }

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  oden.maxTheta = maxTheta;

  std::cout << "Optimization summary:" << std::endl;
  std::cout << "Initial cost: " << summary.initial_cost << std::endl;
  std::cout << "Final cost: " << summary.final_cost << std::endl;
  std::cout << "Iterations: " << summary.iterations.size() << std::endl;
  std::cout << "Termination type: " << summary.termination_type << std::endl;
  std::cout << std::endl;

  return oden;
}

void printOdenCoefficients(const OdenCoefficients &oden,
                           const std::string &cameraName) {
  std::cout << "=== " << cameraName << " ===" << std::endl;
  std::cout << "Oden coefficients:" << std::endl;
  std::cout << std::fixed << std::setprecision(10);
  std::cout << "k[0] = " << oden.k[0] << std::endl;
  std::cout << "k[1] = " << oden.k[1] << std::endl;
  std::cout << "k[2] = " << oden.k[2] << std::endl;
  std::cout << "k[3] = " << oden.k[3] << std::endl;
  std::cout << "k[4] = " << oden.k[4] << std::endl;
  std::cout << "maxTheta = " << oden.maxTheta << std::endl;
  std::cout << std::endl;
}

int main() {
  std::cout << "Kannala-Brandt to Oden Converter" << std::endl;
  std::cout << "=================================" << std::endl;
  std::cout << std::endl;

  // Front Left camera parameters from the email
  KannalaBrandtParams frontLeft(510.6160252953, // FX
                                510.8831206311, // FY
                                963.3519332717, // CX
                                762.5888862920, // CY
                                0.1391740321,   // K1
                                -0.0428092983,  // K2
                                0.0031307063,   // K3
                                -0.0000277339   // K4
  );

  // Front camera parameters from the email
  KannalaBrandtParams front(508.9550327672, // FX
                            508.7736518651, // FY
                            958.2270523983, // CX
                            768.4881619509, // CY
                            0.1330145428,   // K1
                            -0.0330458966,  // K2
                            -0.0026217475,  // K3
                            0.0011783875    // K4
  );

  // Assuming image height (you may need to adjust this based on your actual
  // image dimensions)
  double imageHeight = 1080.0;

  // Convert Front Left camera
  std::cout << "Converting Front Left camera..." << std::endl;
  OdenCoefficients frontLeftOden =
      convertKannalaBrandtToOden(frontLeft, imageHeight);
  printOdenCoefficients(frontLeftOden, "Front Left Camera");

  // Convert Front camera
  std::cout << "Converting Front camera..." << std::endl;
  OdenCoefficients frontOden = convertKannalaBrandtToOden(front, imageHeight);
  printOdenCoefficients(frontOden, "Front Camera");

  // Print extrinsics for reference
  std::cout << "=== Extrinsics (for reference) ===" << std::endl;
  std::cout << "FRONT:" << std::endl;
  std::cout << "  lin_x: -0.0062785" << std::endl;
  std::cout << "  lin_y: 0.1156437" << std::endl;
  std::cout << "  lin_z: 0.1109724" << std::endl;
  std::cout << "  ang_roll: 0.0000000" << std::endl;
  std::cout << "  ang_pitch: 0.0000000" << std::endl;
  std::cout << "  ang_yaw: 1.5707963" << std::endl;
  std::cout << std::endl;

  std::cout << "FRONT LEFT:" << std::endl;
  std::cout << "  lin_x: 0.1151099" << std::endl;
  std::cout << "  lin_y: -0.0007408" << std::endl;
  std::cout << "  lin_z: 0.1109724" << std::endl;
  std::cout << "  ang_roll: 0.0000000" << std::endl;
  std::cout << "  ang_pitch: 0.0000000" << std::endl;
  std::cout << "  ang_yaw: 0.0000000" << std::endl;
  std::cout << std::endl;

  return 0;
}