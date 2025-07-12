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

#include <Eigen/Geometry>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct KannalaBrandtIntrinsics {
    double fx, fy, cx, cy, k1, k2, k3, k4;
};

struct OdenViewParameters {
    int id;
    double k[5];
    Eigen::Vector2d offset;
    Eigen::Vector3d position;
    Eigen::Quaterniond rotation;
    double max_theta;
};

struct GroundTruthPoint {
    Eigen::Vector3d point_3d;
    Eigen::Vector2d point_2d;
};

struct OdenReprojectionError {
    const Eigen::Vector3d point_3d;
    const Eigen::Vector2d target_pixel;
    const int image_width;
    const int image_height;

    OdenReprojectionError(const GroundTruthPoint & gt, int w, int h)
        : point_3d(gt.point_3d), target_pixel(gt.point_2d), image_width(w), image_height(h) {}

    static ceres::CostFunction * Create(const GroundTruthPoint & gt, int w, int h) {
        return new ceres::AutoDiffCostFunction<OdenReprojectionError, 2, 5, 2>(
            new OdenReprojectionError(gt, w, h));
    }

    template <typename T>
    bool operator()(const T * const k_coeffs, const T * const offset, T * residuals) const {
        T p_norm = T(point_3d.norm());
        if (p_norm < T(1e-3)) {
            p_norm = T(1e-3);
        }

        const T theta_oden = acos(T(-point_3d.z()) / p_norm);

        const T t2 = theta_oden * theta_oden;
        const T t3 = t2 * theta_oden;
        const T t4 = t3 * theta_oden;
        const T t5 = t4 * theta_oden;
        const T r_oden = k_coeffs[0] * theta_oden + k_coeffs[1] * t2 + k_coeffs[2] * t3 + k_coeffs[3] * t4 +
                         k_coeffs[4] * t5;

        T u_oden_norm, v_oden_norm;
        T image_plane_norm = sqrt(T(point_3d.x() * point_3d.x()) + T(point_3d.y() * point_3d.y()));

        if (image_plane_norm < T(1e-9)) {
            u_oden_norm = offset[0];
            v_oden_norm = offset[1];
        } else {
            const T image_unit_x = T(point_3d.x()) / image_plane_norm;
            const T image_unit_y = T(point_3d.y()) / image_plane_norm;
            u_oden_norm = r_oden * image_unit_x + offset[0];
            v_oden_norm = r_oden * image_unit_y + offset[1];
        }

        const T aspect = T(image_width) / T(image_height);
        const T u_predicted_pixel = (((u_oden_norm / aspect) + T(1.0)) / T(2.0)) * T(image_width);
        const T v_predicted_pixel = (T(1.0) - v_oden_norm) * T(image_height) / T(2.0);

        residuals[0] = T(target_pixel.x()) - u_predicted_pixel;
        residuals[1] = T(target_pixel.y()) - v_predicted_pixel;

        return true;
    }
};

static double getMaxTheta(const double k_coeffs[5]) {
    double max_theta = 0.0;
    const double step_size = 0.01;
    double prev_radius = 0.0;
    while (max_theta < M_PI) {
        max_theta += step_size;
        const double t2 = max_theta * max_theta;
        const double t3 = t2 * max_theta;
        const double t4 = t3 * max_theta;
        const double t5 = t4 * max_theta;
        double current_radius = k_coeffs[0] * max_theta + k_coeffs[1] * t2 + k_coeffs[2] * t3 +
                                k_coeffs[3] * t4 + k_coeffs[4] * t5;
        if (current_radius < prev_radius && max_theta > 0.1) {
            break;
        }
        prev_radius = current_radius;
    }
    return max_theta;
}

static OdenViewParameters convertToOden(const nlohmann::json & camera_json) {
    OdenViewParameters oden_params;
    oden_params.id = camera_json["id"];

    const int width = camera_json["image_width"];
    const int height = camera_json["image_height"];

    const auto intrinsics_json = camera_json["intrinsics"];
    const KannalaBrandtIntrinsics kb = { intrinsics_json["fx"], intrinsics_json["fy"], intrinsics_json["cx"],
                                         intrinsics_json["cy"], intrinsics_json["k1"], intrinsics_json["k2"],
                                         intrinsics_json["k3"], intrinsics_json["k4"] };

    std::vector<GroundTruthPoint> ground_truth_points;

    for (double alpha = 0.0; alpha < 2.0 * M_PI; alpha += M_PI / 16.0) {
        for (double theta = 0.01; theta < M_PI; theta += 0.05) {
            Eigen::Vector3d point_3d_kb;
            {
                point_3d_kb.x() = sin(theta) * cos(alpha);
                point_3d_kb.y() = sin(theta) * sin(alpha);
                point_3d_kb.z() = cos(theta);
            }

            const double theta3 = pow(theta, 3);
            const double theta5 = pow(theta, 5);
            const double theta7 = pow(theta, 7);
            const double theta9 = pow(theta, 9);

            const double r_kb = theta + kb.k1 * theta3 + kb.k2 * theta5 + kb.k3 * theta7 + kb.k4 * theta9;

            const double u_pixel = kb.fx * r_kb * cos(alpha) + kb.cx;
            const double v_pixel = kb.fy * r_kb * sin(alpha) + kb.cy;

            Eigen::Vector3d point_3d_oden(point_3d_kb.x(), -point_3d_kb.y(), -point_3d_kb.z());

            if (u_pixel >= 0 && u_pixel < width && v_pixel >= 0 && v_pixel < height) {
                ground_truth_points.push_back({ point_3d_oden, { u_pixel, v_pixel } });
            } else {
                break;
            }
        }
    }

    oden_params.k[0] = 0.3;
    oden_params.k[1] = oden_params.k[2] = oden_params.k[3] = oden_params.k[4] = 0.0;
    oden_params.offset = { 0.0, 0.0 };

    ceres::Problem problem;
    for (const auto & gt_point : ground_truth_points) {
        ceres::CostFunction * cost = OdenReprojectionError::Create(gt_point, width, height);
        problem.AddResidualBlock(cost, new ceres::CauchyLoss(0.5), oden_params.k, oden_params.offset.data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 200;
    options.logging_type = ceres::SILENT;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    std::cerr << "Camera \"" << std::string(camera_json["name"])
              << "\": RMS: " << sqrt(summary.final_cost / (problem.NumResiduals() / 2.0))
              << " px, Iters: " << summary.iterations.size() << std::endl;

    const auto extrinsics_json = camera_json["extrinsics"];

    Eigen::Vector3d customer_pos = {
        extrinsics_json["position"][0],
        extrinsics_json["position"][1],
        extrinsics_json["position"][2],
    };
    oden_params.position.x() = customer_pos.x();
    oden_params.position.y() = -customer_pos.y();
    oden_params.position.z() = -customer_pos.z();

    const double roll_rad = extrinsics_json["rotation_rpy_rad"][0].get<double>();
    const double pitch_rad = extrinsics_json["rotation_rpy_rad"][1].get<double>();
    const double yaw_rad = extrinsics_json["rotation_rpy_rad"][2].get<double>();

    oden_params.rotation = Eigen::AngleAxisd(yaw_rad, -Eigen::Vector3d::UnitY()) *
                           Eigen::AngleAxisd(pitch_rad, Eigen::Vector3d::UnitX()) *
                           Eigen::AngleAxisd(roll_rad, -Eigen::Vector3d::UnitZ());

    oden_params.rotation.normalize();

    oden_params.max_theta = getMaxTheta(oden_params.k);
    return oden_params;
}

static void printOdenView(const OdenViewParameters & oden) {
    std::cout << "        {\n";
    std::cout << "            no = " << oden.id + 1 << ";\n";
    std::cout << std::fixed << std::setprecision(16);
    std::cout << "            k = [ " << oden.k[0] << ", " << oden.k[1] << ", " << oden.k[2] << ", "
              << oden.k[3] << ", " << oden.k[4] << " ];\n";
    std::cout << "            offset = [ " << oden.offset.x() << ", " << oden.offset.y() << " ];\n";
    std::cout << "            position = [ " << oden.position.x() << ", " << oden.position.y() << ", "
              << oden.position.z() << " ];\n";
    std::cout << "            rotation = [ " << oden.rotation.w() << ", " << oden.rotation.x() << ", "
              << oden.rotation.y() << ", " << oden.rotation.z() << " ];\n";
    std::cout << "            max_theta = " << oden.max_theta << ";\n";
    std::cout << "        }";
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "Error: Invalid arguments." << std::endl;
        std::cerr << "Usage: " << argv[0] << " /path/to/input_params.json" << std::endl;
        return 1;
    }
    std::ifstream input_file(argv[1]);
    if (!input_file.is_open()) {
        std::cerr << "Error: Could not open file " << argv[1] << std::endl;
        return 1;
    }
    nlohmann::json data;
    try {
        data = nlohmann::json::parse(input_file);
    } catch (nlohmann::json::parse_error & e) {
        std::cerr << "Error: Failed to parse JSON file: " << e.what() << std::endl;
        return 1;
    }

    std::vector<OdenViewParameters> processed_params;
    const auto cameras = data["cameras"];

    for (const auto & camera_json : cameras) {
        OdenViewParameters oden_params = convertToOden(camera_json);
        processed_params.push_back(oden_params);
    }

    std::cout << "views = (";
    for (size_t i = 0; i < processed_params.size(); ++i) {
        std::cout << "\n";
        printOdenView(processed_params[i]);
        if (i < processed_params.size() - 1) {
            std::cout << ",";
        }
    }
    std::cout << "\n);\n";

    return 0;
}
