#ifndef __IMPLICITFUNCTION_HPP__
#define __IMPLICITFUNCTION_HPP__

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Eigen>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <assert.h>
#include <vector>

#define STEP 5.0f
#define TOLERANCE 1.0f

bool isZero(float x) {
    return fabs(x) < TOLERANCE;
}

float RBF(Eigen::Vector3f x) {
    float length = x.norm();
    if (length == 0.0f) {
        return 0.0f;
    }
    return glm::pow(length, 2.0f) * glm::log(length);
}

float implicitFunctionValue(Eigen::Vector3f x,
    const std::vector<std::pair<Eigen::Vector3f, float>>& constraints,
    const Eigen::VectorXf& weights,
    float P0,
    const Eigen::Vector3f& P) {
    float res = 0.0f;
    for (int i = 0; i < constraints.size(); i++) {
        res += weights(i) * RBF(x - constraints[i].first);
    }
    res += P0;
    res += x.dot(P);
    return res;
}

void getZeroValuePoints(
    int rows, int cols,
    const std::vector<std::pair<Eigen::Vector3f, float>>& constraints,
    const Eigen::VectorXf& weights, float P0, const Eigen::Vector3f& P,
    std::vector<Eigen::Vector3f>& result)
{
#pragma omp parallel
    for (float x = 0.0f; x <= static_cast<float>(cols); x += STEP) {
        for (float y = 0.0f; y <= static_cast<float>(rows); y += STEP) {
            for (float z = 0.0f; z <= 0.0f; z += STEP) {
                if (isZero(implicitFunctionValue(Eigen::Vector3f(x, y, z), constraints, weights, P0, P))) {
                    result.push_back(Eigen::Vector3f(x, y, z));
                }
            }
        }
    }
}

void checkConstraints(
    const std::vector<std::pair<Eigen::Vector3f, float>>& constraints,
    const Eigen::VectorXf& weights,
    float P0,
    const Eigen::Vector3f& P) {
    float value;
    for (const auto& constraint : constraints) {
        if (constraint.second == 0.0f) {
            value = implicitFunctionValue(constraint.first, constraints, weights, P0, P);
            assert(fabs(value) < TOLERANCE);
        }
        else if (constraint.second == 1.0f) {
            value = implicitFunctionValue(constraint.first, constraints, weights, P0, P);
            assert(fabs(value - 1.0f) < TOLERANCE);
        }
    }
}

#endif // __IMPLICITFUNCTION_HPP__