#ifndef __IMPLICITFUNCTION_HPP__
#define __IMPLICITFUNCTION_HPP__

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Eigen>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>

#define X_MIN -3.0
#define X_MAX 3.0
#define Y_MIN -3.0
#define Y_MAX 3.0
#define Z_MIN 0.0
#define Z_MAX 0.0
#define X_STEP 0.02
#define Y_STEP 0.02
#define Z_STEP 0.02

bool isZero(float x) {
    return abs(x) < 0.1f;
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

void getZeroPoints(
    const std::vector<std::pair<Eigen::Vector3f, float>>& constraints,
    const Eigen::VectorXf& weights, float P0, const Eigen::Vector3f& P,
    std::vector<Eigen::Vector3f>& result)
{
#pragma omp parallel
    for (float x = X_MIN; x <= X_MAX; x += X_STEP) {
        for (float y = Y_MIN; y <= Y_MAX; y += Y_STEP) {
            for (float z = Z_MIN; z <= Z_MAX; z += Z_STEP) {
                if (isZero(implicitFunctionValue(Eigen::Vector3f(x, y, z), constraints, weights, P0, P))) {
                    result.push_back(Eigen::Vector3f(x, y, z));
                }
            }

        }
    }
}

#endif // __IMPLICITFUNCTION_HPP__