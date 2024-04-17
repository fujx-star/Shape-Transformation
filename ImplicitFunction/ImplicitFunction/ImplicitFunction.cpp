// ImplicitFunction.cpp: 定义应用程序的入口点。

#include "ImplicitFunction.h"
#include "LinearSystem.hpp"
#include "ConvexHull.hpp"
#include "Shader.h"
#include "Camera.h"
#include "setting.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Eigen>
#include <vector>
#define DIMENSION 3
using namespace std;

const int dimension = 2;


bool isZero(float x) {
    return abs(x) < 0.1f;
}

float RBF(Eigen::VectorXf x) {
    float length = x.norm();
    if (length == 0.0f) {
        return 0.0f;
    }
    
    return glm::pow(length, 2.0f) * glm::log(length);
}

float interpolationFunction(Eigen::VectorXf x,
    const vector<pair<Eigen::VectorXf, float>>& constraints,
    const Eigen::VectorXf& weights,
    float P0,
    const Eigen::VectorXf& P) {
    float res = 0.0f;
    for (int i = 0; i < constraints.size(); i++) {
        res += weights(i) * RBF(x - constraints[i].first);
    }
    res += P0;
    res += x.dot(P);
    return res;
}

void solve2D(
    float xmin, float xmax, float ymin, float ymax, float step,
    const vector<pair<Eigen::VectorXf, float>>& constraints,
    const Eigen::VectorXf& weights, float P0, const Eigen::VectorXf& P,
    std::vector<Eigen::VectorXf>& result)
{
    for (float x = xmin; x < xmax; x += step) {
        for (float y = ymin; y < ymax; y += step) {
            if (isZero(interpolationFunction(Eigen::Vector2f(x, y), constraints, weights, P0, P))) {
                result.push_back(Eigen::Vector2f(x, y));
            }
        }
    }
}


int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    //glEnable(GL_DEPTH_TEST);

    vector<pair<Eigen::VectorXf, float>> constraints = {
        // boundary constraints
        {Eigen::Vector2f(1.0f, 1.0f), 0.0f},
        {Eigen::Vector2f(-1.0f, 1.0f), 0.0f},
        {Eigen::Vector2f(-1.0f, -1.0f), 0.0f},
        {Eigen::Vector2f(1.0f, -1.0f), 0.0f},
        // noramal  constraints
        {Eigen::Vector2f(0.9f, 0.9f), 1.0f},
        {Eigen::Vector2f(-0.9f, 0.9f), 1.0f},
        {Eigen::Vector2f(-0.9f, -0.9f), 1.0f},
        {Eigen::Vector2f(0.9f, -0.9f), 1.0f}
    };

    int numConstraints = constraints.size();
    int n = numConstraints + 1 + dimension;

    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(n, n);
    Eigen::VectorXf B = Eigen::VectorXf::Zero(n);
    Eigen::VectorXf X = Eigen::VectorXf::Zero(n);

    for (int i = 0; i < numConstraints; i++) {
        for (int j = 0; j < numConstraints; j++) {
            A(i, j) = RBF(constraints[i].first - constraints[j].first);
        }
    }
    for (int i = 0; i < numConstraints; i++) {
        A(i, numConstraints) = 1.0f;
        for (int j = 0; j < dimension; j++) {
            A(i, numConstraints + j + 1) = constraints[i].first(j);
        }
    }
    for (int i = 0; i < numConstraints; i++) {
        A(numConstraints, i) = 1.0f;
        for (int j = 0; j < dimension; j++) {
            A(numConstraints + j + 1, i) = constraints[i].first(j);
        }
    }
    for (int i = 0; i < n; i++) {
        if (i < numConstraints) {
            B(i) = constraints[i].second;
        }
        else {
            B(i) = 0.0f;
        }
    }

    //cout << "A: " << endl << A << endl;
    //cout << "B: " << endl << B << endl;

    X = A.lu().solve(B);

    //cout << X << endl;

    Eigen::VectorXf weights = X.head(numConstraints);
    float P0 = X(numConstraints);
    Eigen::VectorXf P = X.tail(dimension);

    float xmin = -5.0f;
    float xmax = 5.0f;
    float ymin = -5.0f;
    float ymax = 5.0f;
    float step = 0.1f;

    std::vector<Eigen::VectorXf> points;
    solve2D(xmin, xmax, ymin, ymax, step, constraints, weights, P0, P, points);

    for (int i = 0; i < points.size(); i++) {
        for (int j = 0; j < dimension; j++) {
            std::cout << points[i][j] << " ";
        }
        std::cout << std::endl;
    }
    vector<float> glPoints;
    point2DConvertTriangle(points, glPoints);
    
    int pointSize = glPoints.size() / dimension  * DIMENSION * sizeof(float);
    float* pointVertices = (float*)malloc(pointSize);
    int index = 0;
    for (int i = 0; i < glPoints.size() / dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            pointVertices[index++] = glPoints[i * dimension + j];
        }
        for (int j = dimension; j < DIMENSION; j++) {
            pointVertices[index++] = 0.0f;
        }
    }
    //ConvexHull();
    int lineSize = points.size() * sizeof(float) * dimension;
    float* lineVertices = (float*)malloc(lineSize);
    for (int i = 0; i < points.size(); i++) {
        for (int j = 0; j < DIMENSION; j++) {
            if (j >= dimension) {
                lineVertices[i * dimension + j] = 0.0f;
            }
            else {
                lineVertices[i * dimension + j] = points[i][j];
            }
        }
    }

    //int lineSize = 4 * sizeof(float);
    //float* lineVertices = (float*)malloc(lineSize);
    //lineVertices[0] = -1.0f;
    //lineVertices[1] = -1.0f;
    //lineVertices[2] = 1.0f;
    //lineVertices[3] = 1.0f;

    Shader pointShader("E:/Coding/ShapeTransformation/ImplicitFunction/ImplicitFunction/Point.vert", "E:/Coding/ShapeTransformation/ImplicitFunction/ImplicitFunction/Point.frag");
    Shader lineShader("E:/Coding/ShapeTransformation/ImplicitFunction/ImplicitFunction/Line.vert", "E:/Coding/ShapeTransformation/ImplicitFunction/ImplicitFunction/Line.frag");

    unsigned int VBOs[2], VAOs[2];
    glGenVertexArrays(2, VAOs);
    glGenBuffers(2, VBOs);

    glBindVertexArray(VAOs[0]);
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[0]);
    glBufferData(GL_ARRAY_BUFFER, pointSize, pointVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, DIMENSION, GL_FLOAT, GL_FALSE, DIMENSION * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(VAOs[1]);
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[1]);
    glBufferData(GL_ARRAY_BUFFER, lineSize, lineVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, DIMENSION, GL_FLOAT, GL_FALSE, DIMENSION * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    while (!glfwWindowShouldClose(window))
    {
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        processInput(window);

        pointShader.use();
        glBindVertexArray(VAOs[0]);
        glDrawArrays(GL_TRIANGLES, 0, glPoints.size() / dimension);

        lineShader.use();
        glBindVertexArray(VAOs[1]);
        glDrawArrays(GL_LINE_LOOP, 0, points.size());

        glfwSwapBuffers(window);
        glfwPollEvents();

    }

    glDeleteVertexArrays(2, VAOs);
    glDeleteBuffers(2, VBOs);
    pointShader.Delete();
    lineShader.Delete();

    glfwTerminate();
    return 0;
}