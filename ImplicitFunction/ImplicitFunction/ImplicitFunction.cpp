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
#define DEBUG

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

float interpolationFunction(Eigen::Vector3f x,
    const vector<pair<Eigen::Vector3f, float>>& constraints,
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

void solve(
    float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, float step,
    const vector<pair<Eigen::Vector3f, float>>& constraints,
    const Eigen::VectorXf& weights, float P0, const Eigen::Vector3f& P,
    std::vector<Eigen::Vector3f>& result)
{
#pragma omp parallel
    for (float x = xmin; x <= xmax; x += step) {
        for (float y = ymin; y <= ymax; y += step) {
            for (float z = zmin; z <= zmax; z += step) {
                if (isZero(interpolationFunction(Eigen::Vector3f(x, y, z), constraints, weights, P0, P))) {
                    result.push_back(Eigen::Vector3f(x, y, z));
                }
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

    glEnable(GL_DEPTH_TEST);

    vector<pair<Eigen::Vector3f, float>> constraints = {
        // boundary constraints
        {Eigen::Vector3f(1.0f, 1.0f, 0.0f), 0.0f},
        {Eigen::Vector3f(-1.0f, 1.0f, 0.0f), 0.0f},
        {Eigen::Vector3f(-1.0f, -1.0f, 0.0f), 0.0f},
        {Eigen::Vector3f(1.0f, -1.0f, 0.0f), 0.0f},
        // noramal  constraints
        {Eigen::Vector3f(0.9f, 0.9f, 0.0f), 1.0f},
        {Eigen::Vector3f(-0.9f, 0.9f, 0.0f), 1.0f},
        {Eigen::Vector3f(-0.9f, -0.9f, 0.0f), 1.0f},
        {Eigen::Vector3f(0.9f, -0.9f, 0.0f), 1.0f}
    };

    int numConstraints = constraints.size();
    int n = numConstraints + DIMENSION + 1;

    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(n, n);
    Eigen::VectorXf B = Eigen::VectorXf::Zero(n);
    Eigen::VectorXf X = Eigen::VectorXf::Zero(n);

#ifdef DEBUG
    for (int i = 0; i < numConstraints; i++) {
        for (int j = 0; j < numConstraints; j++) {
            A(i, j) = RBF(constraints[i].first - constraints[j].first);
        }
}
    for (int i = 0; i < numConstraints; i++) {
        A(i, numConstraints) = 1.0f;
        for (int j = 0; j < DIMENSION; j++) {
            A(i, numConstraints + j + 1) = constraints[i].first(j);
        }
    }
    for (int i = 0; i < numConstraints; i++) {
        A(numConstraints, i) = 1.0f;
        for (int j = 0; j < DIMENSION; j++) {
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
#endif // DEBUG

    X = A.lu().solve(B);

#ifdef DEBUG
    cout << "A: " << endl << A << endl;
    cout << "B: " << endl << B << endl;
    cout << X << endl;
#endif // DEBUG

    Eigen::VectorXf weights = X.head(numConstraints);
    float P0 = X(numConstraints);
    Eigen::Vector3f P = X.tail(DIMENSION);

    float xmin = -5.0f;
    float xmax = 5.0f;
    float ymin = -5.0f;
    float ymax = 5.0f;
    float zmin = 0.0f;
    float zmax = 0.0f;
    float step = 0.02f;

    std::vector<Eigen::Vector3f> points;
    solve(xmin, xmax, ymin, ymax, zmin, zmax, step, constraints, weights, P0, P, points);

    std::vector<Eigen::Vector3f> linePoints;
    ConvexHull(points, linePoints);
    int linePointSize = linePoints.size() * DIMENSION * sizeof(float);
    float* linePointVertices = (float*)malloc(linePointSize);
    int index = 0;
    for (int i = 0; i < linePoints.size(); i++) {
        for (int j = 0; j < DIMENSION; j++) {
            linePointVertices[index++] = linePoints[i][j];
        }
    }

    vector<Eigen::Vector3f> actualPoints;
    pointConvertTriangle(linePoints, actualPoints);
    int actualPointSize = actualPoints.size() * DIMENSION * sizeof(float);
    float* actualPointVertices = (float*)malloc(actualPointSize);
    index = 0;
    for (int i = 0; i < actualPoints.size(); i++) {
        for (int j = 0; j < DIMENSION; j++) {
            actualPointVertices[index++] = actualPoints[i][j];
        }
    }

    // 验证数据
    std::cout << "---------------points---------------" << points.size() << std::endl;
    for (int i = 0; i < points.size(); i++) {
        for (int j = 0; j < DIMENSION; j++) {
            std::cout << points[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "---------------line points---------------" << linePoints.size() << std::endl;
    for (int i = 0; i < linePoints.size(); i++) {
        for (int j = 0; j < DIMENSION; j++) {
            std::cout << linePointVertices[i * DIMENSION + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "---------------actual points---------------" << actualPoints.size() << std::endl;
    for (int i = 0; i < actualPoints.size(); i++) {
        for (int j = 0; j < DIMENSION; j++) {
            std::cout << actualPointVertices[i * DIMENSION + j] << " ";
        }
        std::cout << std::endl;
    }
    

    float vertices[] = {
    -0.5f, -0.5f, -0.5f,
     0.5f, -0.5f, -0.5f,
     0.5f,  0.5f, -0.5f,
     0.5f,  0.5f, -0.5f,
    -0.5f,  0.5f, -0.5f,
    -0.5f, -0.5f, -0.5f,
    -0.5f, -0.5f,  0.5f,
     0.5f, -0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,
    -0.5f,  0.5f,  0.5f,
    -0.5f, -0.5f,  0.5f,
    -0.5f,  0.5f,  0.5f,
    -0.5f,  0.5f, -0.5f,
    -0.5f, -0.5f, -0.5f,
    -0.5f, -0.5f, -0.5f,
    -0.5f, -0.5f,  0.5f,
    -0.5f,  0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,
     0.5f,  0.5f, -0.5f,
     0.5f, -0.5f, -0.5f,
     0.5f, -0.5f, -0.5f,
     0.5f, -0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,
    -0.5f, -0.5f, -0.5f,
     0.5f, -0.5f, -0.5f,
     0.5f, -0.5f,  0.5f,
     0.5f, -0.5f,  0.5f,
    -0.5f, -0.5f,  0.5f,
    -0.5f, -0.5f, -0.5f,
    -0.5f,  0.5f, -0.5f,
     0.5f,  0.5f, -0.5f,
     0.5f,  0.5f,  0.5f,
     0.5f,  0.5f,  0.5f,
    -0.5f,  0.5f,  0.5f,
    -0.5f,  0.5f, -0.5f
    };

    Shader pointShader("E:/Coding/ShapeTransformation/ImplicitFunction/ImplicitFunction/Point.vert", "E:/Coding/ShapeTransformation/ImplicitFunction/ImplicitFunction/Point.frag");
    Shader lineShader("E:/Coding/ShapeTransformation/ImplicitFunction/ImplicitFunction/Line.vert", "E:/Coding/ShapeTransformation/ImplicitFunction/ImplicitFunction/Line.frag");

    unsigned int VBOs[2], VAOs[2];
    glGenVertexArrays(2, VAOs);
    glGenBuffers(2, VBOs);

    glBindVertexArray(VAOs[0]);
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[0]);
    glBufferData(GL_ARRAY_BUFFER, actualPointSize, actualPointVertices, GL_STATIC_DRAW);
    //glBufferData(GL_ARRAY_BUFFER, 108 * sizeof(float), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, DIMENSION, GL_FLOAT, GL_FALSE, DIMENSION * sizeof(float), (void*)0);
    //glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, DIMENSION * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(VAOs[1]);
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[1]);
    glBufferData(GL_ARRAY_BUFFER, linePointSize, linePointVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, DIMENSION, GL_FLOAT, GL_FALSE, DIMENSION * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        processInput(window);

        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = glm::mat4(1.0f);
        glm::mat4 projection = glm::mat4(1.0f);
        view = camera.GetViewMatrix();
        projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);

        pointShader.use();
        pointShader.setMat4f("model", model);
        pointShader.setMat4f("view", view);
        pointShader.setMat4f("projection", projection);

        glBindVertexArray(VAOs[0]);
        glDrawArrays(GL_TRIANGLES, 0, actualPoints.size());

        lineShader.use();
        lineShader.setMat4f("model", model);
        lineShader.setMat4f("view", view);
        lineShader.setMat4f("projection", projection);
        glBindVertexArray(VAOs[1]);
        glDrawArrays(GL_LINE_LOOP, 0, linePoints.size());

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