// ImplicitFunction.cpp: 定义应用程序的入口点。

#include "algorithm/ImplicitFunction.hpp"
#include "algorithm/ConvexHull.hpp"
#include "algorithm/ImageProcess.hpp"
#include "settings/Shader.h"
#include "settings/Camera.h"
#include "settings/setting.hpp"
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
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <vector>
#define DIMENSION 3
#define MAX_MATRIX_DIMENSION 200
#define DEBUGx

typedef struct Color {
    int b;
    int g;
    int r;
} Color;

void generateContraints(
    const char* imagePath,
    std::vector<pair<Eigen::Vector3f, float>>& constraints,
    int& rows, int& cols) 
{
    std::vector<Eigen::Vector2f> boundaryPoints, normalPoints;
    processImage3(imagePath, rows, cols, boundaryPoints, normalPoints);
    for (const auto& point : boundaryPoints) {
        constraints.emplace_back(Eigen::Vector3f(point.x(), point.y(), 0.0f), 0.0f);
    }
    for (const auto& point : normalPoints) {
        constraints.emplace_back(Eigen::Vector3f(point.x(), point.y(), 0.0f), 1.0f);
    }
}

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

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

    //vector<pair<Eigen::Vector3f, float>> constraints = {
    //    // boundary constraints
    //    {Eigen::Vector3f(1.0f, 1.0f, 0.0f), 0.0f},
    //    {Eigen::Vector3f(-1.0f, 1.0f, 0.0f), 0.0f},
    //    {Eigen::Vector3f(-1.0f, -1.0f, 0.0f), 0.0f},
    //    {Eigen::Vector3f(1.0f, -1.0f, 0.0f), 0.0f},
    //    // noramal  constraints
    //    {Eigen::Vector3f(0.9f, 0.9f, 0.0f), 1.0f},
    //    {Eigen::Vector3f(-0.9f, 0.9f, 0.0f), 1.0f},
    //    {Eigen::Vector3f(-0.9f, -0.9f, 0.0f), 1.0f},
    //    {Eigen::Vector3f(0.9f, -0.9f, 0.0f), 1.0f}
    //};
    std::vector<std::pair<Eigen::Vector3f, float>> constraints;
    int rows, cols;
    generateContraints("C:/Users/Administrator/Desktop/无标题.png", constraints, rows, cols);

    int numConstraints = constraints.size();
    int n = numConstraints + DIMENSION + 1;
    if (n > MAX_MATRIX_DIMENSION) {
        std::cout << "Too many constraints!" << std::endl;
        return -1;
    }

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

    X = A.lu().solve(B);

#ifdef DEBUG
    cout << "A: " << endl << A << endl;
    cout << "B: " << endl << B << endl;
    cout << X << endl;
#endif // DEBUG

    Eigen::VectorXf weights = X.head(numConstraints);
    float P0 = X(numConstraints);
    Eigen::Vector3f P = X.tail(DIMENSION);

    //float xmin = -3.0f;
    //float xmax = 3.0f;
    //float ymin = -3.0f;
    //float ymax = 3.0f;
    //float zmin = 0.0f;
    //float zmax = 0.0f;
    //float step = 0.02f;
    checkConstraints(constraints, weights, P0, P);

    std::vector<Eigen::Vector3f> points;
    getZeroValuePoints(rows, cols, constraints, weights, P0, P, points);

#ifdef IMAGE_DEBUG
    cv::Mat generatePointImage = cv::Mat::zeros(rows, cols, CV_8UC3);
    for (int i = 0; i < points.size(); i++) {
        cv::Point2f point(points[i][0], points[i][1]);
        cv::circle(generatePointImage, point, 0.5, cv::Scalar(255, 0, 0), 4);
    }
    cv::imshow("generate_point_image", generatePointImage);
    cv::waitKey(0);
#endif // IMAGE_DEBUG

    std::vector<Eigen::Vector3f> linePoints;
    ConvexHull(points, linePoints);

#ifdef IMAGE_DEBUG
    cv::Mat generateContourImage = cv::Mat::zeros(rows, cols, CV_8UC3);
    for (int i = 0; i < linePoints.size(); i++) {
        cv::Point2f startPoint(linePoints[i][0], linePoints[i][1]), endPoint;
        if (i == linePoints.size() - 1) {
            endPoint = { linePoints[0][0], linePoints[0][1] };
        }
        else {
            endPoint = { linePoints[i + 1][0], linePoints[i + 1][1] };
        }
        cv::line(generateContourImage, startPoint, endPoint, cv::Scalar(255, 0, 0), 2);
    }
    cv::imshow("generate_contour_image", generateContourImage);
    cv::waitKey(0);
#endif // IMAGE_DEBUG

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

#ifdef DEBUG
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
#endif // DEBUG

    Shader pointShader("../../../../ImplicitFunction/resources/Point.vert", "../../../../ImplicitFunction/resources/Point.frag");
    Shader lineShader("../../../../ImplicitFunction/resources/Line.vert", "../../../../ImplicitFunction/resources/Line.frag");

    unsigned int VBOs[2], VAOs[2];
    glGenVertexArrays(2, VAOs);
    glGenBuffers(2, VBOs);

    glBindVertexArray(VAOs[0]);
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[0]);
    glBufferData(GL_ARRAY_BUFFER, actualPointSize, actualPointVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, DIMENSION, GL_FLOAT, GL_FALSE, DIMENSION * sizeof(float), (void*)0);
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