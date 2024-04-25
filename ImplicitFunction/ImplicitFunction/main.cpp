#define WRITE_MODEx
#define DATA_DEBUGx
#define IMAGE_DEBUG
#define DIMENSION 3
#define MAX_MATRIX_DIMENSION 200
#define OPENGL_SCALE 100.0f

#include "algorithm/ImplicitFunction.hpp"
#include "algorithm/PointProcess.hpp"
#include "algorithm/ImageProcess.hpp"
#include "settings/Shader.h"
#include "settings/Camera.h"
#include "settings/setting.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

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

bool implicitFunctionInterpolation(
    float weight,
    int& rows,
    int& cols,
    const char* imagePath_1, 
    const char* imagePath_2,
    std::vector<Eigen::Vector3f>& points)
{
#ifdef WRITE_MODE
    std::vector<std::pair<Eigen::Vector3f, float>> constraints_1, constraints_2;
    int rows_1, cols_1;
    int rows_2, cols_2;
    // 根据输入图片得到边界约束和法向约束
    generateContraints(imagePath_1, constraints_1, rows_1, cols_1);
    generateContraints(imagePath_2, constraints_2, rows_2, cols_2);
    // 假设两张图片的大小是一样的
    rows = rows_1;
    cols = cols_1;

    Eigen::VectorXf weights_1, weights_2;
    float P0_1, P0_2;
    Eigen::Vector3f P_1, P_2;
    // 解线性方程组得到隐函数参数
    if (!solveImplicitEquation(constraints_1, weights_1, P0_1, P_1)) {
        return false;
    }
    if (!solveImplicitEquation(constraints_2, weights_2, P0_2, P_2)) {
        return false;
    }

    ofstream file1("image1_value.txt"), file2("image2_value.txt");
    if (!file1 || !file2) {
        return false;
    }
    file1 << rows_1 << std::endl << cols_1 << std::endl << STEP << std::endl;
    file2 << rows_2 << std::endl << cols_2 << std::endl << STEP << std::endl;
    float value;
    for (float x = 0.0f; x <= static_cast<float>(cols); x += STEP) {
        for (float y = 0.0f; y <= static_cast<float>(rows); y += STEP) {
            for (float z = 0.0f; z <= 0.0f; z += STEP) {
                value = implicitFunctionValue(Eigen::Vector3f(x, y, z), constraints_1, weights_1, P0_1, P_1);
                file1 << std::setprecision(std::numeric_limits<float>::max_digits10) << value << std::endl;
                value = implicitFunctionValue(Eigen::Vector3f(x, y, z), constraints_2, weights_2, P0_2, P_2);
                file2 << std::setprecision(std::numeric_limits<float>::max_digits10) << value << std::endl;
            }
        }
        std::cout << "Finish writing cols[" << x << "]" << std::endl;
    }
    file1.close();
    file2.close();
    std::cout << "Suceessfully write image1_value.txt and image2_value.txt" << std::endl;
#else
    float weight_1 = weight;
    float weight_2 = 1.0f - weight;
    ifstream file1("image1_value.txt"), file2("image2_value.txt");
    if (!file1 || !file2) {
        return false;
    }
    std::string line1, line2;

    // 读取图片大小和步长
    for (int i = 0; i < 3; i++) {
        std::getline(file1, line1);
        std::getline(file2, line2);
        if (i == 0) {
            rows = std::stoi(line1);
        }
        else if (i == 1) {
            cols = std::stoi(line1);
        }
    }

    // 读取图片的隐函数值
#pragma omp parallel
    for (float x = 0.0f; x <= static_cast<float>(cols); x += STEP) {
        for (float y = 0.0f; y <= static_cast<float>(rows); y += STEP) {
            for (float z = 0.0f; z <= 0.0f; z += STEP) {
                std::getline(file1, line1);
                std::getline(file2, line2);
                float value1 = std::stof(line1);
                float value2 = std::stof(line2);
                if (isZero(weight_1 * value1 + weight_2 * value2)) {
                    points.push_back(Eigen::Vector3f(x, y, z));
                }
            }
        }
    }
#endif
}

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    std::vector<Eigen::Vector3f> points;

    // 单个图片形状边界的隐函数零值点
    //getZeroValuePoints(rows, cols, constraints, weights, P0, P, points);

    // 两张图片形状边界隐函数线性插值的零值点
    float weight = 0.1f;
    const char* imagePath_1 = "C:/Users/Administrator/Desktop/无标题.png";
    const char* imagePath_2 = "C:/Users/Administrator/Desktop/有标题.png";
    int rows, cols;
    if (!implicitFunctionInterpolation(weight, rows, cols, imagePath_1, imagePath_2, points)) {
        std::cerr << "Implicit function interpolation failed." << std::endl;
        return -1;
    }

#ifndef WRITE_MODE
#ifdef IMAGE_DEBUG
    cv::Mat generatePointImage = cv::Mat::zeros(rows, cols, CV_8UC3);
    for (int i = 0; i < points.size(); i++) {
        cv::Point2f point(points[i][0], points[i][1]);
        cv::circle(generatePointImage, point, 0.5, cv::Scalar(255, 0, 0), 4);
    }
    cv::imshow("generate_point_image", generatePointImage);
    cv::waitKey(0);
#endif // IMAGE_DEBUG

    std::vector<int> pointIndexes;
    std::vector<std::pair<int, int>> edgeIndexes;
    ConcaveHull(points, pointIndexes, edgeIndexes);

#ifdef IMAGE_DEBUG
    cv::Mat hullPointImage = cv::Mat::zeros(rows, cols, CV_8UC3);
    for (int i = 0; i < pointIndexes.size(); i++) {
        cv::Point2f point(points[pointIndexes[i]].x(), points[pointIndexes[i]].y());
        cv::circle(hullPointImage, point, 0.5, cv::Scalar(255, 0, 0), 4);
    }
    cv::imshow("hull_point_image", hullPointImage);
    cv::waitKey(0);
    cv::Mat hullContourImage = cv::Mat::zeros(rows, cols, CV_8UC3);
    for (int i = 0; i < edgeIndexes.size(); i++) {
        cv::Point2f startPoint(points[edgeIndexes[i].first].x(), points[edgeIndexes[i].first].y());
        cv::Point2f endPoint(points[edgeIndexes[i].second].x(), points[edgeIndexes[i].second].y());
        cv::line(hullContourImage, startPoint, endPoint, cv::Scalar(255, 0, 0), 2);
    }
    cv::imshow("hull_contour_image", hullContourImage);
    cv::waitKey(0);
#endif // IMAGE_DEBUG

    // glfw初始化
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    glEnable(GL_DEPTH_TEST);

    // imgui初始化
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    ImGui_ImplGlfw_InitForOpenGL(window, true);          // Second param install_callback=true will install GLFW callbacks and chain to existing ones.
    ImGui_ImplOpenGL3_Init();

    // 得到OpenGL显示数据，使图像偏移到OpenGL窗口的中心位置，方便OpenGL显示
    double offset[DIMENSION] = { -static_cast<float>(cols) / 2, -static_cast<float>(rows) / 2, 0.0f };
    int index = 0;
    vector<Eigen::Vector3f> actualPoints;
    pointConvertTriangle(points, pointIndexes, actualPoints);
    int actualPointSize = actualPoints.size() * DIMENSION * sizeof(float);
    float* actualPointVertices = (float*)malloc(actualPointSize);
    index = 0;
    for (int i = 0; i < actualPoints.size(); i++) {
        for (int j = 0; j < DIMENSION; j++) {
            actualPointVertices[index] = (actualPoints[i][j] + offset[j]) / OPENGL_SCALE;
            // 因为OPENGL坐标系和图片坐标系的Y轴方向相反，所以需要翻转Y轴
            if (j == 1) {
                actualPointVertices[index] = -actualPointVertices[index];
            }
            index++;
        }
    }
    index = 0;
    int edgePointSize = edgeIndexes.size() * 2 * DIMENSION * sizeof(float);
    float* edgePointVertices = (float*)malloc(edgePointSize);
    for (int i = 0; i < edgeIndexes.size(); i++) {
        Eigen::Vector3f startPoint = points[edgeIndexes[i].first];
        Eigen::Vector3f endPoint = points[edgeIndexes[i].second];
        for (int j = 0; j < DIMENSION; j++) {
            edgePointVertices[index] = (startPoint[j] + offset[j]) / OPENGL_SCALE;
            if (j == 1) {
                edgePointVertices[index] = -edgePointVertices[index];
            }
            index++;
        }
        for (int j = 0; j < DIMENSION; j++) {
            edgePointVertices[index] = (endPoint[j] + offset[j]) / OPENGL_SCALE;
            if (j == 1) {
                edgePointVertices[index] = -edgePointVertices[index];
            }
            index++;
        }
    }

#ifdef DATA_DEBUG
    // 验证数据
    std::cout << "---------------points---------------" << points.size() << std::endl;
    for (int i = 0; i < points.size(); i++) {
        for (int j = 0; j < DIMENSION; j++) {
            std::cout << points[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "---------------edge points---------------" << linePoints.size() << std::endl;
    for (int i = 0; i < edgeIndexes.size() * 2; i++) {
        for (int j = 0; j < DIMENSION; j++) {
            std::cout << linePointVertices[i * DIMENSION + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "---------------actual points---------------" << actualPoints.size() << std::endl;
    for (int i = 0; i < pointIndexes.size(); i++) {
        for (int j = 0; j < DIMENSION; j++) {
            std::cout << actualPointVertices[i * DIMENSION + j] << " ";
        }
        std::cout << std::endl;
    }
#endif // DATA_DEBUG

    Shader pointShader("../../../../ImplicitFunction/resources/Point.vert", "../../../../ImplicitFunction/resources/Point.frag");
    Shader edgeShader("../../../../ImplicitFunction/resources/Edge.vert", "../../../../ImplicitFunction/resources/Edge.frag");

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
    glBufferData(GL_ARRAY_BUFFER, edgePointSize, edgePointVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, DIMENSION, GL_FLOAT, GL_FALSE, DIMENSION * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::Begin("window!");
        ImGui::Text("helloworld.");
        ImGui::End();

        processInput(window);
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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

        edgeShader.use();
        edgeShader.setMat4f("model", model);
        edgeShader.setMat4f("view", view);
        edgeShader.setMat4f("projection", projection);
        glBindVertexArray(VAOs[1]);
        glDrawArrays(GL_LINES, 0, edgeIndexes.size() * 2);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glDeleteVertexArrays(2, VAOs);
    glDeleteBuffers(2, VBOs);
    pointShader.Delete();
    edgeShader.Delete();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
#endif // !WRITE_MODE  

    return 0;
}