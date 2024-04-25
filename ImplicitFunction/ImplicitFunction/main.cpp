#define WRITE_MODEx
#define EDGE_MODEx
#define DATA_DEBUGx
#define IMAGE_DEBUGx
#define DIMENSION 3
#define MAX_MATRIX_DIMENSION 200
#define MAX_ALLOC_SIZE 10485760
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

// 根据图片得到约束条件
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

// 将两张图片像素点的隐函数值写入文件
bool writeImageValue(
    int& rows,
    int& cols,
    const char* imagePath_1,
    const char* imagePath_2)
{
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
    for (int x = 0; x < cols; x += STEP) {
        for (int y = 0; y < rows; y += STEP) {
            for (int z = 0; z <= 0; z += STEP) {
                value = implicitFunctionValue(Eigen::Vector3f(x, y, z), constraints_1, weights_1, P0_1, P_1);
                file1 << std::setprecision(std::numeric_limits<float>::max_digits10) << value << std::endl;
                value = implicitFunctionValue(Eigen::Vector3f(x, y, z), constraints_2, weights_2, P0_2, P_2);
                file2 << std::setprecision(std::numeric_limits<float>::max_digits10) << value << std::endl;
            }
        }
        if (x % 10 == 0) {
            std::cout << "Finish writing cols[" << x << "]" << std::endl;
        }
    }
    file1.close();
    file2.close();
    std::cout << "Suceessfully write image1_value.txt and image2_value.txt" << std::endl;
    return true;
}

// 根据文件读取图片的隐函数值，并插值得到新的边界点
void implicitFunctionInterpolation(
    float weight,
    int rows,
    int cols,
    float* fileData_1,
    float* fileData_2,
    std::vector<Eigen::Vector3f>& points)
{
    float weight_1 = weight;
    float weight_2 = 1.0f - weight;

    // 读取图片的隐函数值
    int index = 0;
#pragma omp parallel
    for (int x = 0; x < cols; x += STEP) {
        for (int y = 0; y < rows; y += STEP) {
            for (int z = 0; z <= 0; z += STEP) {
                float value1 = fileData_1[index];
                float value2 = fileData_2[index];
                if (isZero(weight_1 * value1 + weight_2 * value2)) {
                    points.push_back(Eigen::Vector3f(x, y, z));
                }
                index++;
            }
        }
    }
}

// 点模式：OpenGL仅显示点
bool presentPoint(
    std::vector<Eigen::Vector3f> points,
    int& actualPointSize,
    float* actualPointVertices,
    float* offset)
{
    vector<Eigen::Vector3f> actualPoints;
    pointConvertTriangle(points, actualPoints);
    actualPointSize = actualPoints.size() * DIMENSION * sizeof(float);
    if (actualPointSize > MAX_ALLOC_SIZE) {
        std::cerr << "OpenGL data size is too large." << std::endl;
        return false;
    }
    int index = 0;
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
    return true;
}

// 点边模式：OpenGL显示α-shape算法处理后的点和边
bool presentPointAndEdge(
    std::vector<Eigen::Vector3f> points,
    int& actualPointSize,
    int& edgePointSize,
    float* actualPointVertices,
    float* edgePointVertices,
    float* offset)
{
    std::vector<int> pointIndexes;
    std::vector<std::pair<int, int>> edgeIndexes;
    ConcaveHull(points, pointIndexes, edgeIndexes);
    vector<Eigen::Vector3f> actualPoints;
    pointIndexConvertTriangle(points, pointIndexes, actualPoints);

    actualPointSize = actualPoints.size() * DIMENSION * sizeof(float);
    edgePointSize = edgeIndexes.size() * 2 * DIMENSION * sizeof(float);
    if (actualPointSize > MAX_ALLOC_SIZE || edgePointSize > MAX_ALLOC_SIZE) {
        std::cerr << "OpenGL data size is too large." << std::endl;
        return false;
    }
    int index = 0;
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
    return true;
}

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

#ifdef WRITE_MODE
    // 写入图片像素的隐函数值到文件
    const char* imagePath_1 = "C:/Users/Administrator/Desktop/无标题.png";
    const char* imagePath_2 = "C:/Users/Administrator/Desktop/有标题.png";
    int rows, cols;
    if (!writeImageValue(rows, cols, imagePath_1, imagePath_2)) {
        std::cerr << "Implicit function interpolation failed." << std::endl;
        return -1;
    }
#else
    float weight = 0.0f, preWeight = 0.0f;
    ifstream file1("image1_value.txt"), file2("image2_value.txt");
    if (!file1 || !file2) {
        return false;
    }
    std::string line1, line2;

    // 读取文件数据
    int rows, cols;
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
    int dataSize = ((rows / STEP + 1) * (cols / STEP + 1));    // 确保dataSize大小足够
    float* fileData_1 = new float[dataSize];
    float* fileData_2 = new float[dataSize];
    int index = 0;
    for (int x = 0; x < cols; x += STEP) {
        for (int y = 0; y < rows; y += STEP) {
            for (int z = 0; z <= 0; z += STEP) {
                std::getline(file1, line1);
                std::getline(file2, line2);
                fileData_1[index] = std::stof(line1);
                fileData_2[index] = std::stof(line2);
                index++;
            }
        }
    }

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

    // 得到OpenGL显示数据，并且使图像偏移到OpenGL窗口的中心位置，方便OpenGL显示
    float offset[DIMENSION] = { -static_cast<float>(cols) / 2, -static_cast<float>(rows) / 2, 0.0f };
    vector<Eigen::Vector3f> actualPoints;
    int actualPointSize = 0, edgePointSize = 0;
    float* actualPointVertices = new float[MAX_ALLOC_SIZE];
    float* edgePointVertices = new float[MAX_ALLOC_SIZE];

    // OpenGL对象定义
    Shader pointShader("../../../../ImplicitFunction/resources/Point.vert", "../../../../ImplicitFunction/resources/Point.frag");
    Shader edgeShader("../../../../ImplicitFunction/resources/Edge.vert", "../../../../ImplicitFunction/resources/Edge.frag");
    unsigned int VBOs[2], VAOs[2];
    glGenVertexArrays(2, VAOs);
    glGenBuffers(2, VBOs);

    // OpenGL主循环
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
        ImGui::InputFloat("image1 weight", &weight, 0.01f, 1.0f, "%.2f");
        ImGui::SameLine();
        if (weight != preWeight && weight >= 0.0f && weight <= 1.0f) {
            std::vector<Eigen::Vector3f> points;
            implicitFunctionInterpolation(weight, rows, cols, fileData_1, fileData_2, points);
            preWeight = weight;
#ifdef EDGE_MODE
            presentPointAndEdge(points, actualPointSize, edgePointSize, actualPointVertices, edgePointVertices, offset);
#else
            presentPoint(points, actualPointSize, actualPointVertices, offset);
#endif // !EDGE_MODE
        }
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
        glBindBuffer(GL_ARRAY_BUFFER, VBOs[0]);
        glBufferData(GL_ARRAY_BUFFER, actualPointSize, actualPointVertices, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, DIMENSION, GL_FLOAT, GL_FALSE, DIMENSION * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glDrawArrays(GL_TRIANGLES, 0, actualPointSize / DIMENSION / sizeof(float));

        edgeShader.use();
        edgeShader.setMat4f("model", model);
        edgeShader.setMat4f("view", view);
        edgeShader.setMat4f("projection", projection);
        glBindVertexArray(VAOs[1]);
        glBindBuffer(GL_ARRAY_BUFFER, VBOs[1]);
        glBufferData(GL_ARRAY_BUFFER, edgePointSize, edgePointVertices, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, DIMENSION, GL_FLOAT, GL_FALSE, DIMENSION * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glDrawArrays(GL_LINES, 0, edgePointSize / DIMENSION / sizeof(float));

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

    delete[] actualPointVertices;
    delete[] edgePointVertices;
    delete[] fileData_1;
    delete[] fileData_2;
#endif // !WRITE_MODE  

    return 0;
}