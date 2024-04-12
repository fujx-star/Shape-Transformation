// ImplicitFunction.cpp: 定义应用程序的入口点。

#include "ImplicitFunction.h"
#include "LinearSystem.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <vector>

using namespace std;

float RBF(glm::vec2 x) {
    float length = glm::length(x);
    if (length == 0.0f) {
        return 0.0f;
    }
    return glm::pow(length, 2.0f) * glm::log(length);
}

float interpolationFunction(glm::vec2 x, 
    const vector<pair<glm::vec2, float>>& constraints,
    const vector<float>& weights,
    const vector<float>& P) {
    float res = 0.0f;
    for (int i = 0; i < constraints.size(); i++) {
        res += weights[i] * RBF(x - constraints[i].first);
    }
    res += P[0] + P[1] * x.x + P[2] * x.y;
    return res;
}

int main()
{

    vector<pair<glm::vec2, float>> constraints = {
        // boundary constraints
        {{1.0f, 1.0f}, 0.0f},
        {{-1.0f, 1.0f}, 0.0f},
        {{-1.0f, -1.0f}, 0.0f},
        {{1.0f, -1.0f}, 0.0f},
        // noramal  constraints
        {{0.9f, 0.9f}, 1.0f},
        {{-0.9f, 0.9f}, 1.0f},
        {{-0.9f, -0.9f}, 1.0f},
        {{0.9f, -0.9f}, 1.0f}
    };

    int numConstraints = constraints.size();
    int n = numConstraints + 1 + 2;
    vector<vector<float>> A(n, vector<float>(n, 0.0f));
    vector<vector<float>> L(n, vector<float>(n, 0.0f));
    vector<vector<float>> U(n, vector<float>(n, 0.0f));
    vector<float> B(n, 0.0f), X(n, 0.0f), Y(n, 0.0f);

    for (int i = 0; i < numConstraints; i++) {
        for (int j = 0; j < numConstraints; j++) {
            A[i][j] = RBF(constraints[i].first - constraints[j].first);
        }
    }
    for (int i = 0; i < numConstraints; i++) {
        A[i][numConstraints] = 1.0f;
        A[i][numConstraints + 1] = constraints[i].first.x;
        A[i][numConstraints + 2] = constraints[i].first.y;
    }
    for (int i = 0; i < numConstraints; i++) {
        A[numConstraints][i] = 1.0f;
        A[numConstraints + 1][i] = constraints[i].first.x;
        A[numConstraints + 2][i] = constraints[i].first.y;
    }
    for (int i = 0; i < n; i++) {
        if (i < numConstraints) {
            B[i] = constraints[i].second;
        }
        else {
            B[i] = 0.0f;
        }
    }

    // A = L * U
    lu(A, L, U, n);
    // L * Y = B
    LYCompute(L, B, Y, n);
    // U * X = Y
    UXCompute(U, Y, X, n);

    vector<float> weights(numConstraints, 0.0f);
    for (int i = 0; i < numConstraints; i++) {
        weights[i] = X[i];
    }
    vector<float> P(n - numConstraints, 0.0f);
    for (int i = 0; i < n - numConstraints; i++) {
        P[i] = X[numConstraints + i];
    }

    //A[0][0] = 1.0f;
    //A[0][1] = 1.0f;
    //A[1][0] = 2.0f;
    //A[1][1] = 3.0f;
    //B[0] = 3.0f;
    //B[1] = 8.0f;

    

    for (int i = 0; i < n; i++) {
        cout << X[i] << " ";
    }

    //glm::vec3 p = { 0.5f, 0.5f, 0.0f };
    //cout << RBF(p);
	
	return 0;
}


//#include <glad/glad.h>
//#include <GLFW/glfw3.h>
//#include <imgui.h>
//#include <imgui_impl_glfw.h>
//#include <imgui_impl_opengl3.h>
//#include <glm/glm.hpp>
//#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtc/type_ptr.hpp>
//#include <vector>
//#include "settings/settings.hpp"
//#include "settings/Shader.h"
//#include "tools/Quaternion.hpp"
//#include "algorithm/Rotate.hpp"
//#include <iostream>
//
////int main() {
////
////	glm::vec3 point{ 1,4,6 };
////	glm::vec3 axis{ 6,7,1 };
////	double angle = 45;
////
////	glm::vec3 result = rotateByMatrix(point, axis, angle);
////	glm::vec3 result2 = rotateByQuaternion(point, axis, angle);
////	glm::vec3 result3 = rotateTransform(point, axis, angle);
////	for (int i = 0; i < 3; i++) {
////		std::cout << result[i] << " ";
////	}
////	std::cout << std::endl;
////
////	for (int i = 0; i < 3; i++) {
////		std::cout << result2[i] << " ";
////	}
////	std::cout << std::endl;
////
////	for (int i = 0; i < 3; i++) {
////		std::cout << result3[i] << " ";
////	}
////	std::cout << std::endl;
////
////	return 0;
////
////}
//
//struct Point {
//    glm::vec3 position;
//    glm::vec3 normal;
//};
//glm::vec3 lightPos(1.2f, 1.0f, 2.0f);
//
//int main()
//{
//    glfwInit();
//    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
//
//    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
//    if (window == NULL)
//    {
//        std::cout << "Failed to create GLFW window" << std::endl;
//        glfwTerminate();
//        return -1;
//    }
//
//    glfwMakeContextCurrent(window);
//    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
//    glfwSetCursorPosCallback(window, mouse_callback);
//    glfwSetScrollCallback(window, scroll_callback);
//    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
//
//    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
//    {
//        std::cout << "Failed to initialize GLAD" << std::endl;
//        return -1;
//    }
//
//    glEnable(GL_DEPTH_TEST);
//
//    IMGUI_CHECKVERSION();
//    ImGui::CreateContext();
//    ImGuiIO& io = ImGui::GetIO();
//    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
//    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
//
//    ImGui_ImplGlfw_InitForOpenGL(window, true);          // Second param install_callback=true will install GLFW callbacks and chain to existing ones.
//    ImGui_ImplOpenGL3_Init();
//
//
//    Shader objectShader("resources/quaternion/Object.vert", "resources/quaternion/Object.frag");
//    Shader lightShader("resources/quaternion/Light.vert", "resources/quaternion/Light.frag");
//
//    std::vector<Point> points = {
//        Point{glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.0f,  0.0f, -1.0f)},
//        Point{glm::vec3(0.5f, -0.5f, -0.5f), glm::vec3(0.0f,  0.0f, -1.0f)},
//        Point{glm::vec3(0.5f,  0.5f, -0.5f), glm::vec3(0.0f,  0.0f, -1.0f)},
//        Point{glm::vec3(0.5f,  0.5f, -0.5f), glm::vec3(0.0f,  0.0f, -1.0f)},
//        Point{glm::vec3(-0.5f,  0.5f, -0.5f), glm::vec3(0.0f,  0.0f, -1.0f)},
//        Point{glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.0f,  0.0f, -1.0f)},
//
//        Point{glm::vec3(-0.5f, -0.5f,  0.5f), glm::vec3(0.0f,  0.0f,  1.0f)},
//        Point{glm::vec3(0.5f, -0.5f,  0.5f), glm::vec3(0.0f,  0.0f,  1.0f)},
//        Point{glm::vec3(0.5f,  0.5f,  0.5f), glm::vec3(0.0f,  0.0f,  1.0f)},
//        Point{glm::vec3(0.5f,  0.5f,  0.5f), glm::vec3(0.0f,  0.0f,  1.0f)},
//        Point{glm::vec3(-0.5f,  0.5f,  0.5f), glm::vec3(0.0f,  0.0f,  1.0f)},
//        Point{glm::vec3(-0.5f, -0.5f,  0.5f), glm::vec3(0.0f,  0.0f,  1.0f)},
//
//        Point{glm::vec3(-0.5f,  0.5f,  0.5f), glm::vec3(-1.0f,  0.0f,  0.0f)},
//        Point{glm::vec3(-0.5f,  0.5f, -0.5f), glm::vec3(-1.0f,  0.0f,  0.0f)},
//        Point{glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(-1.0f,  0.0f,  0.0f)},
//        Point{glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(-1.0f,  0.0f,  0.0f)},
//        Point{glm::vec3(-0.5f, -0.5f,  0.5f), glm::vec3(-1.0f,  0.0f,  0.0f)},
//        Point{glm::vec3(-0.5f,  0.5f,  0.5f), glm::vec3(-1.0f,  0.0f,  0.0f)},
//
//        Point{glm::vec3(0.5f,  0.5f,  0.5f), glm::vec3(1.0f,  0.0f,  0.0f)},
//        Point{glm::vec3(0.5f,  0.5f, -0.5f), glm::vec3(1.0f,  0.0f,  0.0f)},
//        Point{glm::vec3(0.5f, -0.5f, -0.5f), glm::vec3(1.0f,  0.0f,  0.0f)},
//        Point{glm::vec3(0.5f, -0.5f, -0.5f), glm::vec3(1.0f,  0.0f,  0.0f)},
//        Point{glm::vec3(0.5f, -0.5f,  0.5f), glm::vec3(1.0f,  0.0f,  0.0f)},
//        Point{glm::vec3(0.5f,  0.5f,  0.5f), glm::vec3(1.0f,  0.0f,  0.0f)},
//
//        Point{glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.0f, -1.0f,  0.0f)},
//        Point{glm::vec3(0.5f, -0.5f, -0.5f), glm::vec3(0.0f, -1.0f,  0.0f)},
//        Point{glm::vec3(0.5f, -0.5f,  0.5f), glm::vec3(0.0f, -1.0f,  0.0f)},
//        Point{glm::vec3(0.5f, -0.5f,  0.5f), glm::vec3(0.0f, -1.0f,  0.0f)},
//        Point{glm::vec3(-0.5f, -0.5f,  0.5f), glm::vec3(0.0f, -1.0f,  0.0f)},
//        Point{glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.0f, -1.0f,  0.0f)},
//
//        Point{glm::vec3(-0.5f,  0.5f, -0.5f), glm::vec3(0.0f,  1.0f,  0.0f)},
//        Point{glm::vec3(0.5f,  0.5f, -0.5f), glm::vec3(0.0f,  1.0f,  0.0f)},
//        Point{glm::vec3(0.5f,  0.5f,  0.5f), glm::vec3(0.0f,  1.0f,  0.0f)},
//        Point{glm::vec3(0.5f,  0.5f,  0.5f), glm::vec3(0.0f,  1.0f,  0.0f)},
//        Point{glm::vec3(-0.5f,  0.5f,  0.5f), glm::vec3(0.0f,  1.0f,  0.0f)},
//        Point{glm::vec3(-0.5f,  0.5f, -0.5f), glm::vec3(0.0f,  1.0f,  0.0f)}
//    };
//
//    float lightVertices[108];
//    int index = 0;
//    for (const auto& point : points) {
//        lightVertices[index++] = point.position.x;
//        lightVertices[index++] = point.position.y;
//        lightVertices[index++] = point.position.z;
//    }
//
//    unsigned int objectVBO, objectVAO;
//    glGenVertexArrays(1, &objectVAO);
//    glGenBuffers(1, &objectVBO);
//
//    unsigned int lightVBO, lightVAO;
//    glGenVertexArrays(1, &lightVAO);
//    glGenBuffers(1, &lightVBO);
//
//    while (!glfwWindowShouldClose(window))
//    {
//        float currentFrame = static_cast<float>(glfwGetTime());
//        deltaTime = currentFrame - lastFrame;
//        lastFrame = currentFrame;
//
//        ImGui_ImplOpenGL3_NewFrame();
//        ImGui_ImplGlfw_NewFrame();
//        ImGui::NewFrame();
//        ImGui::Begin("window!");
//        ImGui::Text("helloworld.");
//
//        static float angle = 0.0f;
//        static glm::vec3 axis = glm::vec3(0.0f, 0.0f, 0.0f);
//        ImGui::InputFloat("input axis-x", &axis.x, 0.05f, 1.0f, "%.3f");
//        ImGui::InputFloat("input axis-y", &axis.y, 0.05f, 1.0f, "%.3f");
//        ImGui::InputFloat("input axis-z", &axis.z, 0.05f, 1.0f, "%.3f");
//        ImGui::InputFloat("input angle", &angle, 1.0f, 1.0f, "%.1f");
//        if (ImGui::Button("Rotate")) {
//            for (auto& point : points) {
//                point.position = rotateByQuaternion(point.position, axis, angle);
//                point.normal = rotateByQuaternion(point.normal, axis, angle);
//            }
//        }
//        ImGui::SameLine();
//        ImGui::End();
//
//        processInput(window);
//
//
//        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
//        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
//        glClear(GL_COLOR_BUFFER_BIT);
//
//        // draw object
//        objectShader.use();
//
//        objectShader.setVec3f("objectColor", glm::vec3(1.0f, 0.5f, 0.31f));
//        objectShader.setVec3f("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
//        objectShader.setVec3f("lightPos", lightPos);
//        objectShader.setVec3f("viewPos", camera.Position);
//
//        glm::mat4 model = glm::mat4(1.0f);
//        glm::mat4 view = glm::mat4(1.0f);
//        glm::mat4 projection = glm::mat4(1.0f);
//        view = camera.GetViewMatrix();
//        projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
//
//        objectShader.setMat4f("model", model);
//        objectShader.setMat4f("view", view);
//        objectShader.setMat4f("projection", projection);
//
//        int verticesSize = points.size() * 6 * sizeof(float);
//        float* vertices = (float*)malloc(verticesSize);
//        for (int i = 0; i < points.size(); i++) {
//            vertices[i * 6 + 0] = points[i].position.x;
//            vertices[i * 6 + 1] = points[i].position.y;
//            vertices[i * 6 + 2] = points[i].position.z;
//            vertices[i * 6 + 3] = points[i].normal.x;
//            vertices[i * 6 + 4] = points[i].normal.y;
//            vertices[i * 6 + 5] = points[i].normal.z;
//        }
//
//        glBindVertexArray(objectVAO);
//
//        glBindBuffer(GL_ARRAY_BUFFER, objectVBO);
//        glBufferData(GL_ARRAY_BUFFER, verticesSize, vertices, GL_DYNAMIC_DRAW);
//
//        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
//        glEnableVertexAttribArray(0);
//
//        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
//        glEnableVertexAttribArray(1);
//
//        glDrawArrays(GL_TRIANGLES, 0, 36);
//
//
//        // draw light
//        lightShader.use();
//
//        model = glm::mat4(1.0f);
//        model = glm::translate(model, lightPos);
//        model = glm::scale(model, glm::vec3(0.2f));
//        lightShader.setMat4f("model", model);
//        lightShader.setMat4f("projection", projection);
//        lightShader.setMat4f("view", view);
//
//        glBindVertexArray(lightVAO);
//
//        glBindBuffer(GL_ARRAY_BUFFER, lightVBO);
//        glBufferData(GL_ARRAY_BUFFER, 108 * sizeof(float), lightVertices, GL_STATIC_DRAW);
//
//        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
//        glEnableVertexAttribArray(0);
//
//        glDrawArrays(GL_TRIANGLES, 0, 36);
//
//        ImGui::Render();
//        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
//
//        glfwSwapBuffers(window);
//        glfwPollEvents();
//
//        free(vertices);
//    }
//
//    glDeleteVertexArrays(1, &objectVAO);
//    glDeleteBuffers(1, &objectVBO);
//    glDeleteVertexArrays(1, &lightVAO);
//    glDeleteBuffers(1, &lightVBO);
//    objectShader.Delete();
//    lightShader.Delete();
//
//    ImGui_ImplOpenGL3_Shutdown();
//    ImGui_ImplGlfw_Shutdown();
//    ImGui::DestroyContext();
//
//    glfwTerminate();
//    return 0;
//}