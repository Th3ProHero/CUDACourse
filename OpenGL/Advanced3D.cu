#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include <cmath>

#define N 10000  // Número de partículas
#define DT 0.01f // Paso de tiempo

struct Particle {
    float3 position;
    float3 velocity;
};

// OpenGL y CUDA
GLuint vbo;                          // Vertex Buffer Object para las posiciones
struct cudaGraphicsResource *cudaVBO; // Recurso CUDA para acceder al VBO
GLuint shaderProgram;

const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
out vec3 color;

void main()
{
    gl_Position = vec4(aPos / 50.0, 1.0);
    color = vec3(1.0 - abs(aPos.y / 50.0), abs(aPos.y / 50.0), 1.0);
    gl_PointSize = 3.0;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
in vec3 color;
out vec4 FragColor;

void main()
{
    FragColor = vec4(color, 1.0);
}
)";

// Kernel CUDA: Actualiza la posición de las partículas
__global__ void updateParticles(float3 *positions, float3 *velocities, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        positions[i].x += velocities[i].x * dt;
        positions[i].y += velocities[i].y * dt;
        positions[i].z += velocities[i].z * dt;
        velocities[i].y -= 9.81f * dt;
        if (positions[i].y < -50.0f) velocities[i].y *= -0.8f;
    }
}

void compileShader(GLuint &shader, const char* source, GLenum type) {
    shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
}

void initShaders() {
    GLuint vertexShader, fragmentShader;
    compileShader(vertexShader, vertexShaderSource, GL_VERTEX_SHADER);
    compileShader(fragmentShader, fragmentShaderSource, GL_FRAGMENT_SHADER);
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void initCUDA() {
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(float3), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&cudaVBO, vbo, cudaGraphicsRegisterFlagsNone);
}

void display() {
    float3 *d_positions;
    size_t size;
    cudaGraphicsMapResources(1, &cudaVBO, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_positions, &size, cudaVBO);
    static float3 *d_velocities = nullptr;
    if (!d_velocities) {
        cudaMalloc(&d_velocities, N * sizeof(float3));
        float3 *h_velocities = new float3[N];
        for (int i = 0; i < N; i++) h_velocities[i] = {float(rand() % 10 - 5), float(rand() % 10 - 5), float(rand() % 10 - 5)};
        cudaMemcpy(d_velocities, h_velocities, N * sizeof(float3), cudaMemcpyHostToDevice);
        delete[] h_velocities;
    }
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    updateParticles<<<blocksPerGrid, threadsPerBlock>>>(d_positions, d_velocities, DT);
    cudaDeviceSynchronize();
    cudaGraphicsUnmapResources(1, &cudaVBO, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(shaderProgram);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glDrawArrays(GL_POINTS, 0, N);
    glDisableVertexAttribArray(0);
    glfwSwapBuffers(glfwGetCurrentContext());
}

int main() {
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(800, 600, "CUDA Particles", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();
    initShaders();
    initCUDA();
    while (!glfwWindowShouldClose(window)) {
        display();
        glfwPollEvents();
    }
    cudaGraphicsUnregisterResource(cudaVBO);
    glDeleteBuffers(1, &vbo);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
