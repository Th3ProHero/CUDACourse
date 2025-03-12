// Incluye GLEW antes de cualquier otra cosa de OpenGL
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include <cmath>

// Estructura de una partícula
struct Particle {
    float x, y;
    float vx, vy;
};

// Variables globales
const int numParticles = 100000000;
std::vector<Particle> particles(numParticles);

void initParticles() {
    for (auto &p : particles) {
        p.x = ((rand() % 2000) / 1000.0f) - 1.0f;
        p.y = ((rand() % 2000) / 1000.0f) - 1.0f;
        p.vx = ((rand() % 200) / 1000.0f) - 0.1f;
        p.vy = ((rand() % 200) / 1000.0f) - 0.1f;
    }
}

void updateParticles() {
    for (auto &p : particles) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x > 1.0f || p.x < -1.0f) p.vx = -p.vx;
        if (p.y > 1.0f || p.y < -1.0f) p.vy = -p.vy;
    }
}

void drawParticles() {
    glBegin(GL_POINTS);
    glColor3f(1.0f, 1.0f, 1.0f);
    for (const auto &p : particles) {
        glVertex2f(p.x, p.y);
    }
    glEnd();
}

int main() {
    if (!glfwInit()) {
        std::cerr << "Error al inicializar GLFW" << std::endl;
        return -1;
    }
    
    GLFWwindow* window = glfwCreateWindow(2500, 1080, "Simulación CUDA-OpenGL", NULL, NULL);
    if (!window) {
        std::cerr << "Error al crear ventana GLFW" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    
    if (glewInit() != GLEW_OK) {
        std::cerr << "Error al inicializar GLEW" << std::endl;
        return -1;
    }

    initParticles();

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);
        updateParticles();
        drawParticles();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
