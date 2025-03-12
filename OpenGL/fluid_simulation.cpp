// Incluye GLEW antes de cualquier otra cosa de OpenGL
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

// Estructura de una partícula
struct Particle {
    float x, y;   // Posición
    float vx, vy; // Velocidad
};

// Variables globales
const int numParticles = 100000000;
std::vector<Particle> particles(numParticles);

// Inicializa las partículas con valores aleatorios
void initParticles() {
    srand(static_cast<unsigned>(time(0))); // Semilla para valores aleatorios

    for (auto &p : particles) {
        p.x = ((rand() % 2000) / 1000.0f) - 1.0f; // Posición X entre -1 y 1
        p.y = ((rand() % 2000) / 1000.0f) - 1.0f; // Posición Y entre -1 y 1
        p.vx = ((rand() % 200) / 1000.0f) - 0.1f; // Velocidad X pequeña
        p.vy = ((rand() % 200) / 1000.0f) - 0.1f; // Velocidad Y pequeña
    }
}

// Actualiza la posición de las partículas en cada frame
void updateParticles() {
    for (auto &p : particles) {
        p.x += p.vx;
        p.y += p.vy;

        // Rebote en los bordes
        if (p.x > 1.0f || p.x < -1.0f) p.vx = -p.vx;
        if (p.y > 1.0f || p.y < -1.0f) p.vy = -p.vy;
    }
}

// Dibuja las partículas en pantalla
void drawParticles() {
    glBegin(GL_POINTS);
    glColor3f(1.0f, 1.0f, 1.0f); // Color blanco

    for (const auto &p : particles) {
        glVertex2f(p.x, p.y);
    }

    glEnd();
}

int main() {
    // Inicializa GLFW
    if (!glfwInit()) {
        std::cerr << "Error al inicializar GLFW" << std::endl;
        return -1;
    }

    // Crea la ventana
    GLFWwindow* window = glfwCreateWindow(2500, 1080, "Simulación CUDA-OpenGL", NULL, NULL);
    if (!window) {
        std::cerr << "Error al crear ventana GLFW" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Inicializa GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Error al inicializar GLEW" << std::endl;
        return -1;
    }

    // Inicializa partículas
    initParticles();

    // Loop de renderizado
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        updateParticles();  // Mueve las partículas
        drawParticles();    // Dibuja las partículas

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Limpieza
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
