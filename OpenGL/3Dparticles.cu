#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <GL/glut.h>
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

// Kernel CUDA: Actualiza la posición de las partículas
__global__ void updateParticles(float3 *positions, float3 *velocities, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Actualizar posición con integración de Euler
        positions[i].x += velocities[i].x * dt;
        positions[i].y += velocities[i].y * dt;
        positions[i].z += velocities[i].z * dt;

        // Simular gravedad en el eje Y
        velocities[i].y -= 9.81f * dt;

        // Rebote en el suelo (y < -50)
        if (positions[i].y < -50.0f) {
            velocities[i].y *= -0.8f;  // Rebote con amortiguación
        }
    }
}

// Inicializar OpenGL y CUDA
void initCUDA() {
    // Crear VBO en OpenGL
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(float3), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Registrar el VBO en CUDA
    cudaGraphicsGLRegisterBuffer(&cudaVBO, vbo, cudaGraphicsRegisterFlagsNone);
}

// Renderizar las partículas con OpenGL
void display() {
    // Mapear el VBO a CUDA
    float3 *d_positions;
    size_t size;
    cudaGraphicsMapResources(1, &cudaVBO, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_positions, &size, cudaVBO);

    // Generar velocidades aleatorias
    static float3 *d_velocities = nullptr;
    if (!d_velocities) {
        cudaMalloc(&d_velocities, N * sizeof(float3));
        float3 *h_velocities = new float3[N];
        for (int i = 0; i < N; i++) {
            h_velocities[i] = {float(rand() % 10 - 5), float(rand() % 10 - 5), float(rand() % 10 - 5)};
        }
        cudaMemcpy(d_velocities, h_velocities, N * sizeof(float3), cudaMemcpyHostToDevice);
        delete[] h_velocities;
    }

    // Ejecutar el kernel de CUDA
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    updateParticles<<<blocksPerGrid, threadsPerBlock>>>(d_positions, d_velocities, DT);
    cudaDeviceSynchronize();

    // Desmapear el VBO en CUDA
    cudaGraphicsUnmapResources(1, &cudaVBO, 0);

    // Renderizar con OpenGL
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPointSize(2.0f);
    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glDrawArrays(GL_POINTS, 0, N);
    glDisableClientState(GL_VERTEX_ARRAY);
    glutSwapBuffers();
}

// Configurar OpenGL
void initOpenGL() {
    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45, 1, 1, 1000);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0, 50, 150, 0, 0, 0, 0, 1, 0);
}

// Bucle principal de OpenGL
void idle() {
    glutPostRedisplay();
}

// Función principal
int main(int argc, char **argv) {
    // Inicializar GLUT y OpenGL
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 600);
    glutCreateWindow("CUDA Particles");

    glewInit();
    initOpenGL();
    initCUDA();

    // Configurar callbacks
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutMainLoop();

    // Limpiar recursos
    cudaGraphicsUnregisterResource(cudaVBO);
    glDeleteBuffers(1, &vbo);
    return 0;
}
