#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <GL/glut.h>
#include <iostream>
#include <vector>

#define WIDTH 512
#define HEIGHT 512
#define DEPTH 512
#define MAX_ITER 100

float* h_data;
float* d_data;

// GPU Kernel para calcular el conjunto de Mandelbrot 3D
__global__ void mandelbrot3D(float* d_data, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    float scale = 2.0f / width;
    float cx = (x - width / 2) * scale;
    float cy = (y - height / 2) * scale;
    float cz = (z - depth / 2) * scale;

    float zx = cx, zy = cy, zz = cz;
    int iter;
    for (iter = 0; iter < MAX_ITER; iter++) {
        float zx2 = zx * zx - zy * zy - zz * zz + cx;
        float zy2 = 2.0f * zx * zy + cy;
        float zz2 = 2.0f * zx * zz + cz;
        
        zx = zx2;
        zy = zy2;
        zz = zz2;
        
        if (zx * zx + zy * zy + zz * zz > 4.0f) break;
    }

    d_data[z * width * height + y * width + x] = (float)iter / MAX_ITER;
}

void render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glTranslatef(-WIDTH / 2, -HEIGHT / 2, -DEPTH / 2);

    // Copiar datos de la GPU a la CPU
    cudaMemcpy(h_data, d_data, WIDTH * HEIGHT * DEPTH * sizeof(float), cudaMemcpyDeviceToHost);

    glBegin(GL_POINTS);
    for (int z = 0; z < DEPTH; z++) {
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                float value = h_data[z * WIDTH * HEIGHT + y * WIDTH + x];
                glColor3f(value, value, value);
                glVertex3f(x, y, z);
            }
        }
    }
    glEnd();
    glutSwapBuffers();
}

void initOpenGL() {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)WIDTH / (double)HEIGHT, 1.0, 1000.0);
    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_DEPTH_TEST);
}

int main(int argc, char** argv) {
    cudaMalloc(&d_data, WIDTH * HEIGHT * DEPTH * sizeof(float));
    h_data = (float*)malloc(WIDTH * HEIGHT * DEPTH * sizeof(float));
    
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks(WIDTH / threadsPerBlock.x, HEIGHT / threadsPerBlock.y, DEPTH / threadsPerBlock.z);
    mandelbrot3D<<<numBlocks, threadsPerBlock>>>(d_data, WIDTH, HEIGHT, DEPTH);
    cudaDeviceSynchronize();

    // Inicializar OpenGL
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Mandelbrot 3D CUDA");
    initOpenGL();
    glutDisplayFunc(render);
    glutMainLoop();
    
    cudaFree(d_data);
    free(h_data);
    return 0;
}
