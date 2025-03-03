#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// Kernel para la traslación de la imagen
__global__ void translateImageKernel(unsigned char *inputImage, unsigned char *outputImage, 
                                      int inputWidth, int inputHeight, int outputWidth, int outputHeight, 
                                      int tx, int ty) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < inputWidth && y < inputHeight) {
        // Calcular las nuevas posiciones de los píxeles
        int newX = x + tx;
        int newY = y + ty;

        if (newX >= 0 && newX < outputWidth && newY >= 0 && newY < outputHeight) {
            // Copiar el valor del píxel al nuevo lugar
            int inputIndex = (y * inputWidth + x) * 3;
            int outputIndex = (newY * outputWidth + newX) * 3;
            outputImage[outputIndex] = inputImage[inputIndex];
            outputImage[outputIndex + 1] = inputImage[inputIndex + 1];
            outputImage[outputIndex + 2] = inputImage[inputIndex + 2];
        }
    }
}

int main() {
    // Cargar la imagen usando OpenCV
    cv::Mat inputImage = cv::imread("image.jpg");
    if (inputImage.empty()) {
        std::cerr << "Error al cargar la imagen." << std::endl;
        return -1;
    }

    int inputWidth = inputImage.cols;
    int inputHeight = inputImage.rows;
    int outputWidth = inputWidth;
    int outputHeight = inputHeight;

    unsigned char *d_inputImage, *d_outputImage;

    // Asignar memoria en el dispositivo
    cudaMalloc((void**)&d_inputImage, inputWidth * inputHeight * 3 * sizeof(unsigned char));
    cudaMalloc((void**)&d_outputImage, outputWidth * outputHeight * 3 * sizeof(unsigned char));

    // Copiar la imagen a la memoria del dispositivo
    cudaMemcpy(d_inputImage, inputImage.data, inputWidth * inputHeight * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Configuración de los bloques y hilos
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((outputWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, (outputHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Llamar al kernel de traslación
    translateImageKernel<<<numBlocks, threadsPerBlock>>>(d_inputImage, d_outputImage, inputWidth, inputHeight, outputWidth, outputHeight, 50, 50); // Traslación de 50 píxeles

    // Copiar el resultado de nuevo a la memoria del host
    unsigned char *outputImage = new unsigned char[outputWidth * outputHeight * 3];
    cudaMemcpy(outputImage, d_outputImage, outputWidth * outputHeight * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Mostrar la imagen trasladada usando OpenCV
    cv::Mat outputMat(outputHeight, outputWidth, CV_8UC3, outputImage);
    cv::imshow("Traslación CUDA", outputMat);
    cv::waitKey(0);

    // Liberar memoria
    delete[] outputImage;
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return 0;
}
