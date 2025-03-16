#include <iostream>
#include <cuda_runtime.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>

// Parámetros del Mandelbulb
const int WIDTH = 1024;
const int HEIGHT = 1024;
const int DEPTH = 1024;
const int MAX_ITERATIONS = 600;
const float THRESHOLD = 0.1f; // Umbral para filtrar puntos

// Variables globales para la rotación
float angleX = 0.0f;
float angleY = 0.0f;

// Kernel de CUDA para generar el Mandelbulb
__global__ void mandelbulbKernel(float* output, int width, int height, int depth, int maxIterations) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        float cx = (x - width / 2.0f) / (width / 4.0f);
        float cy = (y - height / 2.0f) / (height / 4.0f);
        float cz = (z - depth / 2.0f) / (depth / 4.0f);

        float zx = cx;
        float zy = cy;
        float zz = cz;

        int iteration = 0;
        while (iteration < maxIterations && zx * zx + zy * zy + zz * zz < 4.0f) {
            float r = sqrtf(zx * zx + zy * zy + zz * zz);
            float theta = atan2f(sqrtf(zx * zx + zy * zy), zz);
            float phi = atan2f(zy, zx);

            float zr = powf(r, 8);
            float thetaNew = theta * 8;
            float phiNew = phi * 8;

            zx = zr * sinf(thetaNew) * cosf(phiNew) + cx;
            zy = zr * sinf(thetaNew) * sinf(phiNew) + cy;
            zz = zr * cosf(thetaNew) + cz;

            iteration++;
        }

        // Almacenar el valor normalizado
        output[z * width * height + y * width + x] = (float)iteration / maxIterations;
    }
}

// Función para generar el Mandelbulb
void generateMandelbulb(float* output, int width, int height, int depth, int maxIterations) {
    float* d_output;
    cudaMalloc(&d_output, width * height * depth * sizeof(float));

    dim3 threadsPerBlock(8, 8, 8);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (depth + threadsPerBlock.z - 1) / threadsPerBlock.z);

    mandelbulbKernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, width, height, depth, maxIterations);

    cudaMemcpy(output, d_output, width * height * depth * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_output);
}

// Shaders
const char* vertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 FragPos;
out vec3 Normal;
void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
in vec3 FragPos;
in vec3 Normal;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;
void main() {
    // Luz ambiental
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // Luz difusa
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Luz especular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    // Combinar componentes
    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}
)";

// Función para compilar shaders
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    return shader;
}

// Función para crear un programa de shaders
GLuint createShaderProgram(const char* vertexSource, const char* fragmentSource) {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    GLint success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

// Función para calcular normales usando diferencias finitas
void calculateNormals(const float* data, std::vector<glm::vec3>& vertices, std::vector<glm::vec3>& normals) {
    for (int z = 1; z < DEPTH - 1; z++) {
        for (int y = 1; y < HEIGHT - 1; y++) {
            for (int x = 1; x < WIDTH - 1; x++) {
                float value = data[z * WIDTH * HEIGHT + y * WIDTH + x];
                if (value > THRESHOLD) {
                    float dx = data[z * WIDTH * HEIGHT + y * WIDTH + (x + 1)] - data[z * WIDTH * HEIGHT + y * WIDTH + (x - 1)];
                    float dy = data[z * WIDTH * HEIGHT + (y + 1) * WIDTH + x] - data[z * WIDTH * HEIGHT + (y - 1) * WIDTH + x];
                    float dz = data[(z + 1) * WIDTH * HEIGHT + y * WIDTH + x] - data[(z - 1) * WIDTH * HEIGHT + y * WIDTH + x];
                    glm::vec3 normal = glm::normalize(glm::vec3(dx, dy, dz));
                    normals.push_back(normal);
                    vertices.push_back(glm::vec3(
                        (x - WIDTH / 2.0f) / (WIDTH / 4.0f),
                        (y - HEIGHT / 2.0f) / (HEIGHT / 4.0f),
                        (z - DEPTH / 2.0f) / (DEPTH / 4.0f)
                    ));
                }
            }
        }
    }
}

// Callback para manejar las teclas
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        switch (key) {
            case GLFW_KEY_UP:
                angleX += 5.0f; // Rotar hacia arriba
                break;
            case GLFW_KEY_DOWN:
                angleX -= 5.0f; // Rotar hacia abajo
                break;
            case GLFW_KEY_LEFT:
                angleY -= 5.0f; // Rotar hacia la izquierda
                break;
            case GLFW_KEY_RIGHT:
                angleY += 5.0f; // Rotar hacia la derecha
                break;
        }
    }
}

int main() {
    // Inicializar GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Crear una ventana GLFW
    GLFWwindow* window = glfwCreateWindow(800, 600, "Mandelbulb 3D", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    // Inicializar GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Configurar el callback del teclado
    glfwSetKeyCallback(window, keyCallback);

    // Crear el programa de shaders
    GLuint shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);

    // Generar el Mandelbulb
    float* output = new float[WIDTH * HEIGHT * DEPTH];
    generateMandelbulb(output, WIDTH, HEIGHT, DEPTH, MAX_ITERATIONS);

    // Filtrar puntos y calcular normales
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    calculateNormals(output, vertices, normals);

    // Configurar VBO y VAO
    GLuint VBO[2], VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(2, VBO);

    glBindVertexArray(VAO);

    // VBO para vértices
    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // VBO para normales
    glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(glm::vec3), normals.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Configurar la cámara y la proyección
    glm::mat4 view = glm::lookAt(
        glm::vec3(0.0f, 0.0f, 5.0f), // Posición de la cámara
        glm::vec3(0.0f, 0.0f, 0.0f), // Punto al que mira la cámara
        glm::vec3(0.0f, 1.0f, 0.0f)  // Vector "arriba"
    );
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);

    // Obtener las ubicaciones de los uniforms
    GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
    GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");
    GLuint projectionLoc = glGetUniformLocation(shaderProgram, "projection");
    GLuint lightPosLoc = glGetUniformLocation(shaderProgram, "lightPos");
    GLuint viewPosLoc = glGetUniformLocation(shaderProgram, "viewPos");
    GLuint lightColorLoc = glGetUniformLocation(shaderProgram, "lightColor");
    GLuint objectColorLoc = glGetUniformLocation(shaderProgram, "objectColor");

    // Configurar la luz
    glm::vec3 lightPos(2.0f, 2.0f, 2.0f);
    glm::vec3 lightColor(1.0f, 1.0f, 1.0f);
    glm::vec3 objectColor(0.5f, 0.5f, 1.0f);

    // Habilitar la prueba de profundidad
    glEnable(GL_DEPTH_TEST);

    // Bucle principal de renderizado
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);

        // Calcular la matriz de modelo con rotación
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::rotate(model, glm::radians(angleX), glm::vec3(1.0f, 0.0f, 0.0f));
        model = glm::rotate(model, glm::radians(angleY), glm::vec3(0.0f, 1.0f, 0.0f));

        // Pasar las matrices a los shaders
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

        // Pasar propiedades de la luz
        glUniform3fv(lightPosLoc, 1, glm::value_ptr(lightPos));
        glUniform3fv(viewPosLoc, 1, glm::value_ptr(glm::vec3(0.0f, 0.0f, 5.0f)));
        glUniform3fv(lightColorLoc, 1, glm::value_ptr(lightColor));
        glUniform3fv(objectColorLoc, 1, glm::value_ptr(objectColor));

        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, vertices.size());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Limpieza
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(2, VBO);
    glDeleteProgram(shaderProgram);

    delete[] output;
    glfwTerminate();
    return 0;
}