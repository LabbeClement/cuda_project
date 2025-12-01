#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../../src/cnn.h"

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// Fonction de mesure comparative (copiée et adaptée de cnn_benchmark)
void benchmark_loaded_cnn(const char* model_name, CNN* cnn, int batch_size, int input_h, int input_w, int iterations) {
    int input_elements = batch_size * cnn->conv_layers[0].in_channels * input_h * input_w;
    int num_mlp = cnn->mlp->num_layers;
    int output_dim = cnn->mlp->layer_sizes[num_mlp];
    int output_elements = batch_size * output_dim;

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, output_elements * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_input, 0, input_elements * sizeof(float)));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    printf("\n>>> Benchmarking Loaded Model: %s (Batch: %d) <<<\n", model_name, batch_size);
    printf("--------------------------------------------------\n");
    printf("| %-20s | %-12s |\n", "Version", "Avg Time (ms)");
    printf("--------------------------------------------------\n");

    // 1. NAIVE
    for(int i=0; i<5; i++) CNN_Forward(cnn, d_input, d_output, batch_size); // Warmup
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for(int i=0; i<iterations; i++) CNN_Forward(cnn, d_input, d_output, batch_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_naive = 0;
    cudaEventElapsedTime(&ms_naive, start, stop);
    printf("| %-20s | %10.4f ms |\n", "Naive Conv2D", ms_naive / iterations);

    // 2. IM2COL + CUBLAS
    for(int i=0; i<5; i++) CNN_Forward_Im2Col(cnn, d_input, d_output, batch_size); // Warmup
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for(int i=0; i<iterations; i++) CNN_Forward_Im2Col(cnn, d_input, d_output, batch_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_im2col = 0;
    cudaEventElapsedTime(&ms_im2col, start, stop);
    printf("| %-20s | %10.4f ms |\n", "Im2Col + cuBLAS", ms_im2col / iterations);
    
    printf("--------------------------------------------------\n");
    printf("Speedup: %.2fx\n", ms_naive / ms_im2col);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║      BENCHMARK CNN SUR FICHIER CHARGÉ        ║\n");
    printf("╚══════════════════════════════════════════════╝\n");

    const char* model_path = "tests/data/cnn_mnist.txt";
    int batch_size = 4096; // Gros batch pour bien voir les gains cuBLAS
    int input_h = 28;
    int input_w = 28;

    // 1. Chargement
    CNN *cnn = load_CNN_from_file(model_path, input_h, input_w);
    
    if (cnn) {
        // 2. Benchmark
        benchmark_loaded_cnn("MNIST Trained Model", cnn, batch_size, input_h, input_w, 20);
        free_CNN(cnn);
    } else {
        printf("Erreur: Impossible de charger %s\n", model_path);
        return 1;
    }

    return 0;
}