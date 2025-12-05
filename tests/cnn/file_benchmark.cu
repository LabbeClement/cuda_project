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

// Fonction de mesure comparative
void benchmark_loaded_cnn(const char* model_name, const char* filename, int batch_size, int input_h, int input_w, int iterations) {
    printf("\n>>> Chargement: %s (%s) <<<\n", model_name, filename);
    
    CNN *cnn = load_CNN_from_file(filename, input_h, input_w);
    if (!cnn) {
        printf("ERREUR: Impossible de charger le fichier. Avez-vous lance le script Python ?\n");
        return;
    }

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

    printf("    Benchmarking (Batch: %d)...\n", batch_size);
    printf("    --------------------------------------------------\n");
    printf("    | %-20s | %-12s |\n", "Version", "Avg Time (ms)");
    printf("    --------------------------------------------------\n");

    // 1. NAIVE
    for(int i=0; i<3; i++) CNN_Forward(cnn, d_input, d_output, batch_size); // Warmup
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for(int i=0; i<iterations; i++) CNN_Forward(cnn, d_input, d_output, batch_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_naive = 0;
    cudaEventElapsedTime(&ms_naive, start, stop);
    printf("    | %-20s | %10.4f ms |\n", "Naive Conv2D", ms_naive / iterations);

    // 2. IM2COL + CUBLAS
    for(int i=0; i<3; i++) CNN_Forward_Im2Col(cnn, d_input, d_output, batch_size); // Warmup
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for(int i=0; i<iterations; i++) CNN_Forward_Im2Col(cnn, d_input, d_output, batch_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_im2col = 0;
    cudaEventElapsedTime(&ms_im2col, start, stop);
    printf("    | %-20s | %10.4f ms |\n", "Im2Col + cuBLAS", ms_im2col / iterations);
    
    printf("    --------------------------------------------------\n");
    printf("    Speedup: %.2fx\n", ms_naive / ms_im2col);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free_CNN(cnn);
}

int main() {
    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║    BENCHMARK CNN SUR FICHIERS REELS          ║\n");
    printf("╚══════════════════════════════════════════════╝\n");

    // TEST 1 : MNIST (Petit modèle, Petite image)
    // Batch 4096 pour essayer de saturer un peu le GPU
    benchmark_loaded_cnn("MNIST Small", "tests/data/cnn_mnist.txt", 4096, 28, 28, 10);

    // TEST 2 : VGG-Like (Moyen modèle, Image moyenne)
    // Batch 64 pour simuler un cas réel d'inférence
    benchmark_loaded_cnn("VGG-Like", "tests/data/cnn_vgg.txt", 64, 64, 64, 10);

    return 0;
}