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

void run_benchmark_compare(const char* name, CNN* cnn, int batch_size, int input_h, int input_w, int iterations) {
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

    printf("\n>>> Benchmarking %s (Batch: %d) <<<\n", name, batch_size);
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
    printf("Speedup: %.2fx\n\n", ms_naive / ms_im2col);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int batch_size = 64;
    int iterations = 20;

    // --- VGG-Style (Plus gros) ---
    // Input 3x64x64
    // Conv1: 3->32, 3x3, pad 1
    // Pool1
    // Conv2: 32->64, 3x3, pad 1
    // Pool2
    // Conv3: 64->128, 3x3, pad 1
    // Pool3
    // Flatten: 128 * 8 * 8 = 8192
    // FC: 8192 -> 512 -> 10
    
    int vgg_conv[] = {
        32, 3, 1, 1, 2, 2,
        64, 3, 1, 1, 2, 2,
        128, 3, 1, 1, 2, 2
    };
    int vgg_mlp[] = {128*8*8, 512, 10};

    CNN *cnn_vgg = create_CNN(3, 64, 64, 3, vgg_conv, vgg_mlp, 2);
    
    // Alloc weights MLP (hack bench)
    for(int i=0; i<2; i++) {
        int w = cnn_vgg->mlp->layer_sizes[i] * cnn_vgg->mlp->layer_sizes[i+1];
        int b = cnn_vgg->mlp->layer_sizes[i+1];
        CHECK_CUDA(cudaMalloc(&cnn_vgg->mlp->weights[i], w * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&cnn_vgg->mlp->biases[i], b * sizeof(float)));
    }

    // CORRECTION : Ajout du paramètre input_w (64) manquant
    run_benchmark_compare("VGG-Like Medium", cnn_vgg, batch_size, 64, 64, 20);

    free_CNN(cnn_vgg);

    return 0;
}