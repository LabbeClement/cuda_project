#include <stdio.h>
#include <cuda_runtime.h>
#include "../../src/cnn.h"

// Macro helper
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

int main() {
    printf("=== TEST CONVOLUTION 2D ===\n");

    // Paramètres simples pour vérification manuelle
    int batch = 1;
    int in_c = 1;
    int out_c = 1;
    int h = 3, w = 3;
    int k = 2; // kernel 2x2
    int stride = 1;
    int padding = 0;

    // Output dim: (3 - 2)/1 + 1 = 2x2
    int out_h = 2, out_w = 2;

    // Allocation Host
    float h_input[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    // Kernel identité (pour tester) ou simple
    float h_kernel[] = {
        1, 1,
        0, 0
    };
    // Attendu:
    // HAUT-GAUCHE: 1*1 + 2*1 + 4*0 + 5*0 = 3
    // HAUT-DROITE: 2*1 + 3*1 + 5*0 + 6*0 = 5
    // BAS-GAUCHE:  4*1 + 5*1 = 9
    // BAS-DROITE:  5*1 + 6*1 = 11

    float *d_input, *d_kernel, *d_output;
    float *h_output = (float*)malloc(out_h * out_w * sizeof(float));

    CHECK_CUDA(cudaMalloc(&d_input, 9 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel, 4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, 4 * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, 9 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, 4 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_output, 0, 4 * sizeof(float)));

    // Lancement Kernel
    Conv2D(d_input, d_kernel, d_output, 
           batch, in_c, out_c, h, w, k, k, stride, padding);
    
    CHECK_CUDA(cudaDeviceSynchronize());

    // Récupération
    CHECK_CUDA(cudaMemcpy(h_output, d_output, 4 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Input:\n1 2 3\n4 5 6\n7 8 9\n\nKernel:\n1 1\n0 0\n\n");
    printf("Output (Attendu: 3, 5, 9, 11):\n");
    for(int i=0; i<out_h; i++) {
        for(int j=0; j<out_w; j++) {
            printf("%.1f ", h_output[i*out_w + j]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_output);

    return 0;
}