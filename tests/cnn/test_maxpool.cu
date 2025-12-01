#include <stdio.h>
#include <cuda_runtime.h>
#include "../../src/cnn.h"

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

int main() {
    printf("=== TEST MAX POOLING 2D ===\n");

    int batch = 1;
    int channels = 1;
    int h = 4, w = 4;
    int pool_size = 2;
    int stride = 2;

    // Output dim: (4 - 2)/2 + 1 = 2
    int out_h = 2, out_w = 2;

    // Input 4x4
    // 1  2  | 3  1
    // 4  5  | 0  2
    // ------+-----
    // 8  1  | 9  9
    // 2  3  | 1  4
    
    // MaxPool 2x2 donne :
    // max(1,2,4,5) = 5   | max(3,1,0,2) = 3
    // -------------------+-----------------
    // max(8,1,2,3) = 8   | max(9,9,1,4) = 9

    float h_input[] = {
        1, 2, 3, 1,
        4, 5, 0, 2,
        8, 1, 9, 9,
        2, 3, 1, 4
    };

    float *d_input, *d_output;
    float *h_output = (float*)malloc(out_h * out_w * sizeof(float));

    CHECK_CUDA(cudaMalloc(&d_input, 16 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, 4 * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, 16 * sizeof(float), cudaMemcpyHostToDevice));

    MaxPool2D(d_input, d_output, batch, channels, h, w, pool_size, stride);
    
    CHECK_CUDA(cudaMemcpy(h_output, d_output, 4 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Input (4x4):\n");
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) printf("%.0f ", h_input[i*4+j]);
        printf("\n");
    }

    printf("\nOutput (Attendu: 5, 3, 8, 9):\n");
    for(int i=0; i<out_h; i++) {
        for(int j=0; j<out_w; j++) {
            printf("%.0f ", h_output[i*out_w + j]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output);

    return 0;
}