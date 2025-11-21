#ifndef MLP_H
#define MLP_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Déclarations des Kernels (Fonctions GPU)
__global__ void MatAdd(float *A, float *B, float *C, int n, int m);
__global__ void MatMult(float *A, float *B, float *C, int n, int m, int common);
__global__ void ReLU(float* x, float *y, int size);
__global__ void AddBias(float *in, float *bias, float *out, int batch_size, int output_dim);

// Déclarations des Fonctions Host (CPU)
void FeedForward(float *input, float *weights, float *bias, 
                         float *output, int batch_size, int input_dim, 
                         int output_dim, bool apply_relu);

// Structure et fonctions MLP
typedef struct {
    int num_layers;
    int *layer_sizes;
    float **weights;
    float **biases;
} MLP;

MLP* create_MLP_on_GPU(int *layer_sizes_host, int num_layers);
void free_MLP(MLP *mlp);
void MLP_Forward(MLP *mlp, float *input, float *output, int batch_size);

#endif // MLP_H