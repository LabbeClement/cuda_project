#ifndef MLP_H
#define MLP_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

// ============== PRIMITIVES DE BASE ==============
__global__ void MatAdd(float *A, float *B, float *C, int n, int m);
__global__ void MatMult(float *A, float *B, float *C, int n, int m, int common);
__global__ void ReLU(float* x, float *y, int size);
__global__ void AddBias(float *in, float *bias, float *out, int batch_size, int output_dim);

// ============== FEEDFORWARD VERSIONS ==============
// Version modulaire (originale)
void FeedForward(float *input, float *weights, float *bias, 
                 float *output, int batch_size, int input_dim, 
                 int output_dim, bool apply_relu);

// OPTIMISATION 1: Kernel fusionné
__global__ void FeedForward_Fused_Kernel(float *input, float *weights, float *bias,
                                          float *output, int batch_size, int input_dim,
                                          int output_dim, bool apply_relu);

void FeedForward_Fused(float *input, float *weights, float *bias,
                       float *output, int batch_size, int input_dim,
                       int output_dim, bool apply_relu);

// OPTIMISATION 2: cuBLAS
void FeedForward_cuBLAS(cublasHandle_t handle, float *input, float *weights, 
                        float *bias, float *output, int batch_size, 
                        int input_dim, int output_dim, bool apply_relu);

// OPTIMISATION 3: Shared Memory Tiled MatMult
#define TILE_SIZE 48
__global__ void MatMult_Tiled(float *A, float *B, float *C, int n, int m, int common);

void FeedForward_Tiled(float *input, float *weights, float *bias,
                       float *output, int batch_size, int input_dim,
                       int output_dim, bool apply_relu);

// OPTIMISATION 4: Kernel fusionné + Tiling + Shared Memory
__global__ void FeedForward_Fused_Tiled_Kernel(float *input, float *weights, float *bias,
                                               float *output, int batch_size, int input_dim,
                                               int output_dim, bool apply_relu);
void FeedForward_Fused_Tiled(float *input, float *weights, float *bias,
                             float *output, int batch_size, int input_dim,
                             int output_dim, bool apply_relu);

// ============== MLP ==============
typedef struct {
    int num_layers;
    int *layer_sizes;
    float **weights;
    float **biases;
} MLP;

MLP* create_MLP_on_GPU(int *layer_sizes_host, int num_layers);
void free_MLP(MLP *mlp);
void MLP_Forward(MLP *mlp, float *input, float *output, int batch_size);

// OPTIMISATION 4: MLP avec buffers pré-alloués
typedef struct {
    int num_layers;
    int *layer_sizes;
    float **weights;
    float **biases;
    float **activation_buffers;  // Pré-alloués
    int batch_size;
    cublasHandle_t cublas_handle;  // Pour version cuBLAS
} MLP_Optimized;

MLP_Optimized* create_MLP_Optimized(int *layer_sizes_host, int num_layers, int batch_size);
void free_MLP_Optimized(MLP_Optimized *mlp);
void MLP_Forward_Optimized(MLP_Optimized *mlp, float *input, float *output);
void MLP_Forward_Optimized_cuBLAS(MLP_Optimized *mlp, float *input, float *output);
void MLP_Forward_Optimized_Fused_Tiled(MLP_Optimized *mlp, float *input, float *output);

// Chargement
MLP* load_MLP_from_file(const char* filename);
void save_MLP_to_file(MLP *mlp, const char* filename, int batch_size);
MLP* load_MLP_from_onnx(const char* onnx_filename);

#endif // MLP_H