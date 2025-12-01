#include <stdio.h>
#include <cuda_runtime.h>
#include "../../src/mlp.h"

const int BATCH_SIZE = 4096;
const int NUM_LAYERS = 3;
int LAYER_SIZES[] = {1024, 512, 256, 10};

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void initialize_weights(float **weights, float **biases, int *layer_sizes, int num_layers) {
    for (int i = 0; i < num_layers; i++) {
        int input_dim = layer_sizes[i];
        int output_dim = layer_sizes[i + 1];
        
        checkCudaError(cudaMalloc(&weights[i], input_dim * output_dim * sizeof(float)),
                      "Malloc Weights");
        checkCudaError(cudaMemset(weights[i], 0x11, input_dim * output_dim * sizeof(float)),
                      "Memset Weights");
        
        checkCudaError(cudaMalloc(&biases[i], output_dim * sizeof(float)),
                      "Malloc Biases");
        checkCudaError(cudaMemset(biases[i], 0x01, output_dim * sizeof(float)),
                      "Memset Biases");
    }
}

void benchmark_version(const char* name, void (*forward_func)(MLP*, float*, float*, int),
                      MLP *mlp, float *d_input, float *d_output) {
    cudaEvent_t start, stop;
    float milliseconds = 0;
    
    checkCudaError(cudaEventCreate(&start), "Create Event Start");
    checkCudaError(cudaEventCreate(&stop), "Create Event Stop");
    
    // Warm-up
    forward_func(mlp, d_input, d_output, BATCH_SIZE);
    checkCudaError(cudaDeviceSynchronize(), "Warmup Sync");
    
    // Mesure
    checkCudaError(cudaEventRecord(start), "Record Start");
    forward_func(mlp, d_input, d_output, BATCH_SIZE);
    checkCudaError(cudaEventRecord(stop), "Record Stop");
    checkCudaError(cudaEventSynchronize(stop), "Final Sync");
    
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Elapsed Time");
    
    printf("[Source: %s] [Op: MLP_Forward (%d L)] [Time: %.4f ms]\n",
           name, NUM_LAYERS, milliseconds);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void benchmark_optimized(const char* name,
                        void (*forward_func)(MLP_Optimized*, float*, float*),
                        MLP_Optimized *mlp, float *d_input, float *d_output) {
    cudaEvent_t start, stop;
    float milliseconds = 0;
    
    checkCudaError(cudaEventCreate(&start), "Create Event Start");
    checkCudaError(cudaEventCreate(&stop), "Create Event Stop");
    
    // Warm-up
    forward_func(mlp, d_input, d_output);
    checkCudaError(cudaDeviceSynchronize(), "Warmup Sync");
    
    // Mesure
    checkCudaError(cudaEventRecord(start), "Record Start");
    forward_func(mlp, d_input, d_output);
    checkCudaError(cudaEventRecord(stop), "Record Stop");
    checkCudaError(cudaEventSynchronize(stop), "Final Sync");
    
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Elapsed Time");
    
    printf("[Source: %s] [Op: MLP_Forward (%d L)] [Time: %.4f ms]\n",
           name, NUM_LAYERS, milliseconds);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


// Wrapper pour utiliser FeedForward_Tiled dans le MLP
void MLP_Forward_Tiled_Wrapper(MLP *mlp, float *input, float *output, int batch_size) {
    int num_layers = mlp->num_layers;
    
    float **activations = (float**)malloc((num_layers + 1) * sizeof(float*));
    activations[0] = input;
    
    for (int i = 1; i <= num_layers; i++) {
        int layer_output_size = mlp->layer_sizes[i];
        cudaMalloc(&activations[i], batch_size * layer_output_size * sizeof(float));
    }
    
    for (int layer = 0; layer < num_layers; layer++) {
        int input_dim = mlp->layer_sizes[layer];
        int output_dim = mlp->layer_sizes[layer + 1];
        bool apply_relu = (layer < num_layers - 1);
        
        FeedForward_Tiled(activations[layer], 
                         mlp->weights[layer], 
                         mlp->biases[layer],
                         activations[layer + 1],
                         batch_size, input_dim, output_dim, apply_relu);
    }
    
    int final_size = mlp->layer_sizes[num_layers];
    cudaMemcpy(output, activations[num_layers], 
               batch_size * final_size * sizeof(float),
               cudaMemcpyDeviceToDevice);
    
    for (int i = 1; i < num_layers; i++) {
        cudaFree(activations[i]);
    }
    free(activations);
}

int main() {
    printf("\n╔════════════════════════════════════════════════════════╗\n");
    printf("║   BENCHMARK OPTIMISATIONS MLP (4 VERSIONS)            ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n\n");
    
    printf("Architecture: [%d → %d → %d → %d]\n",
           LAYER_SIZES[0], LAYER_SIZES[1], LAYER_SIZES[2], LAYER_SIZES[3]);
    printf("Batch size: %d\n\n", BATCH_SIZE);
    
    // Préparer l'input et l'output
    float *d_input, *d_output;
    size_t input_size = (size_t)BATCH_SIZE * LAYER_SIZES[0] * sizeof(float);
    size_t output_size = (size_t)BATCH_SIZE * LAYER_SIZES[NUM_LAYERS] * sizeof(float);
    
    checkCudaError(cudaMalloc(&d_input, input_size), "Malloc Input");
    checkCudaError(cudaMalloc(&d_output, output_size), "Malloc Output");
    checkCudaError(cudaMemset(d_input, 0x05, input_size), "Memset Input");
    
    // ========== VERSION 1 : MODULAIRE (BASELINE) ==========
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║ 1. Version Modulaire (Baseline)                       ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n");
    MLP *mlp_base = create_MLP_on_GPU(LAYER_SIZES, NUM_LAYERS);
    initialize_weights(mlp_base->weights, mlp_base->biases, LAYER_SIZES, NUM_LAYERS);
    benchmark_version("CUDA Modulaire", MLP_Forward, mlp_base, d_input, d_output);
    
    // ========== VERSION 2 : SHARED MEMORY TILING ==========
    printf("\n╔════════════════════════════════════════════════════════╗\n");
    printf("║ 2. Shared Memory + Tiling                             ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n");
    MLP *mlp_tiled = create_MLP_on_GPU(LAYER_SIZES, NUM_LAYERS);
    initialize_weights(mlp_tiled->weights, mlp_tiled->biases, LAYER_SIZES, NUM_LAYERS);
    benchmark_version("CUDA Tiled", MLP_Forward_Tiled_Wrapper, mlp_tiled, d_input, d_output);
    free_MLP(mlp_tiled);
    
    // ========== VERSION 3 : MLP OPTIMISÉ (BUFFERS PRÉ-ALLOUÉS + KERNEL FUSIONNÉ) ==========
    printf("\n╔════════════════════════════════════════════════════════╗\n");
    printf("║ 3. Buffers Pré-alloués + Kernel Fusionné              ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n");
    MLP_Optimized *mlp_opt = create_MLP_Optimized(LAYER_SIZES, NUM_LAYERS, BATCH_SIZE);
    initialize_weights(mlp_opt->weights, mlp_opt->biases, LAYER_SIZES, NUM_LAYERS);
    benchmark_optimized("CUDA Optimisé (Fused)", MLP_Forward_Optimized, mlp_opt, d_input, d_output);
    
    // ========== VERSION 4 : cuBLAS ==========
    printf("\n╔════════════════════════════════════════════════════════╗\n");
    printf("║ 4. cuBLAS (Bibliothèque NVIDIA Optimisée)             ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n");
    benchmark_optimized("CUDA cuBLAS", MLP_Forward_Optimized_cuBLAS, mlp_opt, d_input, d_output);
    
    free_MLP_Optimized(mlp_opt);
    free_MLP(mlp_base);
    
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}