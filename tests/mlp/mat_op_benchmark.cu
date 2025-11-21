#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "../../src/mlp.h"

// --- DIMENSIONS DU BENCHMARK ---
const int N_BENCH = 4096;
const int M_BENCH = 4096;
const int K_BENCH = 4096;

// Fonction pour gérer les erreurs CUDA (bonne pratique !)
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


// ============== 1. BENCHMARK ADDITION (MatAdd) ==============

int benchmark_matrix_add()
{
    // Déclaration des variables
    float *GPU_a, *GPU_b, *GPU_c;
    cudaEvent_t start, stop;
    float milliseconds = 0;

    // 1. Allocation GPU et Initialisation
    checkCudaError(cudaMalloc((void**)&GPU_a, (size_t)N_BENCH * M_BENCH * sizeof(float)), "Malloc A");
    checkCudaError(cudaMalloc((void**)&GPU_b, (size_t)N_BENCH * M_BENCH * sizeof(float)), "Malloc B");
    checkCudaError(cudaMalloc((void**)&GPU_c, (size_t)N_BENCH * M_BENCH * sizeof(float)), "Malloc C");
    
    checkCudaError(cudaMemset(GPU_a, 1.0f, N_BENCH * M_BENCH * sizeof(float)), "Memset A"); 
    checkCudaError(cudaMemset(GPU_b, 2.0f, N_BENCH * M_BENCH * sizeof(float)), "Memset B");
    
    // Création et Configuration du Kernel
    checkCudaError(cudaEventCreate(&start), "Create Event Start");
    checkCudaError(cudaEventCreate(&stop), "Create Event Stop");

    dim3 threadsPerBlock(16, 16); 
    dim3 numBlocks((M_BENCH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N_BENCH + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 2. Warm-up (Chauffage GPU)
    MatAdd<<<numBlocks, threadsPerBlock>>>(GPU_a, GPU_b, GPU_c, N_BENCH, M_BENCH);
    checkCudaError(cudaDeviceSynchronize(), "Warmup Sync");

    // 3. Mesure du temps
    checkCudaError(cudaEventRecord(start), "Record Start"); 

    // Exécution de l'opération
    MatAdd<<<numBlocks, threadsPerBlock>>>(GPU_a, GPU_b, GPU_c, N_BENCH, M_BENCH);

    checkCudaError(cudaEventRecord(stop), "Record Stop");  
    checkCudaError(cudaEventSynchronize(stop), "Final Sync"); 

    // 4. Calculer le temps écoulé
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Elapsed Time");
    
    // NOUVEAU FORMAT STANDARD
    printf("[Source: CUDA Naïf] [Op: MatAdd] [Time: %.4f ms]\n", milliseconds);

    // 5. Nettoyage
    cudaFree(GPU_a); cudaFree(GPU_b); cudaFree(GPU_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}


// ============== 2. BENCHMARK MULTIPLICATION (MatMult) ==============

int benchmark_matrix_mult()
{
    // Déclaration des variables
    float *GPU_a, *GPU_b, *GPU_c;
    cudaEvent_t start, stop;
    float milliseconds = 0;

    // 1. Allocation GPU
    // A [N, K], B [K, M], C [N, M]
    checkCudaError(cudaMalloc((void**)&GPU_a, (size_t)N_BENCH * K_BENCH * sizeof(float)), "Malloc A");
    checkCudaError(cudaMalloc((void**)&GPU_b, (size_t)K_BENCH * M_BENCH * sizeof(float)), "Malloc B");
    checkCudaError(cudaMalloc((void**)&GPU_c, (size_t)N_BENCH * M_BENCH * sizeof(float)), "Malloc C");
    
    checkCudaError(cudaMemset(GPU_a, 1.0f, N_BENCH * K_BENCH * sizeof(float)), "Memset A"); 
    checkCudaError(cudaMemset(GPU_b, 2.0f, K_BENCH * M_BENCH * sizeof(float)), "Memset B");

    // Création et Configuration du Kernel
    checkCudaError(cudaEventCreate(&start), "Create Event Start");
    checkCudaError(cudaEventCreate(&stop), "Create Event Stop");

    // Threads pour la matrice de résultat C [N, M]
    dim3 threadsPerBlock(16, 16); 
    dim3 numBlocks((M_BENCH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N_BENCH + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 2. Warm-up
    MatMult<<<numBlocks, threadsPerBlock>>>(GPU_a, GPU_b, GPU_c, N_BENCH, M_BENCH, K_BENCH);
    checkCudaError(cudaDeviceSynchronize(), "Warmup Sync");

    // 3. Mesure du temps
    checkCudaError(cudaEventRecord(start), "Record Start"); 

    // Exécution de l'opération
    MatMult<<<numBlocks, threadsPerBlock>>>(GPU_a, GPU_b, GPU_c, N_BENCH, M_BENCH, K_BENCH);

    checkCudaError(cudaEventRecord(stop), "Record Stop");  
    checkCudaError(cudaEventSynchronize(stop), "Final Sync"); 

    // 4. Calculer le temps écoulé
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Elapsed Time");
    
    // NOUVEAU FORMAT STANDARD
    printf("[Source: CUDA Naïf] [Op: MatMult] [Time: %.4f ms]\n", milliseconds);

    // 5. Nettoyage
    cudaFree(GPU_a); cudaFree(GPU_b); cudaFree(GPU_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}

// ============== MAIN ==============

int main() {
    benchmark_matrix_add();
    benchmark_matrix_mult();

    return 0;
}