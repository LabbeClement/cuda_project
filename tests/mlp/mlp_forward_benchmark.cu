#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Inclure le contrat de la librairie pour utiliser les kernels et la structure MLP
#include "../../src/mlp.h" 

// --- DIMENSIONS DU BENCHMARK MLP ---
const int BATCH_SIZE = 4096; // Grande taille pour un benchmark significatif
const int NUM_LAYERS_MLP = 3; 
int LAYER_SIZES[] = {1024, 512, 256, 10}; // [Input, Hidden1, Hidden2, Output]

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Helper pour allouer et initialiser les poids/biais d'une couche sur le GPU
void initialize_layer_gpu(MLP *mlp, int layer_index, int input_dim, int output_dim) {
    size_t w_size = (size_t)input_dim * output_dim * sizeof(float);
    size_t b_size = (size_t)output_dim * sizeof(float);

    // Allocation et copie des poids (Weights)
    checkCudaError(cudaMalloc(&mlp->weights[layer_index], w_size), "Malloc Weights");
    // Initialisation simple pour l'exemple (remplacer par lecture de fichier si nécessaire)
    checkCudaError(cudaMemset(mlp->weights[layer_index], 0x11, w_size), "Memset Weights"); 

    // Allocation et copie des biais (Biases)
    checkCudaError(cudaMalloc(&mlp->biases[layer_index], b_size), "Malloc Biases");
    checkCudaError(cudaMemset(mlp->biases[layer_index], 0x01, b_size), "Memset Biases");
}

int benchmark_mlp_forward()
{
    printf("Architecture: [%d → %d → %d → %d]\n", 
           LAYER_SIZES[0], LAYER_SIZES[1], LAYER_SIZES[2], LAYER_SIZES[3]);
    printf("Batch size: %d\n", BATCH_SIZE);

    // Déclaration des variables de timing
    cudaEvent_t start, stop;
    float milliseconds = 0;
    
    // 1. Initialisation du MLP et des poids sur le GPU
    MLP *mlp = create_MLP_on_GPU(LAYER_SIZES, NUM_LAYERS_MLP);

    for (int i = 0; i < NUM_LAYERS_MLP; i++) {
        initialize_layer_gpu(mlp, i, LAYER_SIZES[i], LAYER_SIZES[i+1]);
    }

    // 2. Préparation de l'Input et de l'Output
    float *d_input, *d_output;
    size_t input_size = (size_t)BATCH_SIZE * LAYER_SIZES[0] * sizeof(float);
    size_t output_size = (size_t)BATCH_SIZE * LAYER_SIZES[NUM_LAYERS_MLP] * sizeof(float);
    
    checkCudaError(cudaMalloc(&d_input, input_size), "Malloc Input");
    checkCudaError(cudaMalloc(&d_output, output_size), "Malloc Output");
    checkCudaError(cudaMemset(d_input, 0x05, input_size), "Memset Input"); 

    checkCudaError(cudaEventCreate(&start), "Create Event Start");
    checkCudaError(cudaEventCreate(&stop), "Create Event Stop");

    // // 3. Warm-up (Exécution à vide)
    // MLP_Forward(mlp, d_input, d_output, BATCH_SIZE);
    // checkCudaError(cudaDeviceSynchronize(), "Warmup Sync");

    // 4. Mesure du temps
    checkCudaError(cudaEventRecord(start), "Record Start"); 

    // Exécution de l'opération (Le passage avant complet)
    MLP_Forward(mlp, d_input, d_output, BATCH_SIZE);

    checkCudaError(cudaEventRecord(stop), "Record Stop");  
    checkCudaError(cudaEventSynchronize(stop), "Final Sync"); 

    // 5. Calculer le temps écoulé
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Elapsed Time");
    
    // NOUVEAU FORMAT STANDARD pour le MLP
    printf("[Source: CUDA Naïf] [Op: MLP_Forward (%d L)] [Time: %.4f ms]\n", 
           NUM_LAYERS_MLP, milliseconds);

    // 6. Nettoyage
    cudaFree(d_input); 
    cudaFree(d_output);
    free_MLP(mlp);

    return 0;
}

// ============== MAIN ==============

int main() {
    printf("\n");
    printf("════════════════════════════════════════════════════════\n");
    printf("         BENCHMARK MLP FORWARD (VERSION NAÏVE)          \n");
    printf("════════════════════════════════════════════════════════\n\n");
    
    int result = benchmark_mlp_forward();
    
    printf("\n════════════════════════════════════════════════════════\n\n");
    
    return result;
}