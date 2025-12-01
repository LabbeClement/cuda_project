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

// Helper pour initialiser la mémoire GPU avec une valeur constante
void init_gpu_data(float *ptr, int size, float value) {
    float *temp = (float*)malloc(size * sizeof(float));
    for(int i=0; i<size; i++) temp[i] = value;
    CHECK_CUDA(cudaMemcpy(ptr, temp, size * sizeof(float), cudaMemcpyHostToDevice));
    free(temp);
}

int main() {
    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║   TEST CNN COMPLET (FORWARD PASS INTEGRATION) ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    // 1. CONFIGURATION DU RÉSEAU
    // On simule une petite image 10x10 (pour faciliter le calcul mental des dimensions)
    int batch_size = 2;
    int input_c = 1;
    int input_h = 10;
    int input_w = 10;

    // Architecture Conv: 2 couches
    int num_conv = 2;
    // Format config: [out_c, k, s, p, pool_k, pool_s]
    // Layer 1: 1->4 ch, 3x3, same pad, pool 2x2
    // Layer 2: 4->8 ch, 3x3, same pad, pool 2x2
    int conv_configs[] = {
        4, 3, 1, 1, 2, 2, 
        8, 3, 1, 1, 2, 2   
    };
    
    // CALCUL MANUEL DES DIMENSIONS POUR LE TEST
    // L1: 10x10 -> (Conv 3x3, p=1) 10x10 -> (Pool 2x2) 5x5
    // L2: 5x5   -> (Conv 3x3, p=1) 5x5   -> (Pool 2x2) 2x2 (division entière de 5/2)
    // Flatten Size = 8 channels * 2 * 2 = 32 neurones
    
    int mlp_input_size = 32; 
    int num_mlp_layers = 2; // Une couche cachée + une couche sortie
    // MLP: [Input 32] -> [Hidden 16] -> [Output 10 classes]
    int mlp_layers[] = {mlp_input_size, 16, 10};

    printf("1. Création du CNN...\n");
    printf("   Input: [%d, %d, %d, %d]\n", batch_size, input_c, input_h, input_w);
    CNN *cnn = create_CNN(input_c, input_h, input_w, num_conv, conv_configs, mlp_layers, num_mlp_layers);

    // 2. INITIALISATION DES POIDS (Valeurs arbitraires pour éviter les NaNs)
    printf("2. Initialisation des poids (dummy values)...\n");
    
    // Init Conv Layers
    for(int i=0; i<num_conv; i++) {
        ConvLayer *l = &cnn->conv_layers[i];
        int w_size = l->out_channels * l->in_channels * l->kernel_size * l->kernel_size;
        int b_size = l->out_channels;
        
        // Poids = 0.1 * (layer_index + 1)
        init_gpu_data(l->weights, w_size, 0.1f * (i+1)); 
        init_gpu_data(l->biases, b_size, 0.01f);
        
        printf("   Conv Layer %d initialized (%d weights)\n", i+1, w_size);
    }
    
    // Init MLP Layers
    for(int i=0; i<cnn->mlp->num_layers; i++) {
        int w_size = cnn->mlp->layer_sizes[i] * cnn->mlp->layer_sizes[i+1];
        int b_size = cnn->mlp->layer_sizes[i+1];
        
        // --- CORRECTIF SEGFAULT : Allocation mémoire explicite ---
        CHECK_CUDA(cudaMalloc(&cnn->mlp->weights[i], w_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&cnn->mlp->biases[i], b_size * sizeof(float)));
        // -------------------------------------------------------
        
        init_gpu_data(cnn->mlp->weights[i], w_size, 0.05f);
        init_gpu_data(cnn->mlp->biases[i], b_size, 0.01f);
        
        printf("   MLP Layer %d initialized (%d weights)\n", i+1, w_size);
    }

    // 3. PRÉPARATION INPUT
    int input_elements = batch_size * input_c * input_h * input_w;
    float *d_input, *d_output;
    int output_elements = batch_size * 10; // 10 classes

    CHECK_CUDA(cudaMalloc(&d_input, input_elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, output_elements * sizeof(float)));

    // Image remplie de 1.0f
    init_gpu_data(d_input, input_elements, 1.0f); 

    // 4. EXÉCUTION
    printf("\n3. Lancement Forward Pass...\n");
    
    // On utilise un Event pour mesurer le temps du CNN complet
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    CNN_Forward(cnn, d_input, d_output, batch_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("   Temps d'exécution : %.3f ms\n", milliseconds);

    // 5. VÉRIFICATION RÉSULTATS
    float *h_output = (float*)malloc(output_elements * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, output_elements * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\n4. Résultats (Batch size %d, 10 Classes):\n", batch_size);
    // On affiche juste les 5 premières classes pour voir si ça varie ou non
    for(int b=0; b<batch_size; b++) {
        printf("   Sample %d: [ ", b);
        for(int i=0; i<5; i++) {
            printf("%.4f ", h_output[b*10 + i]);
        }
        printf("... ]\n");
    }
    
    // Vérification basique : les valeurs ne doivent pas être 0, ni NaN, ni Infinity
    if (h_output[0] <= 0.0f || h_output[0] >= 10000.0f) {
        printf("\nÉCHEC : Valeurs suspectes (0 ou NaN).\n");
    }

    // CLEANUP
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output);
    free_CNN(cnn);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}