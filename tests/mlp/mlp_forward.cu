#include <stdio.h>
#include "../../src/mlp.cu"

int test_mlp_forward() {
    printf("\n╔════════════════════════════════════════╗\n");
    printf("║   TEST MLP FORWARD PASS COMPLET        ║\n");
    printf("╚════════════════════════════════════════╝\n\n");
    
    // Architecture du MLP : [4 → 3 → 2]
    const int NUM_LAYERS = 2;
    int layer_sizes[] = {4, 3, 2};  // input=4, hidden=3, output=2
    const int BATCH_SIZE = 2;
    
    // Créer le MLP
    MLP *mlp = create_MLP_on_GPU(layer_sizes, NUM_LAYERS);
    
    // ========== LAYER 1: [4 → 3] ==========
    float h_weights1[4][3] = {
        {0.1, 0.2, 0.3},
        {0.4, 0.5, 0.6},
        {0.7, 0.8, 0.9},
        {1.0, 1.1, 1.2}
    };
    float h_bias1[3] = {0.1, 0.2, 0.3};
    
    cudaMalloc(&mlp->weights[0], 4 * 3 * sizeof(float));
    cudaMalloc(&mlp->biases[0], 3 * sizeof(float));
    cudaMemcpy(mlp->weights[0], h_weights1, 4 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mlp->biases[0], h_bias1, 3 * sizeof(float), cudaMemcpyHostToDevice);
    
    // ========== LAYER 2: [3 → 2] ==========
    float h_weights2[3][2] = {
        {0.5, 1.0},
        {1.5, 2.0},
        {2.5, 3.0}
    };
    float h_bias2[2] = {0.5, 1.0};
    
    cudaMalloc(&mlp->weights[1], 3 * 2 * sizeof(float));
    cudaMalloc(&mlp->biases[1], 2 * sizeof(float));
    cudaMemcpy(mlp->weights[1], h_weights2, 3 * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mlp->biases[1], h_bias2, 2 * sizeof(float), cudaMemcpyHostToDevice);
    
    // ========== INPUT ==========
    float h_input[BATCH_SIZE][4] = {
        {1.0, 2.0, 3.0, 4.0},
        {0.5, 1.0, 1.5, 2.0}
    };
    
    float *d_input;
    cudaMalloc(&d_input, BATCH_SIZE * 4 * sizeof(float));
    cudaMemcpy(d_input, h_input, BATCH_SIZE * 4 * sizeof(float), cudaMemcpyHostToDevice);
    
    // ========== OUTPUT ==========
    float *d_output;
    cudaMalloc(&d_output, BATCH_SIZE * 2 * sizeof(float));
    
    // ========== FORWARD PASS ==========
    printf("Architecture: [%d → %d → %d]\n", 
           layer_sizes[0], layer_sizes[1], layer_sizes[2]);
    printf("Batch size: %d\n\n", BATCH_SIZE);
    
    printf("Input:\n");
    printf("  Sample 0: [%.1f, %.1f, %.1f, %.1f]\n", 
           h_input[0][0], h_input[0][1], h_input[0][2], h_input[0][3]);
    printf("  Sample 1: [%.1f, %.1f, %.1f, %.1f]\n\n", 
           h_input[1][0], h_input[1][1], h_input[1][2], h_input[1][3]);
    
    printf("Forward pass:\n");
    MLP_Forward(mlp, d_input, d_output, BATCH_SIZE);
    
    // ========== RESULTAT ==========
    float h_output[BATCH_SIZE][2];
    cudaMemcpy(h_output, d_output, BATCH_SIZE * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("\nOutput final:\n");
    printf("  Sample 0: [%.3f, %.3f]\n", h_output[0][0], h_output[0][1]);
    printf("  Sample 1: [%.3f, %.3f]\n", h_output[1][0], h_output[1][1]);
    
    // Libération mémoire
    cudaFree(d_input);
    cudaFree(d_output);
    free_MLP(mlp);
    
    return 0;
}

int main() {
    return test_mlp_forward();
}