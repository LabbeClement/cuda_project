#include <stdio.h>
#include "../../src/mlp.h"

int test_load_from_file() {
    printf("\n╔════════════════════════════════════════╗\n");
    printf("║   TEST CHARGEMENT MLP DEPUIS FICHIER   ║\n");
    printf("╚════════════════════════════════════════╝\n\n");
    
    // 1. Charger le MLP depuis le fichier
    MLP *mlp = load_MLP_from_file("tests/data/mlp_weights_small.txt");
    
    if (!mlp) {
        printf("Échec du chargement\n");
        return 1;
    }
    
    // 2. Préparer l'input
    const int BATCH_SIZE = 2;
    float h_input[BATCH_SIZE][4] = {
        {1.0, 2.0, 3.0, 4.0},
        {0.5, 1.0, 1.5, 2.0}
    };
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, BATCH_SIZE * 4 * sizeof(float));
    cudaMalloc(&d_output, BATCH_SIZE * 2 * sizeof(float));
    
    cudaMemcpy(d_input, h_input, BATCH_SIZE * 4 * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // 3. Forward pass
    printf("Input:\n");
    printf("  Sample 0: [%.1f, %.1f, %.1f, %.1f]\n", 
           h_input[0][0], h_input[0][1], h_input[0][2], h_input[0][3]);
    printf("  Sample 1: [%.1f, %.1f, %.1f, %.1f]\n\n", 
           h_input[1][0], h_input[1][1], h_input[1][2], h_input[1][3]);
    
    printf("Forward pass:\n");
    MLP_Forward(mlp, d_input, d_output, BATCH_SIZE);
    
    // 4. Récupérer les résultats
    float h_output[BATCH_SIZE][2];
    cudaMemcpy(h_output, d_output, BATCH_SIZE * 2 * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    printf("\nOutput final:\n");
    printf("  Sample 0: [%.3f, %.3f]\n", h_output[0][0], h_output[0][1]);
    printf("  Sample 1: [%.3f, %.3f]\n", h_output[1][0], h_output[1][1]);
    
    printf("\nRésultats attendus:\n");
    printf("  Sample 0: [39.600, 52.400]\n");
    printf("  Sample 1: [20.600, 27.400]\n");
    
    // 5. Test de sauvegarde (bonus)
    printf("\n--- Test de sauvegarde ---\n");
    save_MLP_to_file(mlp, "tests/data/mlp_weights_saved.txt", BATCH_SIZE);
    
    // Nettoyage
    cudaFree(d_input);
    cudaFree(d_output);
    free_MLP(mlp);
    
    return 0;
}

int main() {
    return test_load_from_file();
}