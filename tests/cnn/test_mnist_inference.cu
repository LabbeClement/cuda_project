#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include "../../src/cnn.h"

// Helper lecture fichier float
void load_data_from_txt(const char* filename, float* buffer, int size) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Erreur ouverture %s\n", filename);
        exit(1);
    }
    for(int i=0; i<size; i++) {
        if(fscanf(f, "%f", &buffer[i]) != 1) {
            fprintf(stderr, "Erreur lecture element %d dans %s\n", i, filename);
            exit(1);
        }
    }
    fclose(f);
}

int argmax(float* data, int size) {
    int idx = 0;
    float max_val = data[0];
    for(int i=1; i<size; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            idx = i;
        }
    }
    return idx;
}

int main() {
    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║   TEST CNN INFERENCE SUR MNIST (Chiffre 7)   ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    const char* model_path = "tests/data/cnn_mnist.txt";
    const char* input_path = "tests/data/mnist_sample_in.txt";
    const char* output_path = "tests/data/mnist_sample_out.txt";
    
    // 1. Charger le Modèle
    // MNIST images are 1x28x28
    CNN *cnn = load_CNN_from_file(model_path, 28, 28);
    if (!cnn) return 1;

    // 2. Charger les Données
    int batch_size = 1;
    int input_size = 1 * 28 * 28;
    int output_size = 10;
    
    float *h_input = (float*)malloc(input_size * sizeof(float));
    float *h_expected = (float*)malloc(output_size * sizeof(float));
    
    load_data_from_txt(input_path, h_input, input_size);
    load_data_from_txt(output_path, h_expected, output_size);
    
    int expected_label = argmax(h_expected, output_size);
    printf("Image chargée. Label attendu (depuis Python): %d\n", expected_label);

    // 3. Préparer GPU
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // 4. Inférence
    printf("Lancement de l'inférence CUDA...\n");
    CNN_Forward(cnn, d_input, d_output, batch_size);
    cudaDeviceSynchronize();
    
    // 5. Résultats
    float *h_output = (float*)malloc(output_size * sizeof(float));
    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    int predicted_label = argmax(h_output, output_size);
    
    printf("\n--- RÉSULTATS ---\n");
    printf("Classe prédite par CUDA : %d\n", predicted_label);
    printf("Confiance (Logits)      :\n");
    printf("   [");
    for(int i=0; i<10; i++) printf("% .2f ", h_output[i]);
    printf("]\n");
    
    if (predicted_label == expected_label) {
        printf("\nSUCCÈS : La prédiction correspond !\n");
    } else {
        printf("\nÉCHEC : Prédiction incorrecte.\n");
    }

    // Cleanup
    free(h_input);
    free(h_expected);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    free_CNN(cnn);

    return 0;
}