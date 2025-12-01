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

// Helper pour initialiser la mémoire GPU
void init_gpu_data(float *ptr, int size, float value) {
    float *temp = (float*)malloc(size * sizeof(float));
    for(int i=0; i<size; i++) temp[i] = value;
    CHECK_CUDA(cudaMemcpy(ptr, temp, size * sizeof(float), cudaMemcpyHostToDevice));
    free(temp);
}

// Fonction générique de benchmark
void run_benchmark(const char* name, CNN* cnn, int batch_size, int input_h, int input_w, int iterations) {
    int input_elements = batch_size * cnn->conv_layers[0].in_channels * input_h * input_w;
    
    // On doit calculer la taille de sortie du MLP pour allouer le buffer de sortie
    // Le dernier layer du MLP donne la taille de sortie
    int num_mlp = cnn->mlp->num_layers;
    int output_dim = cnn->mlp->layer_sizes[num_mlp];
    int output_elements = batch_size * output_dim;

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, output_elements * sizeof(float)));
    
    // Init data
    CHECK_CUDA(cudaMemset(d_input, 0, input_elements * sizeof(float)));

    // Init weights (important pour éviter les NaNs qui pourraient ralentir sur certains HW, même si rare en float)
    // On le fait une fois lors de la création normalement, ici on assume que create_CNN ne remplit pas tout.
    // Pour le bench, on veut juste mesurer les calculs, peu importe les valeurs.

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    printf("Benchmarking %s (Batch: %d)...\n", name, batch_size);

    // Warmup
    for(int i=0; i<5; i++) {
        CNN_Forward(cnn, d_input, d_output, batch_size);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Mesure
    cudaEventRecord(start);
    for(int i=0; i<iterations; i++) {
        CNN_Forward(cnn, d_input, d_output, batch_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / iterations;

    printf("   [Result] Avg Time: %.4f ms\n\n", avg_ms);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║          BENCHMARK CNN PERFORMANCE           ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    int batch_size = 64;
    int iterations = 20;

    // --- MODELE 1 : MNIST (Petit) ---
    // Conv1: 1->4, 3x3
    // Pool1
    // Conv2: 4->8, 3x3
    // Pool2
    // FC: 200 -> 64 -> 10
    int mnist_conv[] = {4, 3, 1, 0, 2, 2,  8, 3, 1, 0, 2, 2};
    int mnist_mlp[] = {8*5*5, 64, 10}; 
    
    CNN *cnn_mnist = create_CNN(1, 28, 28, 2, mnist_conv, mnist_mlp, 2);
    
    // Hack: Allouer la mémoire des poids du MLP car create_CNN ne le fait pas (dépend de load_file)
    // et notre implémentation de create_CNN + create_MLP est disjointe sur ce point.
    for(int i=0; i<2; i++) {
        int w = cnn_mnist->mlp->layer_sizes[i] * cnn_mnist->mlp->layer_sizes[i+1];
        int b = cnn_mnist->mlp->layer_sizes[i+1];
        CHECK_CUDA(cudaMalloc(&cnn_mnist->mlp->weights[i], w * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&cnn_mnist->mlp->biases[i], b * sizeof(float)));
    }

    run_benchmark("MNIST Small (Naive CUDA)", cnn_mnist, batch_size, 28, 28, iterations);


    // --- MODELE 2 : VGG-Style (Plus gros) ---
    // Input 3x64x64
    // Conv1: 3->32, 3x3, pad 1
    // Pool1
    // Conv2: 32->64, 3x3, pad 1
    // Pool2
    // Conv3: 64->128, 3x3, pad 1
    // Pool3
    // Flatten: 128 * 8 * 8 = 8192
    // FC: 8192 -> 512 -> 10
    
    int vgg_conv[] = {
        32, 3, 1, 1, 2, 2,
        64, 3, 1, 1, 2, 2,
        128, 3, 1, 1, 2, 2
    };
    int vgg_mlp[] = {128*8*8, 512, 10};

    CNN *cnn_vgg = create_CNN(3, 64, 64, 3, vgg_conv, vgg_mlp, 2);
    
    for(int i=0; i<2; i++) {
        int w = cnn_vgg->mlp->layer_sizes[i] * cnn_vgg->mlp->layer_sizes[i+1];
        int b = cnn_vgg->mlp->layer_sizes[i+1];
        CHECK_CUDA(cudaMalloc(&cnn_vgg->mlp->weights[i], w * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&cnn_vgg->mlp->biases[i], b * sizeof(float)));
    }

    run_benchmark("VGG-Like Medium (Naive CUDA)", cnn_vgg, batch_size, 64, 64, iterations);

    free_CNN(cnn_mnist);
    free_CNN(cnn_vgg);

    return 0;
}