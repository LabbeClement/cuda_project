#include <stdio.h>
#include "../../src/mlp.cu"

int test_feedforward() {
    const int BATCH_SIZE = 2;
    const int INPUT_DIM = 3;
    const int OUTPUT_DIM = 4;
    
    // Input: 2 samples de 3 features
    float h_input[BATCH_SIZE][INPUT_DIM] = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    // Weights: 4 neurons avec 3 inputs chacun
 float h_weights[INPUT_DIM][OUTPUT_DIM] = {
    {0.1, 0.4, 0.7, 1.0},  // poids pour input[0]
    {0.2, 0.5, 0.8, 1.1},  // poids pour input[1]
    {0.3, 0.6, 0.9, 1.2}   // poids pour input[2]
};
    // Bias: 4 neurons
    float h_bias[OUTPUT_DIM] = {0.1, 0.2, 0.3, 0.4};
    
    float h_output[BATCH_SIZE][OUTPUT_DIM];
    
    // Allocation GPU
    float *d_input, *d_weights, *d_bias, *d_output;
    cudaMalloc((void**)&d_input, BATCH_SIZE * INPUT_DIM * sizeof(float));
    cudaMalloc((void**)&d_weights, INPUT_DIM * OUTPUT_DIM * sizeof(float));
    cudaMalloc((void**)&d_bias, OUTPUT_DIM * sizeof(float));
    cudaMalloc((void**)&d_output, BATCH_SIZE * OUTPUT_DIM * sizeof(float));
    // Copie cpu -> gpu
    cudaMemcpy(d_input, h_input, BATCH_SIZE * INPUT_DIM * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, INPUT_DIM * OUTPUT_DIM * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, OUTPUT_DIM * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // Configuration des threads
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((OUTPUT_DIM + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (BATCH_SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Lancement du kernel SANS ReLU
    printf("\n=== Test FeedForward SANS ReLU ===\n");
    FeedForward(d_input, d_weights, d_bias, 
                         d_output, BATCH_SIZE, INPUT_DIM, 
                         OUTPUT_DIM, false);
    
    cudaMemcpy(h_output, d_output, BATCH_SIZE * OUTPUT_DIM * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    printf("Output (sans ReLU):\n");
    for (int i = 0; i < BATCH_SIZE; i++) {
        printf("Sample %d: ", i);
        for (int j = 0; j < OUTPUT_DIM; j++) {
            printf("%.3f ", h_output[i][j]);
        }
        printf("\n");
    }
    
    // Lancement du kernel AVEC ReLU
    printf("\n=== Test FeedForward AVEC ReLU ===\n");
    FeedForward(d_input, d_weights, d_bias, 
                         d_output, BATCH_SIZE, INPUT_DIM, 
                         OUTPUT_DIM, true);
    
    cudaMemcpy(h_output, d_output, BATCH_SIZE * OUTPUT_DIM * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    printf("Output (avec ReLU):\n");
    for (int i = 0; i < BATCH_SIZE; i++) {
        printf("Sample %d: ", i);
        for (int j = 0; j < OUTPUT_DIM; j++) {
            printf("%.3f ", h_output[i][j]);
        }
        printf("\n");
    }
    
    // Libération mémoire
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
    
    return 0;
}

int main() {
    return test_feedforward();
}