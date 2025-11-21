#include <stdio.h>

#include "mlp.h"

const int N = 5;
const int M = 3;


// ============== Matrix Addition ==============

__global__ void MatAdd(float *A, float *B, float *C, int n, int m)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n && j < m) {
        int idx = i * m + j;
        C[idx] = A[idx] + B[idx];
    }
}

// ============== Matrix Multiplication ==============

__global__ void MatMult(float *A, float *B, float *C, int n, int m, int common)
{
    int row_res = blockIdx.y * blockDim.y + threadIdx.y; // Ligne de la matrice résultat
    int col_res = blockIdx.x * blockDim.x + threadIdx.x; // Colonne de la matrice résultat

    if (row_res < n && col_res < m)
    {
        float value = 0;
        for (int k = 0; k < common; k++)
        {
            value += A[row_res * common + k] * B[k * m + col_res];
        }
        C[row_res * m + col_res] = value;
    }
}


// ============== RELU ==============

__global__ void ReLU(float* x, float *y, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (x[idx] < 0)
        {
            y[idx] = 0.0f;
        }
        else {
            y[idx] = fmaxf(0.0f, x[idx]);
        }
    }
}

// Kernel pour ajouter le biais (broadcast sur les colonnes)
__global__ void AddBias(float *in, float *bias, float *out, int batch_size, int output_dim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch_size && col < output_dim) {
        int idx = row * output_dim + col;
        out[idx] = in[idx] + bias[col];
    }
}

// ============== FEEDFORWARD LAYER ==============

// Feedforward simple (W * input + bias)
// input: [batch_size x input_dim]
// weights: [output_dim x input_dim]
// bias: [output_dim]
// output: [batch_size x output_dim]

void FeedForward(float *input, float *weights, float *bias, 
                         float *output, int batch_size, int input_dim, 
                         int output_dim, bool apply_relu)
{
    float *temp; // Résultat temporaire
    cudaMalloc(&temp, batch_size * output_dim * sizeof(float));
    
    // 1. Multiplication matricielle : output = weights × input
    dim3 threads1(16, 16);
    dim3 blocks1((output_dim + 15) / 16, (batch_size + 15) / 16);
    MatMult<<<blocks1, threads1>>>(input, weights, temp, 
                                batch_size, output_dim, input_dim);
    
    // 2. Addition du bias (broadcast)
    // Besoin d'un kernel pour ajouter bias à chaque ligne
    AddBias<<<blocks1, threads1>>>(temp, bias, temp, batch_size, output_dim);
    
    // 3. ReLU
    if (apply_relu) {
        int threads2 = 256;
        int blocks2 = (batch_size * output_dim + 255) / 256;
        ReLU<<<blocks2, threads2>>>(temp, output, batch_size * output_dim);
    } else {
        cudaMemcpy(output, temp, batch_size * output_dim * sizeof(float), 
                   cudaMemcpyDeviceToDevice);
    }
    
    cudaFree(temp);
}


// ============== MLP FORWARD PASS ==============


// // Structure pour représenter un MLP
// typedef struct {
//     int num_layers;           // Nombre de couches (sans compter l'input)
//     int *layer_sizes;         // Tailles de chaque couche [input_size, hidden1, hidden2, ..., output_size]
//     float **weights;          // Pointeurs vers les poids de chaque couche
//     float **biases;           // Pointeurs vers les biais de chaque couche
// } MLP;

// Forward pass MLP complet
void MLP_Forward(MLP *mlp, float *input, float *output, int batch_size) {
    int num_layers = mlp->num_layers;
    
    // Allocations temporaires pour les activations intermédiaires
    float **activations = (float**)malloc((num_layers + 1) * sizeof(float*));
    
    // La première activation est l'input
    activations[0] = input;
    
    // Allouer la mémoire pour les activations intermédiaires
    for (int i = 1; i <= num_layers; i++) {
        int layer_output_size = mlp->layer_sizes[i];
        cudaMalloc(&activations[i], batch_size * layer_output_size * sizeof(float));
    }
    
    // Forward pass à travers chaque couche
    for (int layer = 0; layer < num_layers; layer++) {
        int input_dim = mlp->layer_sizes[layer];
        int output_dim = mlp->layer_sizes[layer + 1];
        
        // Appliquer ReLU sauf pour la dernière couche (souvent softmax ou linéaire)
        bool apply_relu = (layer < num_layers - 1);
        
        printf("Layer %d: [%d → %d] %s\n", 
               layer + 1, input_dim, output_dim, 
               apply_relu ? "+ ReLU" : "");
        
        FeedForward(activations[layer], 
                           mlp->weights[layer], 
                           mlp->biases[layer],
                           activations[layer + 1],
                           batch_size,
                           input_dim,
                           output_dim,
                           apply_relu);
    }
    
    // Copier le résultat final
    int final_size = mlp->layer_sizes[num_layers];
    cudaMemcpy(output, activations[num_layers], 
               batch_size * final_size * sizeof(float),
               cudaMemcpyDeviceToDevice);
    
    // Libérer les activations intermédiaires (sauf input et output)
    for (int i = 1; i < num_layers; i++) {
        cudaFree(activations[i]);
    }
    
    free(activations);
}

// Fonction helper pour créer un MLP sur GPU
MLP* create_MLP_on_GPU(int *layer_sizes_host, int num_layers) {
    MLP *mlp = (MLP*)malloc(sizeof(MLP));
    mlp->num_layers = num_layers;
    
    // Copier les tailles des couches
    mlp->layer_sizes = (int*)malloc((num_layers + 1) * sizeof(int));
    memcpy(mlp->layer_sizes, layer_sizes_host, (num_layers + 1) * sizeof(int));
    
    // Allouer les tableaux de ptr
    mlp->weights = (float**)malloc(num_layers * sizeof(float*));
    mlp->biases = (float**)malloc(num_layers * sizeof(float*));
    
    return mlp;
}

// Fonction pour libérer un MLP
void free_MLP(MLP *mlp) {
    for (int i = 0; i < mlp->num_layers; i++) {
        cudaFree(mlp->weights[i]);
        cudaFree(mlp->biases[i]);
    }
    free(mlp->weights);
    free(mlp->biases);
    free(mlp->layer_sizes);
    free(mlp);
}