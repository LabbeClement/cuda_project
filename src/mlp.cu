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

// ============== CHARGEMENT/SAUVEGARDE MLP ==============

// Charger un MLP depuis un fichier texte
MLP* load_MLP_from_file(const char* filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Erreur: impossible d'ouvrir le fichier '%s'\n", filename);
        return NULL;
    }
    
    printf("Chargement du MLP depuis '%s'\n", filename);
    
    // 1. Lire le nombre de couches
    int num_layers;
    if (fscanf(f, "%d", &num_layers) != 1) {
        fprintf(stderr, "Erreur: lecture du nombre de couches\n");
        fclose(f);
        return NULL;
    }
    
    // 2. Lire l'architecture (layer_sizes)
    int *layer_sizes = (int*)malloc((num_layers + 1) * sizeof(int));
    printf("   Architecture: [");
    for (int i = 0; i <= num_layers; i++) {
        if (fscanf(f, "%d", &layer_sizes[i]) != 1) {
            fprintf(stderr, "Erreur: lecture de l'architecture\n");
            free(layer_sizes);
            fclose(f);
            return NULL;
        }
        printf("%d", layer_sizes[i]);
        if (i < num_layers) printf(" → ");
    }
    printf("]\n");
    
    // 3. Créer la structure MLP
    MLP *mlp = create_MLP_on_GPU(layer_sizes, num_layers);
    
    // 4. Charger les poids et biais de chaque couche
    for (int layer = 0; layer < num_layers; layer++) {
        int input_dim = layer_sizes[layer];
        int output_dim = layer_sizes[layer + 1];
        
        printf("   Layer %d: [%d × %d] weights + [%d] bias\n", 
               layer + 1, input_dim, output_dim, output_dim);
        
        // Charger les weights [input_dim × output_dim]
        float *h_weights = (float*)malloc(input_dim * output_dim * sizeof(float));
        for (int i = 0; i < input_dim * output_dim; i++) {
            if (fscanf(f, "%f", &h_weights[i]) != 1) {
                fprintf(stderr, "Erreur: lecture des poids (layer %d)\n", layer);
                free(h_weights);
                free_MLP(mlp);
                fclose(f);
                return NULL;
            }
        }
        
        // Copier sur le GPU
        cudaMalloc(&mlp->weights[layer], input_dim * output_dim * sizeof(float));
        cudaMemcpy(mlp->weights[layer], h_weights, 
                   input_dim * output_dim * sizeof(float), 
                   cudaMemcpyHostToDevice);
        free(h_weights);
        
        // Charger les bias [output_dim]
        float *h_bias = (float*)malloc(output_dim * sizeof(float));
        for (int i = 0; i < output_dim; i++) {
            if (fscanf(f, "%f", &h_bias[i]) != 1) {
                fprintf(stderr, "Erreur: lecture du biais (layer %d)\n", layer);
                free(h_bias);
                free_MLP(mlp);
                fclose(f);
                return NULL;
            }
        }
        
        // Copier sur le GPU
        cudaMalloc(&mlp->biases[layer], output_dim * sizeof(float));
        cudaMemcpy(mlp->biases[layer], h_bias, 
                   output_dim * sizeof(float), 
                   cudaMemcpyHostToDevice);
        free(h_bias);
    }
    
    fclose(f);
    free(layer_sizes);
    
    printf("MLP chargé !\n\n");
    return mlp;
}

// Sauvegarder un MLP dans un fichier texte
void save_MLP_to_file(MLP *mlp, const char* filename, int batch_size) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Erreur: impossible de créer le fichier '%s'\n", filename);
        return;
    }
    
    printf("Sauvegarde du MLP dans '%s'\n", filename);
    
    // 1. Écrire le nombre de couches
    fprintf(f, "%d\n", mlp->num_layers);
    
    // 2. Écrire l'architecture
    for (int i = 0; i <= mlp->num_layers; i++) {
        fprintf(f, "%d ", mlp->layer_sizes[i]);
    }
    fprintf(f, "\n\n");
    
    // 3. Écrire les poids et biais de chaque couche
    for (int layer = 0; layer < mlp->num_layers; layer++) {
        int input_dim = mlp->layer_sizes[layer];
        int output_dim = mlp->layer_sizes[layer + 1];
        
        fprintf(f, "# Layer %d weights [%d x %d]\n", layer, input_dim, output_dim);
        
        // Copier les weights depuis le GPU
        float *h_weights = (float*)malloc(input_dim * output_dim * sizeof(float));
        cudaMemcpy(h_weights, mlp->weights[layer], 
                   input_dim * output_dim * sizeof(float), 
                   cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < input_dim; i++) {
            for (int j = 0; j < output_dim; j++) {
                fprintf(f, "%.6f ", h_weights[i * output_dim + j]);
            }
            fprintf(f, "\n");
        }
        free(h_weights);
        
        fprintf(f, "\n# Layer %d bias [%d]\n", layer, output_dim);
        
        // Copier les bias depuis le GPU
        float *h_bias = (float*)malloc(output_dim * sizeof(float));
        cudaMemcpy(h_bias, mlp->biases[layer], 
                   output_dim * sizeof(float), 
                   cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < output_dim; i++) {
            fprintf(f, "%.6f ", h_bias[i]);
        }
        fprintf(f, "\n\n");
        free(h_bias);
    }
    
    fclose(f);
    printf("MLP sauvegardé !\n");
}