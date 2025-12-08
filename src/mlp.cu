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
        
        //printf("Layer %d: [%d → %d] %s\n", layer + 1, input_dim, output_dim, apply_relu ? "+ ReLU" : "");
        
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


// ============== CHARGEMENT DEPUIS ONNX ==============

MLP* load_MLP_from_onnx(const char* onnx_filename) {
    printf("Chargement du MLP depuis ONNX '%s'...\n", onnx_filename);
    
    // 1. Créer un nom de fichier temporaire pour le .txt
    char txt_filename[512];
    snprintf(txt_filename, sizeof(txt_filename), "%s.tmp.txt", onnx_filename);
    
    // 2. Appeler le script Python pour convertir ONNX → TXT
    char command[1024];
    snprintf(command, sizeof(command), 
             "python3 tools/onnx_to_txt.py \"%s\" \"%s\"", 
             onnx_filename, txt_filename);
    
    printf("   Conversion ONNX → TXT en cours...\n\n");
    int ret = system(command);
    
    if (ret != 0) {
        fprintf(stderr, "\nErreur lors de la conversion ONNX\n");
        fprintf(stderr, "   Vérifiez que :\n");
        fprintf(stderr, "   - Python 3 est installé\n");
        fprintf(stderr, "   - Le package 'onnx' est installé: pip install onnx\n");
        fprintf(stderr, "   - Le fichier tools/onnx_to_txt.py existe\n");
        return NULL;
    }
    
    printf("\n");
    
    // 3. Charger depuis le fichier texte
    MLP *mlp = load_MLP_from_file(txt_filename);
    
    // 4. Supprimer le fichier temporaire
    if (mlp != NULL) {
        remove(txt_filename);
        printf("MLP chargé depuis ONNX avec succès!\n\n");
    }
    
    return mlp;
}


// ============== OPTIMISATION 1 : KERNEL FUSIONNÉ ==============

__global__ void FeedForward_Fused_Kernel(float *input, float *weights, float *bias,
                                          float *output, int batch_size, int input_dim,
                                          int output_dim, bool apply_relu)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // output neuron index
    
    if (row < batch_size && col < output_dim) {
        float value = 0.0f;
        
        // W * input (dot product)
        for (int k = 0; k < input_dim; k++) {
            value += weights[k * output_dim + col] * input[row * input_dim + k];
        }
        
        // + bias
        value += bias[col];
        
        // ReLU
        if (apply_relu) {
            value = fmaxf(0.0f, value);
        }
        
        output[row * output_dim + col] = value;
    }
}

void FeedForward_Fused(float *input, float *weights, float *bias,
                       float *output, int batch_size, int input_dim,
                       int output_dim, bool apply_relu)
{
    dim3 threads(16, 16);
    dim3 blocks((output_dim + 15) / 16, (batch_size + 15) / 16);
    
    FeedForward_Fused_Kernel<<<blocks, threads>>>(input, weights, bias, output,
                                                    batch_size, input_dim, output_dim,
                                                    apply_relu);
}

// ============== OPTIMISATION 2 : cuBLAS ==============

void FeedForward_cuBLAS(cublasHandle_t handle, float *input, float *weights,
                        float *bias, float *output, int batch_size,
                        int input_dim, int output_dim, bool apply_relu)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // cuBLAS: C = alpha * A * B + beta * C
    // output [batch_size x output_dim] = input [batch_size x input_dim] * weights^T [input_dim x output_dim]
    // En notation cuBLAS (column-major): output = weights * input
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                output_dim, batch_size, input_dim,
                &alpha,
                weights, output_dim,
                input, input_dim,
                &beta,
                output, output_dim);
    
    // Ajouter le biais
    dim3 threads(16, 16);
    dim3 blocks((output_dim + 15) / 16, (batch_size + 15) / 16);
    AddBias<<<blocks, threads>>>(output, bias, output, batch_size, output_dim);
    
    // ReLU
    if (apply_relu) {
        int threads_relu = 256;
        int blocks_relu = (batch_size * output_dim + 255) / 256;
        ReLU<<<blocks_relu, threads_relu>>>(output, output, batch_size * output_dim);
    }
}

// ============== OPTIMISATION 3 : SHARED MEMORY TILED ==============

__global__ void MatMult_Tiled(float *A, float *B, float *C, int n, int m, int common)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float value = 0.0f;
    
    // Boucle sur les tuiles
    for (int t = 0; t < (common + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Charger la tuile de A dans la shared memory
        if (row < n && (t * TILE_SIZE + threadIdx.x) < common)
            As[threadIdx.y][threadIdx.x] = A[row * common + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        
        // Charger la tuile de B dans la shared memory
        if ((t * TILE_SIZE + threadIdx.y) < common && col < m)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * m + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // Calculer le produit partiel
        for (int k = 0; k < TILE_SIZE; k++)
            value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        
        __syncthreads();
    }
    
    if (row < n && col < m)
        C[row * m + col] = value;
}

void FeedForward_Tiled(float *input, float *weights, float *bias,
                       float *output, int batch_size, int input_dim,
                       int output_dim, bool apply_relu)
{
    float *temp;
    cudaMalloc(&temp, batch_size * output_dim * sizeof(float));
    
    // Multiplication matricielle avec tiling
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((output_dim + TILE_SIZE - 1) / TILE_SIZE,
                (batch_size + TILE_SIZE - 1) / TILE_SIZE);
    
    MatMult_Tiled<<<blocks, threads>>>(input, weights, temp,
                                       batch_size, output_dim, input_dim);
    
    // Ajouter le biais
    dim3 threads_bias(16, 16);
    dim3 blocks_bias((output_dim + 15) / 16, (batch_size + 15) / 16);
    AddBias<<<blocks_bias, threads_bias>>>(temp, bias, temp, batch_size, output_dim);
    
    // ReLU
    if (apply_relu) {
        int threads_relu = 256;
        int blocks_relu = (batch_size * output_dim + 255) / 256;
        ReLU<<<blocks_relu, threads_relu>>>(temp, output, batch_size * output_dim);
    } else {
        cudaMemcpy(output, temp, batch_size * output_dim * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }
    
    cudaFree(temp);
}

// ============== OPTIMISATION 4 : KERNEL FUSED & TILED ==============

__global__ void FeedForward_Fused_Tiled_Kernel(float *input, float *weights, float *bias,
                                               float *output, int batch_size, int input_dim,
                                               int output_dim, bool apply_relu)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // Index de ligne (Batch)
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // Index de colonne (Neurone)
    
    float value = 0.0f; // Accumulateur pour le produit XW[row, col]
    
    // Boucle sur les tuiles (Tiling)
    for (int t = 0; t < (input_dim + TILE_SIZE - 1) / TILE_SIZE; t++) {
        
        // Charger la tuile de A (Input) dans la shared memory (A est [Batch x Input])
        int tile_A_idx = t * TILE_SIZE + threadIdx.x;
        if (row < batch_size && tile_A_idx < input_dim)
            As[threadIdx.y][threadIdx.x] = input[row * input_dim + tile_A_idx];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f; // Padding avec zéro
        
        // Charger la tuile de B (Weights) dans la shared memory (B est [Input x Output])
        int tile_B_idx = t * TILE_SIZE + threadIdx.y;
        if (tile_B_idx < input_dim && col < output_dim)

            Bs[threadIdx.y][threadIdx.x] = weights[tile_B_idx * output_dim + col]; 
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f; // Padding avec zéro
        
        __syncthreads(); // Attente que toutes les données soient chargées
        
        // Calculer le produit partiel 
        for (int k = 0; k < TILE_SIZE; k++)
            value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        
        __syncthreads(); 
    }
    
    // FUSION ET ECRITURE FINALE
    if (row < batch_size && col < output_dim) {
        
        // Ajout du Biais
        value += bias[col]; // value = XW + bias
        
        // ctivation (ReLU)
        if (apply_relu) {
            value = fmaxf(0.0f, value);
        }
        
        // Écriture du résultat
        output[row * output_dim + col] = value;
    }
}


void FeedForward_Fused_Tiled(float *input, float *weights, float *bias,
                             float *output, int batch_size, int input_dim,
                             int output_dim, bool apply_relu)
{
    // Configuration de la grille
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((output_dim + TILE_SIZE - 1) / TILE_SIZE,
                (batch_size + TILE_SIZE - 1) / TILE_SIZE);
    
    // Lancement du kernel combiné
    FeedForward_Fused_Tiled_Kernel<<<blocks, threads>>>(input, weights, bias, output,
                                                         batch_size, input_dim, output_dim,
                                                         apply_relu);
}

// ============== OPTIMISATION 5 : MLP OPTIMISÉ ==============

MLP_Optimized* create_MLP_Optimized(int *layer_sizes_host, int num_layers, int batch_size)
{
    MLP_Optimized *mlp = (MLP_Optimized*)malloc(sizeof(MLP_Optimized));
    mlp->num_layers = num_layers;
    mlp->batch_size = batch_size;
    
    mlp->layer_sizes = (int*)malloc((num_layers + 1) * sizeof(int));
    memcpy(mlp->layer_sizes, layer_sizes_host, (num_layers + 1) * sizeof(int));
    
    mlp->weights = (float**)malloc(num_layers * sizeof(float*));
    mlp->biases = (float**)malloc(num_layers * sizeof(float*));
    
    // Pré-allouer les buffers d'activation
    mlp->activation_buffers = (float**)malloc((num_layers + 1) * sizeof(float*));
    for (int i = 0; i <= num_layers; i++) {
        cudaMalloc(&mlp->activation_buffers[i],
                   batch_size * layer_sizes_host[i] * sizeof(float));
    }
    
    // Créer le handle cuBLAS
    cublasCreate(&mlp->cublas_handle);
    
    return mlp;
}

void free_MLP_Optimized(MLP_Optimized *mlp)
{
    for (int i = 0; i < mlp->num_layers; i++) {
        cudaFree(mlp->weights[i]);
        cudaFree(mlp->biases[i]);
    }
    
    for (int i = 0; i <= mlp->num_layers; i++) {
        cudaFree(mlp->activation_buffers[i]);
    }
    
    cublasDestroy(mlp->cublas_handle);
    
    free(mlp->weights);
    free(mlp->biases);
    free(mlp->activation_buffers);
    free(mlp->layer_sizes);
    free(mlp);
}

void MLP_Forward_Optimized(MLP_Optimized *mlp, float *input, float *output)
{
    int batch_size = mlp->batch_size;
    
    // Copier l'input dans le premier buffer
    cudaMemcpy(mlp->activation_buffers[0], input,
               batch_size * mlp->layer_sizes[0] * sizeof(float),
               cudaMemcpyDeviceToDevice);
    
    // Forward pass avec kernel fusionné
    for (int layer = 0; layer < mlp->num_layers; layer++) {
        int input_dim = mlp->layer_sizes[layer];
        int output_dim = mlp->layer_sizes[layer + 1];
        bool apply_relu = (layer < mlp->num_layers - 1);
        
        FeedForward_Fused(mlp->activation_buffers[layer],
                         mlp->weights[layer],
                         mlp->biases[layer],
                         mlp->activation_buffers[layer + 1],
                         batch_size, input_dim, output_dim, apply_relu);
    }
    
    // Copier le résultat final
    int final_size = mlp->layer_sizes[mlp->num_layers];
    cudaMemcpy(output, mlp->activation_buffers[mlp->num_layers],
               batch_size * final_size * sizeof(float),
               cudaMemcpyDeviceToDevice);
}

void MLP_Forward_Optimized_cuBLAS(MLP_Optimized *mlp, float *input, float *output)
{
    int batch_size = mlp->batch_size;
    
    cudaMemcpy(mlp->activation_buffers[0], input,
               batch_size * mlp->layer_sizes[0] * sizeof(float),
               cudaMemcpyDeviceToDevice);
    
    // Forward pass avec cuBLAS
    for (int layer = 0; layer < mlp->num_layers; layer++) {
        int input_dim = mlp->layer_sizes[layer];
        int output_dim = mlp->layer_sizes[layer + 1];
        bool apply_relu = (layer < mlp->num_layers - 1);
        
        FeedForward_cuBLAS(mlp->cublas_handle,
                          mlp->activation_buffers[layer],
                          mlp->weights[layer],
                          mlp->biases[layer],
                          mlp->activation_buffers[layer + 1],
                          batch_size, input_dim, output_dim, apply_relu);
    }
    
    int final_size = mlp->layer_sizes[mlp->num_layers];
    cudaMemcpy(output, mlp->activation_buffers[mlp->num_layers],
               batch_size * final_size * sizeof(float),
               cudaMemcpyDeviceToDevice);
}

void MLP_Forward_Optimized_Fused_Tiled(MLP_Optimized *mlp, float *input, float *output)
{
    int batch_size = mlp->batch_size;
    
    cudaMemcpy(mlp->activation_buffers[0], input,
               batch_size * mlp->layer_sizes[0] * sizeof(float),
               cudaMemcpyDeviceToDevice);
    
    // Forward pass avec kernel fusionné et tiling
    for (int layer = 0; layer < mlp->num_layers; layer++) {
        int input_dim = mlp->layer_sizes[layer];
        int output_dim = mlp->layer_sizes[layer + 1];
        bool apply_relu = (layer < mlp->num_layers - 1);
        
        FeedForward_Fused_Tiled(mlp->activation_buffers[layer],
                                mlp->weights[layer],
                                mlp->biases[layer],
                                mlp->activation_buffers[layer + 1],
                                batch_size, input_dim, output_dim, apply_relu);
    }
    
    int final_size = mlp->layer_sizes[mlp->num_layers];
    cudaMemcpy(output, mlp->activation_buffers[mlp->num_layers],
               batch_size * final_size * sizeof(float),
               cudaMemcpyDeviceToDevice);
}
