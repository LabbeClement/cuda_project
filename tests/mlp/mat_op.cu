#include <stdio.h>
#include "../../src/mlp.h"

const int N = 5; 
const int M = 3;

int test_matrix_add() {
    float a[N][M] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {10, 11, 12},
        {13, 14, 15}
    };
    
    float b[N][M] = {
        {10, 20, 30},
        {40, 50, 60},
        {70, 80, 90},
        {100, 110, 120},
        {130, 140, 150}
    };
    
    float c[N][M];

    float *GPU_a, *GPU_b, *GPU_c;

    // Allocation mémoire sur le GPU
    cudaMalloc((void**)&GPU_a, N*M*sizeof(float));
    cudaMalloc((void**)&GPU_b, N*M*sizeof(float));
    cudaMalloc((void**)&GPU_c, N*M*sizeof(float));
    
    // Copie des données vers le GPU pour les calculs car GPU ne supporte que vecteurs 1D
    cudaMemcpy(GPU_a, a, N*M*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(GPU_b, b, N*M*sizeof(float), cudaMemcpyHostToDevice);
    
    // Configuration des threads
    dim3 threadsPerBlock(16, 16);  // 16x16 threads par bloc
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Lancement du kernel
    // Ici 1 block de (16,16) threads
    MatAdd<<<numBlocks, threadsPerBlock>>>(GPU_a, GPU_b, GPU_c, N, M);
    
    // Copie des résultats vers le CPU pour print
    cudaMemcpy(c, GPU_c, N*M*sizeof(float), cudaMemcpyDeviceToHost);
    
    // Affichage de la matrice résultat
    printf("Matrice C (résultat) :\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%.0f\t", c[i][j]);
        }
        printf("\n");
    }
    
    // Libération de la mémoire
    cudaFree(GPU_a);
    cudaFree(GPU_b);
    cudaFree(GPU_c);


    return 0;
}

int test_matrix_mult() {
    // Matrices d'exemple
    float a[N][M] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {10, 11, 12},
        {13, 14, 15}
    };
    
    float b[M][N] = {
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15}
    };
    
    float c[N][N];
    
    float *GPU_a, *d_b, *d_c;
    
    // Allocation mémoire sur le GPU
    cudaMalloc((void**)&GPU_a, N*M*sizeof(float));
    cudaMalloc((void**)&d_b, M*N*sizeof(float));
    cudaMalloc((void**)&d_c, N*N*sizeof(float));
    
    // Copie des données vers le GPU
    cudaMemcpy(GPU_a, a, N*M*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, M*N*sizeof(float), cudaMemcpyHostToDevice);
    
    // Configuration des threads
    dim3 threadsPerBlock(16, 16);  // 16x16 threads par bloc
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Lancement du kernel
    MatMult<<<numBlocks, threadsPerBlock>>>(GPU_a, d_b, d_c, N, N, M);
    
    // Copie des résultats vers le CPU
    cudaMemcpy(c, d_c, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    
    // Affichage de la matrice résultat
    printf("Matrice C (résultat) :\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.0f\t", c[i][j]);
        }
        printf("\n");
    }
    // Libération de la mémoire
    cudaFree(GPU_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}


const int N_BENCH = 4096;
const int M_BENCH = 4096;
const int K_BENCH = 4096;

int test_matrix_add_benchmark()
{
    printf("\n--- BENCHMARK CUDA : MatAdd (%d x %d) ---\n", N_BENCH, M_BENCH);

    // 1. Déclarer les chronomètres GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 2. Préparation (Allocation GPU)
    float *GPU_a, *GPU_b, *GPU_c;
    cudaMalloc((void**)&GPU_a, (size_t)N_BENCH * M_BENCH * sizeof(float));
    cudaMalloc((void**)&GPU_b, (size_t)N_BENCH * M_BENCH * sizeof(float));
    cudaMalloc((void**)&GPU_c, (size_t)N_BENCH * M_BENCH * sizeof(float));

    // 3. Configuration du Kernel
    dim3 threadsPerBlock(16, 16); 
    dim3 numBlocks((M_BENCH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N_BENCH + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 4. Warm-up (Chauffage GPU)
    MatAdd<<<numBlocks, threadsPerBlock>>>(GPU_a, GPU_b, GPU_c, N_BENCH, M_BENCH);

    // 5. Mesure
    cudaEventRecord(start); // <-- DÉPART CHRONO GPU

    // Exécution de l'opération
    MatAdd<<<numBlocks, threadsPerBlock>>>(GPU_a, GPU_b, GPU_c, N_BENCH, M_BENCH);

    cudaEventRecord(stop);  // <-- FIN CHRONO GPU
    cudaEventSynchronize(stop); // Attend que le GPU ait vraiment fini

    // 6. Calculer le temps écoulé
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Temps CUDA (MatAdd naïf): %.4f ms\n", milliseconds);

    // 7. Nettoyage
    cudaFree(GPU_a); cudaFree(GPU_b); cudaFree(GPU_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}

int main() {
    test_matrix_add();
    test_matrix_mult();
    test_matrix_add_benchmark();
    return 0;
}