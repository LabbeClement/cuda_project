#include <stdio.h>
#include <cuda_runtime.h>
#include "../../src/mlp.h"

// Macro pour vérifier les erreurs CUDA
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (err_code=%d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Helper : Wrapper pour la version Tiled (qui n'a pas de fonction MLP_Forward propre dans mlp.h)
void MLP_Forward_Tiled_Wrapper(MLP *mlp, float *input, float *output, int batch_size) {
    int num_layers = mlp->num_layers;
    
    // Allocation temporaire (comme dans la version naïve, mais appelle le kernel Tiled)
    float **activations = (float**)malloc((num_layers + 1) * sizeof(float*));
    activations[0] = input;
    
    for (int i = 1; i <= num_layers; i++) {
        int layer_output_size = mlp->layer_sizes[i];
        CHECK_CUDA(cudaMalloc(&activations[i], batch_size * layer_output_size * sizeof(float)));
    }
    
    for (int layer = 0; layer < num_layers; layer++) {
        int input_dim = mlp->layer_sizes[layer];
        int output_dim = mlp->layer_sizes[layer + 1];
        bool apply_relu = (layer < num_layers - 1);
        
        // Appel de la version Tiled définie dans mlp.h
        FeedForward_Tiled(activations[layer], 
                         mlp->weights[layer], 
                         mlp->biases[layer],
                         activations[layer + 1],
                         batch_size, input_dim, output_dim, apply_relu);
    }
    
    // Copie résultat final
    int final_size = mlp->layer_sizes[num_layers];
    CHECK_CUDA(cudaMemcpy(output, activations[num_layers], 
               batch_size * final_size * sizeof(float),
               cudaMemcpyDeviceToDevice));
    
    for (int i = 1; i < num_layers; i++) {
        CHECK_CUDA(cudaFree(activations[i]));
    }
    free(activations);
}

// Helper : Convertit un MLP standard (chargé depuis fichier) vers MLP_Optimized
MLP_Optimized* convert_to_optimized(MLP *src, int batch_size) {
    // 1. Créer la structure optimisée (alloue les buffers d'activation)
    MLP_Optimized *opt = create_MLP_Optimized(src->layer_sizes, src->num_layers, batch_size);
    
    // 2. Allouer et Copier les poids et biais
    for (int i = 0; i < src->num_layers; i++) {
        int input_dim = src->layer_sizes[i];
        int output_dim = src->layer_sizes[i+1];
        
        // --- CORRECTION : Allocation mémoire GPU pour les poids et biais ---
        CHECK_CUDA(cudaMalloc(&opt->weights[i], input_dim * output_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&opt->biases[i], output_dim * sizeof(float)));
        // -----------------------------------------------------------------

        // Copie (Device to Device)
        CHECK_CUDA(cudaMemcpy(opt->weights[i], src->weights[i], 
                   input_dim * output_dim * sizeof(float), cudaMemcpyDeviceToDevice));
                   
        CHECK_CUDA(cudaMemcpy(opt->biases[i], src->biases[i], 
                   output_dim * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    return opt;
}

// Fonction générique de mesure de temps
template <typename T>
float benchmark_op(const char* name, void (*func)(T*, float*, float*), 
                  T* model, float* d_input, float* d_output, int iterations) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Warm-up
    func(model, d_input, d_output);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start));
    for(int i=0; i<iterations; i++) {
        func(model, d_input, d_output);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / iterations;
}

// Wrapper pour adapter la signature de MLP_Forward (qui prend batch_size) au template
void wrapper_std_forward(MLP* m, float* in, float* out) {
    // On assume batch_size fixe défini ailleurs ou on triche un peu pour le bench
    // Ici on ne peut pas passer batch_size facilement sans changer la signature du template
    // Une astuce simple : on passe 0 ou une valeur dummy si la fct le demande, 
    // MAIS MLP_Forward demande batch_size.
    // Pour simplifier, on va faire un benchmark manuel dans la boucle principale.
}

void run_benchmark_on_file(const char* filename, int batch_size) {
    printf("\n==========================================================\n");
    printf("BENCHMARKING MODEL: %s\n", filename);
    printf("Batch Size: %d\n", batch_size);
    
    // 1. Chargement du modèle
    MLP *mlp = load_MLP_from_onnx(filename);
    if (!mlp) {
        printf("Skipping %s (failed to load)\n", filename);
        return;
    }

    // Affichage info couches
    printf("Architecture: [");
    for(int i=0; i<=mlp->num_layers; i++) printf("%d ", mlp->layer_sizes[i]);
    printf("]\n");

    // 2. Préparation Données (Aléatoires pour le bench de perf)
    int input_dim = mlp->layer_sizes[0];
    int output_dim = mlp->layer_sizes[mlp->num_layers];
    
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, batch_size * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, batch_size * output_dim * sizeof(float)));
    
    // Remplir input avec des données dummy
    CHECK_CUDA(cudaMemset(d_input, 0x11, batch_size * input_dim * sizeof(float)));

    // 3. Conversion vers version optimisée
    MLP_Optimized *mlp_opt = convert_to_optimized(mlp, batch_size);

    // --- MESURES ---
    const int ITER = 20;
    float time_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    printf("\nRunning %d iterations per version...\n", ITER);
    printf("%-25s | %-15s\n", "Version", "Avg Time (ms)");
    printf("--------------------------|----------------\n");

    // 1. Baseline (Modular)
    cudaEventRecord(start);
    for(int i=0; i<ITER; i++) MLP_Forward(mlp, d_input, d_output, batch_size);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("%-25s | %.4f ms\n", "1. Modular (Baseline)", time_ms/ITER);

    // 2. Tiled
    cudaEventRecord(start);
    for(int i=0; i<ITER; i++) MLP_Forward_Tiled_Wrapper(mlp, d_input, d_output, batch_size);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("%-25s | %.4f ms\n", "2. Shared Mem Tiled", time_ms/ITER);

    // 3. Fused + Pre-alloc
    cudaEventRecord(start);
    for(int i=0; i<ITER; i++) MLP_Forward_Optimized(mlp_opt, d_input, d_output);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("%-25s | %.4f ms\n", "3. Fused + Pre-alloc", time_ms/ITER);

    // 4. cuBLAS
    cudaEventRecord(start);
    for(int i=0; i<ITER; i++) MLP_Forward_Optimized_cuBLAS(mlp_opt, d_input, d_output);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("%-25s | %.4f ms\n", "4. cuBLAS", time_ms/ITER);

    // 5. Fused + Tiled
    cudaEventRecord(start);
    for(int i=0; i<ITER; i++) MLP_Forward_Optimized_Fused_Tiled(mlp_opt, d_input, d_output);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("%-25s | %.4f ms\n", "5. Fulted", time_ms/ITER);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free_MLP(mlp);
    free_MLP_Optimized(mlp_opt);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char **argv) {
    // Si des fichiers sont passés en argument, les utiliser, sinon utiliser les défauts
    const char* files[] = {
        "tests/data/mlp_model.onnx",
        "tests/data/mlp_model_large.onnx"
    };
    int num_files = 2;

    int batch_size = 4096; // Batch size par défaut conséquent pour voir les différences GPU

    for (int i = 0; i < num_files; i++) {
        run_benchmark_on_file(files[i], batch_size);
    }
    
    // Test avec un petit batch size pour voir la latence
    printf("\n\n>>> TEST AVEC PETIT BATCH SIZE (64) <<<\n");
    run_benchmark_on_file("tests/data/mlp_model.onnx", 64);

    return 0;
}