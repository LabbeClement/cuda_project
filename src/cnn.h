#ifndef CNN_H
#define CNN_H

#include <cublas_v2.h>
#include "mlp.h" // Nécessaire pour la structure MLP

// ============== PRIMITIVES CNN ==============

// 1. Convolution 2D (Naive)
__global__ void Conv2D_Naive(float *input, float *kernel, float *output, 
                             int batch_size, int in_channels, int out_channels,
                             int in_height, int in_width,
                             int kernel_height, int kernel_width,
                             int stride, int padding);

void Conv2D(float *input, float *kernel, float *output, 
            int batch_size, int in_channels, int out_channels,
            int in_height, int in_width,
            int kernel_height, int kernel_width,
            int stride, int padding);

// 2. Max Pooling 2D
__global__ void MaxPool2D_Naive(float *input, float *output,
                                int batch_size, int channels,
                                int in_height, int in_width,
                                int pool_size, int stride);

void MaxPool2D(float *input, float *output,
               int batch_size, int channels,
               int in_height, int in_width,
               int pool_size, int stride);

// 3. Add Bias 4D
__global__ void AddBias4D_Kernel(float *input, float *bias, int batch_size, int channels, int height, int width);
void AddBias4D(float *input, float *bias, int batch_size, int channels, int height, int width);

// 4. Flatten
void Flatten(float *input, float *output, int batch_size, int total_elements_per_sample);

// 5. Im2Col (Optimisation)
__global__ void Im2Col_Kernel(float *data_im, float *data_col,
                              int channels, int height, int width,
                              int ksize, int stride, int pad,
                              int height_col, int width_col);

void Conv2D_Im2Col(cublasHandle_t handle, float *input, float *kernel, float *output, 
                   float *col_buffer, // Buffer temporaire [in_c*k*k, h_out*w_out]
                   int batch_size, int in_channels, int out_channels,
                   int in_height, int in_width,
                   int kernel_size, int stride, int padding);

// ============== STRUCTURES ==============

// Structure représentant une couche de convolution
typedef struct {
    int in_channels;
    int out_channels;
    int in_height;
    int in_width;
    int kernel_size;
    int stride;
    int padding;
    
    // Pooling (si pool_size > 1)
    int pool_size;
    int pool_stride;
    
    // Dimensions de sortie
    int conv_out_height;
    int conv_out_width;
    int out_height; // Final output dimensions (après pool)
    int out_width;
    
    // Paramètres GPU
    float *weights; // [out_channels, in_channels, kernel_size, kernel_size]
    float *biases;  // [out_channels]
} ConvLayer;

// Structure du réseau complet
typedef struct {
    int num_conv_layers;
    ConvLayer *conv_layers; // Tableau de couches
    MLP *mlp;               // Le MLP classifieur à la fin
    cublasHandle_t cublas_handle; // Handle pour cuBLAS
} CNN;

// ============== GESTION ==============

CNN* create_CNN(int input_c, int input_h, int input_w, 
                int num_conv_layers, 
                int *conv_configs, 
                int *mlp_layer_sizes, int num_mlp_layers);

void free_CNN(CNN *cnn);

// Version Naïve
void CNN_Forward(CNN *cnn, float *input, float *output, int batch_size);

// Version Optimisée (Im2Col + cuBLAS)
void CNN_Forward_Im2Col(CNN *cnn, float *input, float *output, int batch_size);

CNN* load_CNN_from_file(const char* filename, int input_h, int input_w);

#endif // CNN_H