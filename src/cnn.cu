#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cublas_v2.h>
#include "cnn.h"
#include "mlp.h"

// ============== CONVOLUTION 2D (NAIVE) ==============

__global__ void Conv2D_Naive(float *input, float *kernel, float *output, 
                             int batch_size, int in_channels, int out_channels,
                             int in_height, int in_width,
                             int kernel_height, int kernel_width,
                             int stride, int padding)
{
    int out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_width) / stride + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;

    if (idx < total_elements) {
        int w_out = idx % out_width;
        int temp = idx / out_width;
        int h_out = temp % out_height;
        temp = temp / out_height;
        int k = temp % out_channels; 
        int b = temp / out_channels; 

        float sum = 0.0f;

        int h_in_start = h_out * stride - padding;
        int w_in_start = w_out * stride - padding;

        for (int c = 0; c < in_channels; c++) {
            for (int p = 0; p < kernel_height; p++) {
                for (int q = 0; q < kernel_width; q++) {
                    int h_in = h_in_start + p;
                    int w_in = w_in_start + q;
                    if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                        int in_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
                        int k_idx = ((k * in_channels + c) * kernel_height + p) * kernel_width + q;
                        sum += input[in_idx] * kernel[k_idx];
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

void Conv2D(float *input, float *kernel, float *output, 
            int batch_size, int in_channels, int out_channels,
            int in_height, int in_width,
            int kernel_height, int kernel_width,
            int stride, int padding)
{
    int out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_width) / stride + 1;
    int total_elements = batch_size * out_channels * out_height * out_width;

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    Conv2D_Naive<<<blocks, threads>>>(input, kernel, output, 
                                      batch_size, in_channels, out_channels,
                                      in_height, in_width,
                                      kernel_height, kernel_width,
                                      stride, padding);
}

// ============== IM2COL (OPTIMISATION) ==============

// Kernel inspiré de Caffe / Darknet
__global__ void Im2Col_Kernel(float *data_im, float *data_col,
                              int channels, int height, int width,
                              int ksize, int stride, int pad,
                              int height_col, int width_col) 
{
    // Un thread par pixel de la matrice COLONNE résultante
    // Dimensions virtuelles : (channels * ksize * ksize) X (height_col * width_col)
    // On parallélise sur l'ensemble des éléments de la matrice Col
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int num_kernels = channels * ksize * ksize * height_col * width_col;

    if (index < num_kernels) {
        // Calcul des indices
        int w_out = index % width_col;
        int h_out = (index / width_col) % height_col;
        int channel_in = index / (width_col * height_col);
        
        int channel = channel_in / (ksize * ksize);
        int k_y = (channel_in / ksize) % ksize;
        int k_x = channel_in % ksize;

        int h_in = h_out * stride - pad + k_y;
        int w_in = w_out * stride - pad + k_x;

        float *data_im_ptr = data_im + (channel * height + h_in) * width + w_in;
        float *data_col_ptr = data_col + index; // Rappel: index linéaire correspond déjà à la position dans Col

        if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
            *data_col_ptr = *data_im_ptr;
        } else {
            *data_col_ptr = 0;
        }
    }
}

void Conv2D_Im2Col(cublasHandle_t handle, float *input, float *kernel, float *output, 
                   float *col_buffer, // Buffer pré-alloué
                   int batch_size, int in_channels, int out_channels,
                   int in_height, int in_width,
                   int kernel_size, int stride, int padding)
{
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    // Dimensions pour im2col
    // Matrice Col : [ (in_channels * k * k)  x  (out_height * out_width) ]
    int m_col = in_channels * kernel_size * kernel_size;
    int n_col = out_height * out_width;
    
    float alpha = 1.0f;
    float beta = 0.0f;

    // Pour chaque image du batch
    for (int b = 0; b < batch_size; b++) {
        // 1. Im2Col
        // Offset input et output pour le batch courant
        float *d_im = input + b * (in_channels * in_height * in_width);
        float *d_out = output + b * (out_channels * out_height * out_width);
        
        int num_kernels = in_channels * kernel_size * kernel_size * out_height * out_width;
        int threads = 256;
        int blocks = (num_kernels + threads - 1) / threads;
        
        Im2Col_Kernel<<<blocks, threads>>>(d_im, col_buffer,
                                           in_channels, in_height, in_width,
                                           kernel_size, stride, padding,
                                           out_height, out_width);
        
        // 2. GEMM (Matrix Multiplication) via cuBLAS
        // On veut: Output = Weights * Col_Buffer
        // Weights: [out_channels, m_col]
        // Col_Buffer: [m_col, n_col]
        // Output: [out_channels, n_col]
        
        // ATTENTION: cuBLAS est Column-Major, C++ est Row-Major.
        // Si A, B, C sont Row-Major.
        // C = A * B   =>   C^T = B^T * A^T
        // On demande à cuBLAS de calculer: C = B * A (avec des dims inversées)
        // cublasSgemm(handle, OP_N, OP_N, N, M, K, ...)
        
        // M (rows of C) = n_col (pixels)
        // N (cols of C) = out_channels
        // K (common)    = m_col (in_features * k * k)
        // A (Col_Buffer) : [n_col, m_col] en vue transposée
        // B (Kernel)     : [m_col, out_channels] en vue transposée
        
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    n_col,          // m
                    out_channels,   // n
                    m_col,          // k
                    &alpha,
                    col_buffer,     // A
                    n_col,          // lda
                    kernel,         // B
                    m_col,          // ldb
                    &beta,
                    d_out,          // C
                    n_col);         // ldc
    }
}


// ============== MAX POOLING 2D ==============

__global__ void MaxPool2D_Naive(float *input, float *output,
                                int batch_size, int channels,
                                int in_height, int in_width,
                                int pool_size, int stride)
{
    int out_height = (in_height - pool_size) / stride + 1;
    int out_width = (in_width - pool_size) / stride + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_height * out_width;

    if (idx < total_elements) {
        int w_out = idx % out_width;
        int temp = idx / out_width;
        int h_out = temp % out_height;
        temp = temp / out_height;
        int c = temp % channels;
        int b = temp / channels;

        int h_start = h_out * stride;
        int w_start = w_out * stride;

        float max_val = -1e30f;

        for (int p = 0; p < pool_size; p++) {
            for (int q = 0; q < pool_size; q++) {
                int h_in = h_start + p;
                int w_in = w_start + q;

                if (h_in < in_height && w_in < in_width) {
                    int in_idx = ((b * channels + c) * in_height + h_in) * in_width + w_in;
                    float val = input[in_idx];
                    if (val > max_val) max_val = val;
                }
            }
        }
        output[idx] = max_val;
    }
}

void MaxPool2D(float *input, float *output,
               int batch_size, int channels,
               int in_height, int in_width,
               int pool_size, int stride)
{
    int out_height = (in_height - pool_size) / stride + 1;
    int out_width = (in_width - pool_size) / stride + 1;
    int total_elements = batch_size * channels * out_height * out_width;

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    MaxPool2D_Naive<<<blocks, threads>>>(input, output, 
                                         batch_size, channels, 
                                         in_height, in_width, 
                                         pool_size, stride);
}

// ============== ADD BIAS 4D ==============

__global__ void AddBias4D_Kernel(float *input, float *bias, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * height * width;
    
    if (idx < total) {
        int px_per_channel = height * width;
        int px_per_sample = channels * px_per_channel;
        
        int temp = idx % px_per_sample;
        int c = temp / px_per_channel;
        
        input[idx] += bias[c];
    }
}

void AddBias4D(float *input, float *bias, int batch_size, int channels, int height, int width) {
    int total = batch_size * channels * height * width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    AddBias4D_Kernel<<<blocks, threads>>>(input, bias, batch_size, channels, height, width);
}

// ============== FLATTEN ==============

void Flatten(float *input, float *output, int batch_size, int total_elements_per_sample) {
    cudaMemcpy(output, input, batch_size * total_elements_per_sample * sizeof(float), cudaMemcpyDeviceToDevice);
}

// ============== GESTION CNN ==============

CNN* create_CNN(int input_c, int input_h, int input_w, 
                int num_conv_layers, 
                int *conv_configs, 
                int *mlp_layer_sizes, int num_mlp_layers) 
{
    CNN *cnn = (CNN*)malloc(sizeof(CNN));
    cnn->num_conv_layers = num_conv_layers;
    cnn->conv_layers = (ConvLayer*)malloc(num_conv_layers * sizeof(ConvLayer));
    
    // INIT CUBLAS
    cublasCreate(&cnn->cublas_handle);
    
    int current_c = input_c;
    int current_h = input_h;
    int current_w = input_w;
    
    for (int i = 0; i < num_conv_layers; i++) {
        ConvLayer *l = &cnn->conv_layers[i];
        
        int idx = i * 6;
        l->in_channels = current_c;
        l->in_height = current_h;
        l->in_width = current_w;
        l->out_channels = conv_configs[idx + 0];
        l->kernel_size = conv_configs[idx + 1];
        l->stride = conv_configs[idx + 2];
        l->padding = conv_configs[idx + 3];
        l->pool_size = conv_configs[idx + 4];
        l->pool_stride = conv_configs[idx + 5];
        
        l->conv_out_height = (current_h + 2 * l->padding - l->kernel_size) / l->stride + 1;
        l->conv_out_width = (current_w + 2 * l->padding - l->kernel_size) / l->stride + 1;
        
        if (l->pool_size > 1) {
            l->out_height = (l->conv_out_height - l->pool_size) / l->pool_stride + 1;
            l->out_width = (l->conv_out_width - l->pool_size) / l->pool_stride + 1;
        } else {
            l->out_height = l->conv_out_height;
            l->out_width = l->conv_out_width;
        }
        
        cudaMalloc(&l->weights, l->out_channels * l->in_channels * l->kernel_size * l->kernel_size * sizeof(float));
        cudaMalloc(&l->biases, l->out_channels * sizeof(float));
        
        current_c = l->out_channels;
        current_h = l->out_height;
        current_w = l->out_width;
    }
    
    int mlp_input_dim = current_c * current_h * current_w;
    
    cnn->mlp = create_MLP_on_GPU(mlp_layer_sizes, num_mlp_layers);
    
    return cnn;
}

void free_CNN(CNN *cnn) {
    for (int i = 0; i < cnn->num_conv_layers; i++) {
        cudaFree(cnn->conv_layers[i].weights);
        cudaFree(cnn->conv_layers[i].biases);
    }
    free(cnn->conv_layers);
    free_MLP(cnn->mlp);
    cublasDestroy(cnn->cublas_handle);
    free(cnn);
}

// Forward Naif
void CNN_Forward(CNN *cnn, float *input, float *output, int batch_size) {
    float *current_input = input;
    float *layer_output = NULL;
    
    for (int i = 0; i < cnn->num_conv_layers; i++) {
        ConvLayer *l = &cnn->conv_layers[i];
        
        float *conv_out;
        int conv_out_elements = batch_size * l->out_channels * l->conv_out_height * l->conv_out_width;
        cudaMalloc(&conv_out, conv_out_elements * sizeof(float));
        
        Conv2D(current_input, l->weights, conv_out,
               batch_size, l->in_channels, l->out_channels,
               l->in_height, l->in_width,
               l->kernel_size, l->kernel_size, l->stride, l->padding);
               
        AddBias4D(conv_out, l->biases, batch_size, l->out_channels, l->conv_out_height, l->conv_out_width);
        
        int threads = 256;
        int blocks = (conv_out_elements + threads - 1) / threads;
        ReLU<<<blocks, threads>>>(conv_out, conv_out, conv_out_elements);
        
        if (l->pool_size > 1) {
            int pool_out_elements = batch_size * l->out_channels * l->out_height * l->out_width;
            float *pool_out;
            cudaMalloc(&pool_out, pool_out_elements * sizeof(float));
            MaxPool2D(conv_out, pool_out,
                      batch_size, l->out_channels,
                      l->conv_out_height, l->conv_out_width,
                      l->pool_size, l->pool_stride);
            cudaFree(conv_out);
            layer_output = pool_out;
        } else {
            layer_output = conv_out;
        }
        
        if (i > 0) cudaFree(current_input);
        current_input = layer_output;
    }
    MLP_Forward(cnn->mlp, current_input, output, batch_size);
    cudaFree(current_input);
}

// Forward Optimisé (Im2Col)
void CNN_Forward_Im2Col(CNN *cnn, float *input, float *output, int batch_size) {
    float *current_input = input;
    float *layer_output = NULL;
    
    for (int i = 0; i < cnn->num_conv_layers; i++) {
        ConvLayer *l = &cnn->conv_layers[i];
        
        float *conv_out;
        int conv_out_elements = batch_size * l->out_channels * l->conv_out_height * l->conv_out_width;
        cudaMalloc(&conv_out, conv_out_elements * sizeof(float));
        
        // Allocation buffer col pour im2col
        // Taille : [in_c * k * k] * [h_out * w_out]
        int col_size = (l->in_channels * l->kernel_size * l->kernel_size) * (l->conv_out_height * l->conv_out_width);
        float *d_col;
        cudaMalloc(&d_col, col_size * sizeof(float));
        
        // Appel Optimisé
        Conv2D_Im2Col(cnn->cublas_handle, current_input, l->weights, conv_out, d_col,
                      batch_size, l->in_channels, l->out_channels,
                      l->in_height, l->in_width,
                      l->kernel_size, l->stride, l->padding);
        
        cudaFree(d_col); // Libération buffer col
               
        AddBias4D(conv_out, l->biases, batch_size, l->out_channels, l->conv_out_height, l->conv_out_width);
        
        int threads = 256;
        int blocks = (conv_out_elements + threads - 1) / threads;
        ReLU<<<blocks, threads>>>(conv_out, conv_out, conv_out_elements);
        
        if (l->pool_size > 1) {
            int pool_out_elements = batch_size * l->out_channels * l->out_height * l->out_width;
            float *pool_out;
            cudaMalloc(&pool_out, pool_out_elements * sizeof(float));
            MaxPool2D(conv_out, pool_out,
                      batch_size, l->out_channels,
                      l->conv_out_height, l->conv_out_width,
                      l->pool_size, l->pool_stride);
            cudaFree(conv_out);
            layer_output = pool_out;
        } else {
            layer_output = conv_out;
        }
        
        if (i > 0) cudaFree(current_input);
        current_input = layer_output;
    }
    
    MLP_Forward(cnn->mlp, current_input, output, batch_size);
    cudaFree(current_input);
}


// ============== CHARGEMENT FICHIER ==============

CNN* load_CNN_from_file(const char* filename, int input_h, int input_w) {
    FILE *f = fopen(filename, "r");
    if (!f) return NULL;
    
    char tag[16];
    
    std::vector<int> conv_configs; 
    std::vector<std::vector<float>> conv_weights;
    std::vector<std::vector<float>> conv_biases;
    int num_conv = 0;
    int input_c = 0; 
    
    std::vector<int> mlp_sizes;
    std::vector<std::vector<float>> mlp_weights;
    std::vector<std::vector<float>> mlp_biases;
    int num_mlp = 0;
    
    while (fscanf(f, "%s", tag) != EOF) {
        if (std::string(tag) == "CONV") {
            int in_c, out_c, k, s, p, pool_k, pool_s;
            fscanf(f, "%d %d %d %d %d %d %d", &in_c, &out_c, &k, &s, &p, &pool_k, &pool_s);
            
            if (num_conv == 0) input_c = in_c;
            
            conv_configs.push_back(out_c);
            conv_configs.push_back(k);
            conv_configs.push_back(s);
            conv_configs.push_back(p);
            conv_configs.push_back(pool_k);
            conv_configs.push_back(pool_s);
            
            int w_count = out_c * in_c * k * k;
            std::vector<float> w(w_count);
            for(int i=0; i<w_count; i++) fscanf(f, "%f", &w[i]);
            conv_weights.push_back(w);
            
            int b_count = out_c;
            std::vector<float> b(b_count);
            for(int i=0; i<b_count; i++) fscanf(f, "%f", &b[i]);
            conv_biases.push_back(b);
            
            num_conv++;
        }
        else if (std::string(tag) == "FC") {
            fscanf(f, "%d", &num_mlp);
            
            for(int i=0; i<=num_mlp; i++) {
                int size;
                fscanf(f, "%d", &size);
                mlp_sizes.push_back(size);
            }
            
            for(int i=0; i<num_mlp; i++) {
                int input_dim = mlp_sizes[i];
                int output_dim = mlp_sizes[i+1];
                
                int w_count = input_dim * output_dim;
                std::vector<float> w(w_count);
                for(int j=0; j<w_count; j++) fscanf(f, "%f", &w[j]);
                mlp_weights.push_back(w);
                
                int b_count = output_dim;
                std::vector<float> b(b_count);
                for(int j=0; j<b_count; j++) fscanf(f, "%f", &b[j]);
                mlp_biases.push_back(b);
            }
            break;
        }
    }
    
    fclose(f);
    
    CNN *cnn = create_CNN(input_c, input_h, input_w, 
                          num_conv, conv_configs.data(), 
                          mlp_sizes.data(), num_mlp);
                          
    for(int i=0; i<num_conv; i++) {
        ConvLayer *l = &cnn->conv_layers[i];
        cudaMemcpy(l->weights, conv_weights[i].data(), 
                   conv_weights[i].size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(l->biases, conv_biases[i].data(), 
                   conv_biases[i].size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    for(int i=0; i<num_mlp; i++) {
        int w_size = mlp_weights[i].size();
        int b_size = mlp_biases[i].size();
        cudaMalloc(&cnn->mlp->weights[i], w_size * sizeof(float));
        cudaMalloc(&cnn->mlp->biases[i], b_size * sizeof(float));
        cudaMemcpy(cnn->mlp->weights[i], mlp_weights[i].data(), 
                   w_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(cnn->mlp->biases[i], mlp_biases[i].data(), 
                   b_size * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    return cnn;
}