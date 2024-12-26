#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include "smith-waterman.h"

#define S_LEN 512  // Length of sequences
#define N 1000     // Number of sequences
#define MATCH 1    // Score for a match
#define MISMATCH -1 // Score for a mismatch
#define INS -2     // Score for an insertion
#define DEL -2     // Score for a deletion

// Function to get the current time (implementation not shown here)

// Main function
int main() {
    srand(time(NULL));  // Seed the random number generator
    char alphabet[5] = {'A', 'C', 'G', 'T', 'N'};  // DNA alphabet

    char *query, *reference;
    int *res;

    // Allocate pinned host memory
    cudaMallocHost(&query, N * S_LEN * sizeof(char));
    cudaMallocHost(&reference, N * S_LEN * sizeof(char));
    cudaMallocHost(&res, N * sizeof(int));

    // Initialize sequences with random characters
    for (int i = 0; i < N * S_LEN; i++) {
        query[i] = alphabet[rand() % 5];
        reference[i] = alphabet[rand() % 5];
    }

    char *d_query, *d_reference;
    int *d_res;

    // Allocate device memory
    cudaMalloc(&d_query, N * S_LEN * sizeof(char));
    cudaMalloc(&d_reference, N * S_LEN * sizeof(char));
    cudaMalloc(&d_res, N * sizeof(int));

    // Copy sequences to device memory
    cudaMemcpyAsync(d_query, query, N * S_LEN * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_reference, reference, N * S_LEN * sizeof(char), cudaMemcpyHostToDevice);

    // Initialize cuDNN
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Create tensor descriptors
    cudnnTensorDescriptor_t query_desc, reference_desc, res_desc;
    cudnnCreateTensorDescriptor(&query_desc);
    cudnnCreateTensorDescriptor(&reference_desc);
    cudnnCreateTensorDescriptor(&res_desc);

    // Set tensor descriptors
    cudnnSetTensor4dDescriptor(query_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, 1, S_LEN, 1);
    cudnnSetTensor4dDescriptor(reference_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, 1, S_LEN, 1);
    cudnnSetTensor4dDescriptor(res_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, 1, S_LEN, 1);

    // Create convolution descriptor
    cudnnFilterDescriptor_t filter_desc;
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, N, 1, S_LEN, 1);

    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // Get convolution algorithm
    cudnnConvolutionFwdAlgoPerf_t algo_perf;
    int returned_algo_count;
    cudnnFindConvolutionForwardAlgorithm(cudnn, query_desc, filter_desc, conv_desc, res_desc, 1, &returned_algo_count, &algo_perf);
    cudnnConvolutionFwdAlgo_t algo = algo_perf.algo;

    // Get workspace size
    size_t workspace_size;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, query_desc, filter_desc, conv_desc, res_desc, algo, &workspace_size);

    // Allocate workspace
    void *workspace;
    cudaMalloc(&workspace, workspace_size);

    // Perform convolution
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn, &alpha, query_desc, d_query, filter_desc, d_reference, conv_desc, algo, workspace, workspace_size, &beta, res_desc, d_res);

    // Measure execution time
    double start_gpu = get_time();
    cudaDeviceSynchronize();
    double end_gpu = get_time();

    // Copy results back to host
    cudaMemcpyAsync(res, d_res, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("SW Time GPU with cuDNN: %.10lf\n", end_gpu - start_gpu);

    // Free device memory
    cudaFree(d_query);
    cudaFree(d_reference);
    cudaFree(d_res);
    cudaFree(workspace);

    // Free host memory
    cudaFreeHost(query);
    cudaFreeHost(reference);
    cudaFreeHost(res);

    // Destroy cuDNN resources
    cudnnDestroyTensorDescriptor(query_desc);
    cudnnDestroyTensorDescriptor(reference_desc);
    cudnnDestroyTensorDescriptor(res_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);

    return 0;
}