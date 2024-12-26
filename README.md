# Smith-Waterman

Parallelized implementation of the Smith-Waterman algorithm for PiA GPU101 @PoliMi

## Table of Contents

- Introduction
- Requirements
- Installation
- Usage
- Code Explanation
- Performance
- License

## Introduction

This project implements a parallelized version of the Smith-Waterman algorithm, designed for the PiA GPU101 course at Politecnico di Milano (PoliMi). The Smith-Waterman algorithm is used for local sequence alignment, and this implementation leverages CUDA to accelerate the computation on NVIDIA GPUs.

## Requirements

- CUDA 12.6
- NVIDIA GPU with CUDA support
- C++ compiler

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Penzo00/Smith-Waterman.git
   cd Smith-Waterman
Compile the code:
nvcc -o smith_waterman smith_waterman.cu -lcudnn
Usage
Run the executable:

./smith_waterman
The program will generate random sequences and compute the alignment scores using the Smith-Waterman algorithm on the GPU.

Code Explanation
max4: A device function to find the maximum of four values.
smith_waterman_kernel: The CUDA kernel that performs the Smith-Waterman algorithm.
get_time: A function to get the current time for performance measurement.
main: The main function that initializes data, launches the kernel, and measures performance.
Detailed Code Explanation
max4 Function:

__device__ inline int max4(int a, int b, int c, int d) {
    return max(max(a, b), max(c, d));
}
This function computes the maximum of four integers, used to determine the best score in the Smith-Waterman algorithm.

smith_waterman_kernel Function:

__global__ void smith_waterman_kernel(const char *query, const char *reference, int *res) {
    extern __shared__ int sc_mat[];
    int n = blockIdx.x;
    int max_score = 0;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    for (int i = thread_row; i <= S_LEN; i += blockDim.y) {
        for (int j = thread_col; j <= S_LEN; j += blockDim.x) {
            sc_mat[(i * (S_LEN + 1) + j) ^ 1] = 0;
        }
    }
    __syncthreads();

    for (int diag = 2; diag <= 2 * S_LEN; diag++) {
        int i = diag - thread_col - 1;
        int j = thread_col;

        if (i >= 1 && i <= S_LEN && j >= 1 && j <= S_LEN) {
            int idx_q = n * S_LEN + i - 1;
            int idx_r = n * S_LEN + j - 1;

            int comparison = (query[idx_q] == reference[idx_r]) ? MATCH : MISMATCH;
            int score_diag = sc_mat[(i - 1) * (S_LEN + 1) + (j - 1)] + comparison;
            int score_up = sc_mat[(i - 1) * (S_LEN + 1) + j] + DEL;
            int score_left = sc_mat[i * (S_LEN + 1) + (j - 1)] + INS;
            int best_score = max4(score_diag, score_up, score_left, 0);

            sc_mat[i * (S_LEN + 1) + j] = best_score;
            max_score = max(max_score, best_score);

            if (best_score == 0 && score_diag < 0 && score_up < 0 && score_left < 0) break;
        }
        __syncthreads();
    }

    int lane_id = threadIdx.x % 32;
    for (int offset = 16; offset > 0; offset /= 2) {
        max_score = max(max_score, __shfl_down_sync(0xFFFFFFFF, max_score, offset));
    }

    if (lane_id == 0) atomicMax(&res[n], max_score);
}
This kernel function performs the Smith-Waterman algorithm on the GPU. It initializes the scoring matrix, fills it based on sequence comparisons, and updates the result array with the maximum score.

get_time Function:

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}
This function returns the current time in seconds, used for performance measurement.

main Function:

int main() {
    srand(time(NULL));
    char alphabet[5] = {'A', 'C', 'G', 'T', 'N'};

    char *query, *reference;
    int *res;

    cudaMallocHost(&query, N * S_LEN * sizeof(char));
    cudaMallocHost(&reference, N * S_LEN * sizeof(char));
    cudaMallocHost(&res, N * sizeof(int));

    for (int i = 0; i < N * S_LEN; i++) {
        query[i] = alphabet[rand() % 5];
        reference[i] = alphabet[rand() % 5];
    }

    char *d_query, *d_reference;
    int *d_res;

    cudaMalloc(&d_query, N * S_LEN * sizeof(char));
    cudaMalloc(&d_reference, N * S_LEN * sizeof(char));
    cudaMalloc(&d_res, N * sizeof(int));

    cudaMemcpyAsync(d_query, query, N * S_LEN * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_reference, reference, N * S_LEN * sizeof(char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(N);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaGraphExec_t instance;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    smith_waterman_kernel<<<blocksPerGrid, threadsPerBlock, (S_LEN + 1) * (S_LEN + 1) * sizeof(int), stream>>>(d_query, d_reference, d_res);

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

    double start_gpu = get_time();
    cudaGraphLaunch(instance, stream);
    cudaStreamSynchronize(stream);
    double end_gpu = get_time();

    cudaMemcpyAsync(res, d_res, N * sizeof(int), cudaMemcpyDeviceToHost, stream);

    printf("SW Time GPU: %.10lf\n", end_gpu - start_gpu);

    cudaFree(d_query);
    cudaFree(d_reference);
    cudaFree(d_res);

    cudaFreeHost(query);
    cudaFreeHost(reference);
    cudaFreeHost(res);

    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(instance);

    return 0;
}
The main function initializes the sequences, allocates memory, launches the kernel, and measures the execution time.

Performance
The CUDA implementation significantly reduces the computation time compared to a CPU-only implementation. The use of CUDA graphs further optimizes the kernel launch overhead.

License
This project is licensed under the MIT License.