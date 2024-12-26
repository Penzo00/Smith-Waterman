#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define S_LEN 512  // Length of sequences
#define N 1000     // Number of sequences
#define MATCH 1    // Score for a match
#define MISMATCH -1 // Score for a mismatch
#define INS -2     // Score for an insertion
#define DEL -2     // Score for a deletion

// Function to find the maximum of four values
__device__ inline int max4(int a, int b, int c, int d) {
    return max(max(a, b), max(c, d));
}

// Kernel function for Smith-Waterman algorithm
__global__ void smith_waterman_kernel(const char *query, const char *reference, int *res) {
    extern __shared__ int sc_mat[]; // Shared memory for scoring matrix

    int n = blockIdx.x; // Sequence index
    int max_score = 0;  // Variable to store the maximum score
    int thread_row = threadIdx.y; // Row index for the thread
    int thread_col = threadIdx.x; // Column index for the thread

    // Initialize the scoring matrix
    for (int i = thread_row; i <= S_LEN; i += blockDim.y) {
        for (int j = thread_col; j <= S_LEN; j += blockDim.x) {
            sc_mat[(i * (S_LEN + 1) + j) ^ 1] = 0;
        }
    }
    __syncthreads(); // Synchronize threads

    // Fill the scoring matrix
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
        __syncthreads(); // Synchronize threads
    }

    // Reduce max_score across the warp
    int lane_id = threadIdx.x % 32;
    for (int offset = 16; offset > 0; offset /= 2) {
        max_score = max(max_score, __shfl_down_sync(0xFFFFFFFF, max_score, offset));
    }

    // Update the result array with the maximum score
    if (lane_id == 0) atomicMax(&res[n], max_score);
}

// Function to get the current time
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    srand(time(NULL));
    char alphabet[5] = {'A', 'C', 'G', 'T', 'N'};

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

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(N);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaGraphExec_t instance;

    // Capture the CUDA graph
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // Launch the kernel
    smith_waterman_kernel<<<blocksPerGrid, threadsPerBlock, (S_LEN + 1) * (S_LEN + 1) * sizeof(int), stream>>>(d_query, d_reference, d_res);

    // End capture and instantiate the graph
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

    // Measure execution time
    double start_gpu = get_time();
    cudaGraphLaunch(instance, stream);
    cudaStreamSynchronize(stream);
    double end_gpu = get_time();

    // Copy results back to host
    cudaMemcpyAsync(res, d_res, N * sizeof(int), cudaMemcpyDeviceToHost, stream);

    printf("SW Time GPU: %.10lf\n", end_gpu - start_gpu);

    // Free device memory
    cudaFree(d_query);
    cudaFree(d_reference);
    cudaFree(d_res);

    // Free host memory
    cudaFreeHost(query);
    cudaFreeHost(reference);
    cudaFreeHost(res);

    // Destroy CUDA resources
    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(instance);

    return 0;
}