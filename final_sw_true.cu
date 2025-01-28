#include <array>      // Modern C++ array
#include <chrono>     // For time measurement
#include <cstring>    // For C-style string handling
#include <iostream>   // For input and output
#include <memory>     // For smart pointers
#include <cuda_runtime.h> // CUDA runtime API

constexpr int S_LEN = 512;                               // Length of sequences
constexpr int N = 1000;                                  // Number of sequence pairs
constexpr int BLOCK_SIZE = 512;                          // CUDA block size
constexpr int INS = -2;                                  // Gap insertion penalty
constexpr int DEL = -2;                                  // Gap deletion penalty
constexpr int MATCH = 1;                                 // Match reward
constexpr int MISMATCH = -1;                             // Mismatch penalty
constexpr char alphabet[5] = {'A', 'C', 'G', 'T', 'N'};  // Nucleotide alphabet

using namespace std;

// Collatz-Weyl generator state
static array<__uint128_t, 4> c;

// Collatz-Weyl Generator implementation
__uint128_t CWG128() {
    c[1] = (c[1] >> 1) * ((c[2] += c[1]) | 1) ^ (c[3] += c[0]);
    return c[2] >> 96 ^ c[1];
}

// Initialize the Collatz-Weyl generator
void init_CWG(__uint128_t seed) {
    c[0] = seed | 1;  // Ensure c[0] is odd
    c[1] = seed ^ 0xdeadbeefcafebabeULL;
    c[2] = seed ^ 0x123456789abcdef0ULL;
    c[3] = seed ^ 0xfedcba9876543210ULL;

    // Decorrelate states by skipping 96 outputs
    for (int i = 0; i < 96; ++i) {
        CWG128();
    }
}

// Measure time in seconds
inline double get_time() {
    const auto now = chrono::high_resolution_clock::now();
    const auto duration = chrono::duration_cast<chrono::microseconds>(now.time_since_epoch());
    return duration.count() * 1e-6;
}

__device__ int d_max4(const int n1, const int n2, const int n3, const int n4)
{
    int tmp1 = n1 > n2 ? n1 : n2;
    const int tmp2 = n3 > n4 ? n3 : n4;
    tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
    return tmp1;
}

// CUDA device function: Get matrix index for block and position
__device__ int idx_mat(const int block, const int dim, const int i, const int j) {
    return block * dim * dim + i * dim + j;
}

// CUDA device function: Get array index for block and position
__device__ int idx_arr(const int block, const int dim, const int i) {
    return block * dim + i;
}

__device__ void dir_score(const char* query, const char* reference, int* sc_mat, char* dir_mat, const int x, const int y, const int block,
    const int dim, int& max_score, int& max_i, int& max_j) {
    const int comparison = (query[idx_arr(block, S_LEN, x - 1)] == reference[idx_arr(block, S_LEN, y - 1)]) ? MATCH : MISMATCH;
    const int diag = sc_mat[idx_mat(block, dim, x - 1, y - 1)] + comparison;
    const int up = sc_mat[idx_mat(block, dim, x - 1, y)] + DEL;
    const int left = sc_mat[idx_mat(block, dim, x, y - 1)] + INS;
    const int score = d_max4(0, diag, up, left);

    const char dir = (score == diag) ? (comparison == MATCH ? 1 : 2) : (score == up) ? 3 : (score == left) ? 4 : 0;

    dir_mat[idx_mat(block, dim, x, y)] = dir;
    sc_mat[idx_mat(block, dim, x, y)] = score;

    if (score > max_score) {
        max_score = score;
        max_i = x;
        max_j = y;
    }
}

// Optimized sw_parallelized kernel with better memory access and warp-level reduction
__global__ void sw_parallelized(const char* query, const char* reference, int* sc_mat, char* dir_mat, int* res, int* max_i_arr,
    int* max_j_arr, char* simple_rev_cigar) {
    const int block = blockIdx.x;
    constexpr int dim = S_LEN + 1;
    __shared__ int max_score_shared;
    __shared__ int max_i_shared;
    __shared__ int max_j_shared;

    if (threadIdx.x == 0) {
        max_score_shared = 0;
        max_i_shared = 0;
        max_j_shared = 0;
    }
    __syncthreads();

    // Initialize first row and column. Use coalesced access.
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        sc_mat[idx_mat(block, dim, i, 0)] = 0;
        dir_mat[idx_mat(block, dim, i, 0)] = 0;
        sc_mat[idx_mat(block, dim, 0, i)] = 0;
        dir_mat[idx_mat(block, dim, 0, i)] = 0;
    }
    __syncthreads();

    // Optimized diagonal traversal for better data locality
    for (int diag = 1; diag < 2 * S_LEN; ++diag) {
        for (int x = max(1, diag - S_LEN + 1) + threadIdx.x; x <= min(S_LEN, diag); x += blockDim.x) {
            const int y = diag - x + 1;
            dir_score(query, reference, sc_mat, dir_mat, x, y, block, dim, max_score_shared, max_i_shared, max_j_shared);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        res[block] = max_score_shared;
        max_i_arr[block] = max_i_shared;
        max_j_arr[block] = max_j_shared;
    }
    __syncthreads(); // Important sync before backtrace

    //Backtrace on GPU
    int i = max_i_shared;
    int j = max_j_shared;
    for (int n = threadIdx.x; n < 2 * S_LEN && dir_mat[idx_mat(block, dim, i, j)] != 0; n += blockDim.x)
    {
       const int dir = dir_mat[idx_mat(block, dim, i, j)];
       if (dir == 1 || dir == 2)
       {
          i--;
          j--;
       }
       else if (dir == 3)
          i--;
       else if (dir == 4)
          j--;

       simple_rev_cigar[block * 2 * S_LEN + n] = dir;
    }
}

__host__ int h_max4(const int n1, const int n2, const int n3, const int n4)
{
    int tmp1 = n1 > n2 ? n1 : n2;
	const int tmp2 = n3 > n4 ? n3 : n4;
	tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
	return tmp1;
}

__host__ void h_backtrace(char *simple_rev_cigar, char **dir_mat, int i, int j, const int max_cigar_len)
{
    for (int n = 0; n < max_cigar_len && dir_mat[i][j] != 0; n++)
	{
		const int dir = dir_mat[i][j];
		if (dir == 1 || dir == 2)
		{
			i--;
			j--;
		}
		else if (dir == 3)
			i--;
		else if (dir == 4)
			j--;

		simple_rev_cigar[n] = dir;
	}
}

__host__ double sw_serialized(char **query, char **reference) {

    const auto sc_mat {static_cast<int **>(malloc((S_LEN + 1) * sizeof(int *)))};

    for (int i = 0; i < (S_LEN + 1); i++)
        sc_mat[i] = static_cast<int *>(malloc((S_LEN + 1) * sizeof(int)));

    const auto dir_mat {static_cast<char **>(malloc((S_LEN + 1) * sizeof(char *)))};

    for (int i = 0; i < (S_LEN + 1); i++)
        dir_mat[i] = static_cast<char *>(malloc((S_LEN + 1) * sizeof(char)));

    const auto res {static_cast<int *>(malloc(N * sizeof(int)))};

    const auto simple_rev_cigar {static_cast<char **>(malloc(N * sizeof(char *)))};

    for (int i = 0; i < N; i++)
        simple_rev_cigar[i] = static_cast<char *>(malloc(S_LEN * 2 * sizeof(char)));

    const double start_cpu {get_time()};

    int global_max {0};         // Track the highest score
    int global_max_index {-1}; // Index of the sequence pair with the highest score

    for (int n = 0; n < N; n++)
    {
        int max = INS; // in SW all scores of the alignment are >= 0, so this will be for sure changed
        int maxi{0}, maxj{0};
        // initialize the scoring matrix and direction matrix to 0
        for (int i {0}; i < S_LEN + 1; i++)
        {
            for (int j {0}; j < S_LEN + 1; j++)
            {
                sc_mat[i][j] = 0;
                dir_mat[i][j] = 0;
            }
        }
        // compute the alignment
        for (int i {1}; i < S_LEN + 1; i++)
        {
            for (int j = 1; j < S_LEN + 1; j++)
            {
                // compare the sequences characters
                const int comparison {(query[n][i - 1] == reference[n][j - 1]) ? MATCH : MISMATCH};

                // compute the cell knowing the comparison result
                const int tmp {h_max4(sc_mat[i - 1][j - 1] + comparison, sc_mat[i - 1][j] + DEL,
                    sc_mat[i][j - 1] + INS, 0)};

                char dir;

                if (tmp == (sc_mat[i - 1][j - 1] + comparison))
                    dir = comparison == MATCH ? 1 : 2;
                else if (tmp == (sc_mat[i - 1][j] + DEL))
                    dir = 3;
                else if (tmp == (sc_mat[i][j - 1] + INS))
                    dir = 4;
                else
                    dir = 0;

                dir_mat[i][j] = dir;
                sc_mat[i][j] = tmp;

                if (tmp > max)
                {
                    max = tmp;
                    maxi = i;
                    maxj = j;
                }
            }
        }
        res[n] = sc_mat[maxi][maxj];
        h_backtrace(simple_rev_cigar[n], dir_mat, maxi, maxj, S_LEN * 2);

        // Update global max if necessary
        if (res[n] > global_max)
        {
            global_max = res[n];
            global_max_index = n;
        }
    }

    const double cpu_time_used {get_time() - start_cpu};

    // Print the sequences with the highest score
    if (global_max_index != -1)
    {
        std::cout << "Reference: " << reference[global_max_index] << "\n\n";
        std::cout << "Query: " << query[global_max_index] << "\n\n";
        std::cout << "Highest score: " << global_max << "\n\n";
        std::cout << "CPU Time: " << cpu_time_used << "\n\n";
    }

    return cpu_time_used;
}

int main(const int argc, char * argv[]) {
	// Initialize Collatz-Weyl Generator with a reproducible seed
	__uint128_t seed = 0xabcdef123456789ULL; // Default seed (can be customized)
	if (argc > 1) seed = strtoull(argv[1], nullptr, 16); // Optionally pass a seed
	init_CWG(seed);

	const auto query {static_cast<char **>(malloc(N * sizeof(char *)))};
	for (int i = 0; i < N; i++)
		query[i] = static_cast<char *>(malloc(S_LEN * sizeof(char)));

	const auto reference {static_cast<char **>(malloc(N * sizeof(char *)))};
	for (int i = 0; i < N; i++)
		reference[i] = static_cast<char *>(malloc(S_LEN * sizeof(char)));

	const auto queries {static_cast<char *>(malloc(N * S_LEN * sizeof(char)))};
	const auto references {static_cast<char *>(malloc(N * S_LEN * sizeof(char)))};

	// Device memory
	char *d_query, *d_reference, *d_dir_mat, *d_simple_rev_cigar;
	int *d_sc_mat, *d_res;

	// Generate sequences using CWG128
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < S_LEN; j++)
		{
			query[i][j] = alphabet[CWG128() % 5];
			reference[i][j] = alphabet[CWG128() % 5];
		}

        // From matrix to array
		strncpy(queries + i * S_LEN, query[i], S_LEN);
		strncpy(references + i * S_LEN, reference[i], S_LEN);

	}

	const double cpu_time_used {sw_serialized(query, reference)};

    const auto h_simple_rev_cigar = static_cast<char *>(malloc(N * 2 * S_LEN * sizeof(char)));

	// Device memory allocation
	cudaMalloc(&d_query, N*S_LEN * sizeof(char));
	cudaMalloc(&d_reference, N*S_LEN * sizeof(char));
	cudaMalloc(&d_sc_mat, N*(S_LEN+1)*(S_LEN+1) * sizeof(int));
	cudaMalloc(&d_dir_mat, N*(S_LEN+1)*(S_LEN+1) * sizeof(char));
	cudaMalloc(&d_simple_rev_cigar, N*2*(S_LEN) * sizeof(char));
	cudaMalloc(&d_res, N * sizeof(int));

    int *d_max_i_arr, *d_max_j_arr;

    int *d_reduced_max, *d_global_max, *d_max_index;

    const auto h_res = static_cast<int *>(malloc(N * sizeof(int)));

    cudaMalloc(&d_reduced_max, (N + BLOCK_SIZE - 1) / BLOCK_SIZE * sizeof(int)); // For reduction
    cudaMalloc(&d_global_max, sizeof(int));
    cudaMalloc(&d_max_index, sizeof(int));

	cudaMemcpy(d_query, queries, N*S_LEN * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_reference, references, N*S_LEN * sizeof(char), cudaMemcpyHostToDevice);
    cudaMalloc(&d_max_i_arr, N * sizeof(int));
    cudaMalloc(&d_max_j_arr, N * sizeof(int));

    int max_index_host;
    cudaMemcpy(&max_index_host, d_max_index, sizeof(int), cudaMemcpyDeviceToHost);

    // Get max_i and max_j from the index
    int max_i, max_j;
    cudaMemcpy(&max_i, &d_max_i_arr[max_index_host], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_j, &d_max_j_arr[max_index_host], sizeof(int), cudaMemcpyDeviceToHost);

    // Execution on GPU started
    const double time_start {get_time()};

    sw_parallelized<<<N, S_LEN>>>(d_query, d_reference, d_sc_mat, d_dir_mat, d_res, d_max_i_arr, d_max_j_arr, d_simple_rev_cigar);

    const double kernel_time_used {get_time() - time_start};

    cudaMemcpy(h_res, d_res, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_simple_rev_cigar, d_simple_rev_cigar, N * 2 * S_LEN * sizeof(char), cudaMemcpyDeviceToHost);

    // Find the highest score and its index
    int highest_score = 0, highest_index = -1;
    for (int i = 0; i < N; i++) {
        if (h_res[i] > highest_score) {
            highest_score = h_res[i];
            highest_index = i;
        }
    }

    const double gpu_time_used {get_time() - time_start};

    // Print the highest score and corresponding sequences
    if (highest_index != -1) {

        // Extract and print the sequences
        std::cout << "Reference: ";
        for (int i = 0; i < S_LEN; i++) {
            std::cout << references[highest_index * S_LEN + i];
        }
        std::cout << "\n\n";

        std::cout << "Query: ";
        for (int i = 0; i < S_LEN; i++) {
            std::cout << queries[highest_index * S_LEN + i];
        }
        std::cout << "\n\n";
        std::cout << "Highest score: " << highest_score << "\n\n";
        std::cout << "GPU Time: " << gpu_time_used << "\n\n";
        std::cout << "Speed up of: " << cpu_time_used / gpu_time_used << "X" << "\n\n";
        std::cout << "Kernel Time: " << kernel_time_used << "\n\n";
        std::cout << "Speed up of: " << cpu_time_used / kernel_time_used << "X" << "\n\n";
    }

    // Free resources
    cudaFree(d_query);
    cudaFree(d_reference);
    cudaFree(d_sc_mat);
    cudaFree(d_dir_mat);
    cudaFree(d_res);
    cudaFree(d_simple_rev_cigar);
    cudaFree(d_reduced_max);
    cudaFree(d_global_max);
    cudaFree(d_max_index);

    free(queries);
    free(references);
    free(query);
    free(reference);
    free(h_res);

    return 0;
}