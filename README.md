# CUDA and cuDNN-Based Sequence Processing

Parallelized implementation of the Smith-Waterman algorithm for PiA GPU101 @PoliMi on how to use CUDA and cuDNN to process nucleotide sequences. The program generates random sequences, applies a convolution operation using cuDNN, and measures the GPU computation time.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Features](#features)
- [How It Works](#how-it-works)
- [Build and Run Instructions](#build-and-run-instructions)
- [Code Breakdown](#code-breakdown)
- [Performance](#performance)
- [License](#license)

---

## Overview

This program:
- Generates random nucleotide sequences (`A`, `C`, `G`, `T`, and `N`) as query and reference data.
- Leverages CUDA and cuDNN for high-performance sequence processing.
- Uses convolution operations to simulate sequence alignment tasks.
- Measures GPU execution time for performance evaluation.

## Prerequisites

To build and run the program, ensure you have:
1. **NVIDIA GPU** with CUDA Compute Capability 3.0 or higher.
2. **CUDA Toolkit** installed ([Download CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)).
3. **cuDNN Library** installed ([Download cuDNN](https://developer.nvidia.com/cudnn)).
4. **CMake** version 3.28 or higher installed ([Download CMake](https://cmake.org/download/)).

## Features

- **Random Sequence Generation:** Uses a simple nucleotide alphabet for random data creation.
- **Pinned Memory Allocation:** Allocates page-locked memory for efficient GPU-host data transfers.
- **cuDNN Convolution Operations:** Employs cuDNN for high-performance convolution.
- **Performance Timing:** Measures the GPU execution time with microsecond precision.

---

## How It Works

1. **Memory Allocation:**
   - Allocates pinned host memory for sequences and results.
   - Allocates GPU memory for processing.

2. **Sequence Initialization:**
   - Fills the query and reference sequences with random nucleotide characters.

3. **cuDNN Tensor Setup:**
   - Configures tensor descriptors for query, reference, and result data.
   - Sets up convolution and filter descriptors.

4. **Convolution Execution:**
   - Selects the optimal convolution algorithm using `cudnnFindConvolutionForwardAlgorithm`.
   - Allocates workspace for the algorithm.
   - Performs the convolution operation on the GPU.

5. **Performance Timing:**
   - Measures GPU execution time using high-precision timing.

6. **Cleanup:**
   - Frees all allocated resources, including GPU memory, pinned memory, and cuDNN descriptors.

---

## Build and Run Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Penzo00/Smith-Waterman.git
   cd Smith-Waterman
   ```

2. **Configure the Build System with CMake:**
   Use the provided `CMakeLists.txt` file to configure the project. Run the following commands:
   ```bash
   mkdir build
   cd build
   cmake ..
   ```

3. **Build the Program:**
   Compile the program using `make`:
   ```bash
   make
   ```

4. **Run the Executable:**
   ```bash
   ./GPU101
   ```

5. **Output Example:**
   ```plaintext
   Execution Time: 0.0032457340 seconds
   ```

---

## Code Breakdown

### Constants

- `S_LEN` (512): Length of each nucleotide sequence.
- `N` (1000): Number of sequences processed in parallel.

### Functions

- `get_time()`: Returns the current time with microsecond precision for performance measurement.

### Memory Allocation

- **Pinned Host Memory:** Allocated using `cudaMallocHost` for fast host-to-device transfers.
- **Device Memory:** Allocated using `cudaMalloc`.

### cuDNN Setup

- Tensor descriptors are configured as 4D arrays for input, reference, and results.
- Convolution and filter descriptors are set up for cross-correlation operations.

### Execution Flow

- **Data Transfer:** Copies input data to the GPU asynchronously using `cudaMemcpyAsync`.
- **Algorithm Selection:** Optimizes the convolution operation with `cudnnFindConvolutionForwardAlgorithm`.
- **Convolution:** Executes the forward pass using `cudnnConvolutionForward`.

### Cleanup

- Frees all resources, ensuring no memory leaks.

---

## Performance

- The program leverages pinned memory and asynchronous data transfers for optimal performance.
- Timing precision ensures accurate measurement of GPU processing time.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
