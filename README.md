# CUDA-Based Sequence Processing

Parallelized implementation of the Smith-Waterman algorithm for PiA GPU101 @PoliMi.

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Build and Run Instructions](#build-and-run-instructions)
---

## Features

- Parallelized Score and Direction Matrix Computation;
- GPU Backtrace Implementation;
- CPU-GPU Comparison;
- Pseudo-Random Sequence Generation using the Collatz-Weyl Generator (CWG128).

---

## How It Works

1. **Sequence Generation:**
   - Sequences are randomly generated using the Collatz-Weyl Generator, producing nucleotide sequences of given length (S_LEN) for alignment.

2. **Parallelized Computation:**
   - The CUDA kernel sw_parallelized computes the score matrix, direction matrix, and performs the traceback. Each CUDA block handles one sequence pair.

3. **CPU Baseline:**
   - The sw_serialized function implements a single-threaded CPU version of Smith-Waterman for comparison.
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
4. Use `nvcc` to compile the CUDA code:
   ```bash
   nvcc -o sw_parallel_final_true sw_parallel_final_true.cu -O3
   ```

5. **Run the Executable. Optionally, pass a seed for the Collatz-Weyl Generator:**
   ```bash
   ./sw_parallel_final_true [seed]
   ```
---
