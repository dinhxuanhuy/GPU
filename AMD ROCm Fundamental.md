---
tags: [amd, rocm, gpu-computing, hpc, ai-infrastructure]
module: 1 - ROCm Orientation, Setup & Containers
---

# (1) ROCm Orientation, Setup & Containers

## 1. GPU Computing Foundations
* **CPU (Latency-oriented):** Few powerful cores with large caches and complex control logic; excels at sequential tasks, OS orchestration, and heavy branching (`if-else`).
* **GPU (Throughput-oriented):** Thousands of smaller cores (Compute Units / Stream Processors) dedicating maximum silicon to ALUs; excels at massively parallel workloads like matrix multiplication (GEMM), convolutions, and simulations.

## 2. The AMD ROCm Software Stack
ROCm (Radeon Open Compute) is AMD’s open-source GPU computing ecosystem (counterpart to NVIDIA CUDA), structured bottom-up:
1. **Driver & Runtime:**
   * `AMDGPU` Kernel Driver: Interfaces with CDNA (Data Center: MI200/MI300) and RDNA (Consumer) hardware.
   * `ROCr` (ROCm Runtime): Manages memory allocation, execution queues, and task scheduling via **HSA (Heterogeneous System Architecture)** for unified CPU-GPU memory addressing.
2. **Compilers & Languages:**
   * **HIP (Heterogeneous-compute Interface for Portability):** C++ runtime API and kernel language portable across AMD and NVIDIA GPUs.
   * **OpenMP / Triton:** High-level parallel programming models for HPC and AI kernels (used by PyTorch 2.x).
3. **Math & AI Libraries (CUDA Equivalents):**
   * `rocBLAS` / `hipBLAS` $\leftrightarrow$ `cuBLAS` (Linear Algebra)
   * `MIOpen` $\leftrightarrow$ `cuDNN` (Deep Neural Networks)
   * `RCCL` $\leftrightarrow$ `NCCL` (Multi-GPU / Multi-node communication)
   * `rocRAND`, `rocFFT`, `hipSPARSE`
4. **Frameworks:** Upstream support for **PyTorch**, **TensorFlow**, **JAX**, **vLLM**, and **ONNX Runtime**.

## 3. Portable GPU Programming with HIP
* **Syntactic Parity:** Nearly identical to CUDA (`cudaMalloc` $\rightarrow$ `hipMalloc`, `cudaMemcpy` $\rightarrow$ `hipMemcpy`).
* **Compilation Flow (`hipcc`):**
  * **On AMD GPUs:** Invokes **LLVM/Clang** to compile directly to AMD GCN/CDNA/RDNA machine code.
  * **On NVIDIA GPUs:** Acts as a thin header wrapper mapping HIP calls to CUDA APIs and delegates compilation to `nvcc` with near-zero overhead.
* **Auto-Porting (`HIPIFY`):** `hipify-perl` and `hipify-clang` automatically convert existing CUDA source code into portable HIP code.

## 4. Container Setup & Essential CLI Tools
* **Containers (Docker / Singularity):** Avoids bare-metal kernel/user-space dependency conflicts. Only requires the host `amdgpu` kernel driver while mounting `/dev/kfd` (Kernel Fusion Driver) and `/dev/dri` (Direct Rendering Infrastructure).

```bash
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 16G \
  rocm/pytorch:latest
```
  
## 5. **Core CLI Tools:**

- `rocm-smi` / `amd-smi`: Monitor GPU utilization, thermals, power draw, and VRAM (e.g., `rocm-smi --showuse --showmeminfo vram`).
    
- `hipcc`: Compile `.cpp` / `.hip` source files (e.g., `hipcc vector_add.cpp -o vector_add`).
    
- `rocprof` / `rocprofv3`: Profile kernel execution time, memory bandwidth, and bottlenecks (e.g., `rocprof --stats ./vector_add`).
  

