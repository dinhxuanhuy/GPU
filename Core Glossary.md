---
tags: [amd, rocm, gpu-computing, glossary, hardware, architecture]
topic: GPU & ROCm Core Glossary
---

# GPU & ROCm Core Hardware Glossary

## 1. Fundamental Compute & Execution Terms

| Term | Full Name | Plain-English Definition & Role |
| :--- | :--- | :--- |
| **==ALU==** | Arithmetic Logic Unit | The core "calculator" circuit inside a processor that performs actual math (`+`, `-`, `*`, `/`) and logic operations (`AND`, `OR`). |
| **==CPU vs. GPU==** | Central vs. Graphics Processing Unit | **CPU** dedicates chip area to big caches & complex control logic to finish *single tasks* with minimal latency. **GPU** strips those away to pack thousands of ==ALUs== for massive *parallel throughput*. |
| **==CU==** | Compute Unit | The fundamental processing block of an AMD GPU (analogous to NVIDIA's Streaming Multiprocessor - SM). Contains scalar units, ==SIMD== vector ALUs, ==Matrix Cores==, and local memory. |
| **==SIMD==** | Single Instruction, Multiple Data | Hardware vector lanes inside a CU that apply **one single operation** across a vector of multiple data elements simultaneously. |
| **==SIMT==** | Single Instruction, Multiple Threads | Programming model where you write code for 1 thread, and hardware groups 64 threads into a ==Wavefront== executing the same instruction in lockstep. |
| **==Thread==** | Execution Thread | The smallest software worker unit, responsible for computing one single element (e.g., one pixel or one matrix cell). |
| **==Wavefront==** | Hardware Thread Group | A bundle of **==64 threads (Wave64)==** scheduled and executed together in lockstep on AMD CDNA GPUs (analogous to NVIDIA's 32-thread *Warp*). |
| **==Block / Workgroup==** | Thread Block | A group of multiple wavefronts assigned to a single ==CU==, able to share data via ==LDS== and synchronize using barriers. |
| **==Grid==** | Kernel Launch Grid | The entire collection of blocks launched onto the GPU to execute a kernel function. |
| **==Matrix Core (MFMA)==** | Matrix Fused Multiply-Add | Specialized hardware inside each CU (analogous to NVIDIA *Tensor Cores*) that multiplies entire small matrices in a single clock cycle for AI. |

---

## 2. Memory & Interconnect Terms (Fastest $\rightarrow$ Largest)

| Term | Full Name | Plain-English Definition & Role |
| :--- | :--- | :--- |
| **==Registers==** | VGPR / SGPR | Ultra-fast, private memory slots right next to the ==ALUs== inside a CU (`0-cycle` latency) storing local variables for active threads. |
| **==LDS==** | Local Data Share | Fast, programmer-managed scratchpad memory inside a CU (analogous to CUDA *Shared Memory*) allowing threads in the same ==Block== to share data without accessing VRAM. |
| **==L1 / L2 / Infinity Cache==** | Hardware Cache Hierarchy | Automatic on-chip high-speed buffers that store recently used data to avoid slow trips to main HBM memory. |
| **==HBM==** | High Bandwidth Memory | 3D-stacked DRAM chips placed directly beside the GPU die on an interposer, delivering massive capacity (e.g., ==192 GB==) and multi-TB/s bandwidth. |
| **==Chiplet (XCD / IOD)==** | Accelerator Complex / I/O Die | Modular chip design splitting compute cores (**==XCD==**) and cache/memory controllers (**==IOD==**) into smaller dies stacked together. |
| **==Infinity Fabric™==** | AMD High-Speed Interconnect | High-bandwidth data highway linking chiplets inside a GPU, connecting multiple GPUs (**GPU-to-GPU**), and linking CPUs to GPUs. |

---

## 3. Performance Metrics & Precision Formats

| Term | Full Name | Plain-English Definition & Role |
| :--- | :--- | :--- |
| **==Latency==** | Execution Delay | Time taken to finish **one single task** from start to finish (measured in `ms` or clock cycles). |
| **==Throughput==** | Processing Rate | Total amount of work completed **per second** (measured in `FLOPS`, `tokens/sec`, or `GB/s`). |
| **==FLOPS==** | Floating-Point Ops / Sec | Raw compute speed measuring decimal math operations per second (`TFLOPS` = $10^{12}$, `PFLOPS` = $10^{15}$). |
| **==Bandwidth==** | Memory Transfer Rate | Width of the data highway between memory and compute cores (measured in `TB/s` or `GB/s`). |
| **==FP64 / FP32==** | Double / Single Precision | 64-bit and 32-bit decimal numbers used for high-accuracy **HPC scientific simulations** and baseline AI training. |
| **==BF16 / FP16==** | Brain Float / Half Precision | 16-bit decimal formats offering **2x faster compute** and **half the VRAM** of FP32; ==BF16== is the standard for LLM training. |
| **==FP8 / FP4 / INT8==** | Low-Bit Quantized Formats | 8-bit and 4-bit formats maximizing ==PFLOPS== and shrinking model size for high-speed LLM inference. |