---
tags: [amd, rocm, instinct, cdna, rdna, gpu-architecture, simt, hpc]
module: 2 - AMD Instinct Architecture
---
## 1. AMD Instinct™ Lineage

Evolution of AMD data-center compute GPUs where each new CDNA generation roughly doubles peak AI throughput and expands memory. Knowing the Instinct generation tells you the **supported precisions** and **available HBM capacity**:

* **MI100 (CDNA 1, 2020):** Introduced first-gen Matrix Cores.
* **MI200 (CDNA 2, 2021):** Multi-die packaging with dominant FP64 HPC throughput.
* **MI300 / MI325 (CDNA 3, 2023–24):** 3D Chiplet architecture adding **FP8** support and massive HBM3/HBM3E capacity.
* **MI350 / MI355 (CDNA 4, 2025):** Adds native **FP6 / FP4** support (~10 PFLOPS FP8 / ~20 PFLOPS FP4).
* **MI400 (CDNA Next, H2 2026):** 2nm process, HBM4 (432 GB, 19.6 TB/s), and UALink interconnect.

---

## 2. MI300X Architecture & Anatomy

![[Pasted image 20261119155221.png]]

### Zooming Into the MI300X Package
1. **Full Package:** Looks like one GPU, but built with **8 XCDs** (Accelerator Complex Dies) stacked on top of I/O Dies containing **256 MB Infinity Cache**, surrounded by **8 HBM3 stacks (192 GB total)**.
   * **Host interconnect:** PCIe Gen 5.
   * **Peer-to-peer GPU interconnect:** AMD Infinity Fabric™.
   * **Practical advantage of 192 GB HBM3:** Fits large models and large batch sizes on a **single GPU**, reducing multi-GPU sharding and communication overhead.
2. **One XCD (Accelerator Complex Die):** Houses **38 active Compute Units (CUs)** (304 active CUs total across 8 XCDs) sharing a **4 MB L2 Cache**.
3. **One Compute Unit (CU):** The core execution building block we optimize kernels for, containing:
   * **4 SIMD Units** (16 lanes/ALUs each = **64 vector ALUs** total).
   * **MFMA (Matrix Fused Multiply-Add):** Dedicated Matrix Core engine (4 per CU, one per SIMD).
   * **Scalar Unit:** Handles control flow, loops, and memory address calculations.

### SIMD Unit — The CU's Vector Engine
![[Pasted image 20261119155645.png]]
* **SIMD (Single Instruction, Multiple Data):** Where vector math happens.
* Each SIMD unit is **16 lanes wide** (1 lane = 1 ALU $\rightarrow$ 16 ALUs per SIMD).
* 1 CU = 4 SIMDs $\times$ 16 lanes = **64 ALUs**.

### MFMA — One Instruction, One Full Matrix Tile
![[Pasted image 20261119155932.png]]
* While SIMD ALUs execute 1D vector operations (one scalar op per lane), **MFMA** computes a full 2D matrix tile multiply-accumulate ($D = A \times B + C$) in a **single instruction**—the core building block of GEMM.
* **Mixed-precision pattern:** Inputs often use lower precision (e.g., FP16, BF16, FP8) for high throughput, while the **accumulator uses FP32** to preserve numerical accuracy and avoid overflow/underflow.

---

## 3. On-Chip Memory Hierarchy

> **Rule of thumb:** Moving **one level DOWN** the memory hierarchy (e.g., LDS $\rightarrow$ L2 $\rightarrow$ HBM) is roughly **~10x slower and much larger**.

| Memory Level | Scope | Managed By | Key Characteristics |
| :--- | :--- | :--- | :--- |
| **1. Registers (VGPR / SGPR)** | Private per thread | Compiler | Fastest (zero-cycle latency), inside the CU. Stores thread-local variables. |
| **2. LDS (64 KB) & L1 Cache (32 KB)** | Shared by **1 workgroup** | **Programmer** (explicitly) | Ultra-fast on-chip scratchpad (analogous to CUDA Shared Memory, e.g., `__shared__` in HIP). Enables thread data reuse without touching HBM. |
| **3. L2 Cache (4 MB/XCD) & Infinity Cache (256 MB)** | Shared across GPU | **Hardware** (automatic) | On-chip caches that filter global memory traffic. |
| **4. HBM3 (192 GB)** | Global (all workgroups) | Programmer (allocation) | Off-die 3D-stacked main VRAM. Largest capacity, highest latency. |

---

## 4. GPU Programming & Execution Model

### Software-to-Hardware Mapping
$$\text{Grid} \longrightarrow \text{Block (Workgroup)} \longrightarrow \text{Wavefront} \longrightarrow \text{SIMD}$$

* **Grid:** Entire kernel launch across the GPU.
* **Block (Workgroup):** Group of threads assigned to a single CU, sharing LDS and synchronizing via barriers.
* **Wavefront:** Hardware scheduling unit of **64 threads (Wave64)** executed in lockstep on CDNA (analogous to NVIDIA's 32-thread Warp).
  * *Alignment note:* Hardware always allocates full 64-lane wavefronts. Launching a workgroup of **96 threads** executes **2 wavefronts** (64 + 32 active), **wasting 32 lanes** on the second wavefront.
* **SIMD Units:** Physical vector lanes inside the CU executing wavefront instructions.

### SIMT & Hiding Memory Latency
* **SIMT (Single Instruction, Multiple Threads):** Programmers write thread-level code while hardware groups 64 threads into a wavefront executing the same instruction simultaneously on different data.
* **Latency Hiding:** Instead of relying on massive caches like a CPU, each CU keeps dozens of active wavefronts resident. When one wavefront stalls waiting on slow HBM data, the hardware scheduler switches to another ready wavefront in **zero cycles**, keeping ALUs utilized.

---

## 5. CDNA vs. RDNA Architecture

AMD splits its GPU architectures by workload while preserving **single-source portability via HIP** (the same HIP code compiles via `hipcc` for both architectures):

| Feature | CDNA (Compute DNA) | RDNA (Radeon DNA) |
| :--- | :--- | :--- |
| **Target Domain** | Data-center compute, AI, HPC | Professional visualization & consumer gaming/compute |
| **Product Line** | AMD Instinct™ (MI100, MI200, MI300, MI350) | Radeon™ RX / Radeon™ PRO |
| **Graphics Hardware** | Stripped out (pure compute silicon) | Included (Ray Accelerators, rasterizers, display engines) |
| **Matrix Math** | Dedicated Matrix Cores (MFMA) | Lightweight AI Accelerators |
| **Memory Subsystem** | HBM2e / HBM3 / HBM3E (multi-TB/s bandwidth) | GDDR6 |
| **Default Wavefront** | Wave64 | Wave32 (supports Wave64) |
| **Portability** | Same HIP source compiles via `hipcc` to both | Same HIP source compiles via `hipcc` to both |