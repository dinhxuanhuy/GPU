# GPU — Lộ trình AMD ROCm 🚀

Repo ghi chú + nhật ký học để lấy chứng chỉ **AMD ROCm™ Certified Developer — Associate**, sau đó mở rộng dần (hướng tới [BrrrViz](https://brrrviz.com)).

> **Ngắn hạn:** pass kỳ thi *Core AMD ROCm™ Software Application Development — Certified Associate* (miễn phí, 5 module, badge Credly, hiệu lực 24 tháng).
> **Dài hạn:** dùng ROCm/HIP cho phần compute/visualization của BrrrViz.

---

## 1. Chứng chỉ

Nguồn chính thức: <https://developer.amd.com/rocm-certified/>

| Cấp | Tên | Trạng thái | Nội dung |
| :-- | :-- | :-- | :-- |
| **Associate** | Core AMD ROCm™ Software Application Development | ✅ Đang mở, miễn phí | ROCm setup, Instinct architecture, libraries/frameworks, HIP + CUDA porting, profiling |
| **Professional** | Advanced … Production Readiness | 🔜 Q1'27 | RCCL, multi-GPU, vLLM/SGLang, distributed |
| **Expert** | Architecture-aware … Performance Optimization and Scale | 🔜 2027 | Capstone, K8s/MLOps, fleet-scale infra |

**Lab ở đâu?** Trang chủ ghi: *"Demonstrate hands-on knowledge through **Docker-based AMD GPU labs**, module assessments, and a certification exam."* Các lab này **nằm trong AI Academy, chỉ mở sau khi enroll** — không public, chạy trên AMD GPU (cloud hoặc AMD AI DevDay). Repo này chỉ giữ **ghi chú + nhật ký**, không chép lại nội dung khóa học.

- [ ] Tạo AMD account → join **AI Developer Program**
- [ ] Enroll **ROCm Certified Associate** trên **AI Academy**
- [ ] Đặt ngày thi: __________

---

## 2. Năm module (theo giáo trình chính thức)

### Module 1 — ROCm Orientation, Setup & GPU Architecture Fundamentals
📄 [AMD ROCm Fundamental.md](AMD%20ROCm%20Fundamental.md)

CPU (latency) vs GPU (throughput) · ROCm stack: `amdgpu` driver → ROCr/HSA → HIP/OpenMP/Triton → rocBLAS/MIOpen/RCCL → PyTorch/vLLM · container (`/dev/kfd`, `/dev/dri`) · CLI: `rocm-smi`, `amd-smi`, `hipcc`, `rocprof`.

- [ ] Hiểu + ghi lại bằng lời của mình

### Module 2 — AMD Instinct™ Architecture
📄 [AMD Instinct Architecture.md](AMD%20Instinct%20Architecture.md) · 📄 [Core Glossary.md](Core%20Glossary.md)

MI100 → MI400 (CDNA1→CDNA4) · FP64/FP32/BF16/FP16/FP8/FP4 · MI300X: 8 XCD + 256 MB Infinity Cache + 192 GB HBM3 · CU = 4 SIMD × 16 lane + MFMA · memory hierarchy VGPR → LDS → L2 → HBM · **Wave64** · SIMT & latency hiding · CDNA vs RDNA.

- [ ] Vẽ lại sơ đồ MI300X không nhìn tài liệu
- [ ] Giải thích Wave64 và chuyện workgroup 96 thread = 2 wavefront

### Module 3 — ROCm Libraries & AI Frameworks
`rocBLAS`/`hipBLAS` ↔ cuBLAS · `MIOpen` ↔ cuDNN · `RCCL` ↔ NCCL · rocFFT, rocRAND, hipSPARSE · PyTorch/TF/JAX/ONNX/vLLM · `torch.cuda` API vẫn dùng được trên ROCm.

- [ ] Ghi chú bảng ánh xạ CUDA ↔ ROCm

### Module 4 — HIP Programming & CUDA Porting
HIP ≈ CUDA (`cudaMalloc`→`hipMalloc`…) · `hipcc` (LLVM/Clang; trên NVIDIA wrap `nvcc`) · streams & events · `__shared__` = LDS · **HIPIFY**: `hipify-perl`, `hipify-clang`.

- [ ] Viết được vector add / reduction từ đầu
- [ ] Port thử 1 sample CUDA sang HIP

### Module 5 — Performance Optimization & Profiling
`rocprof`/`rocprofv3`, ROCm Compute Profiler, roofline · coalescing · LDS bank conflict · occupancy (VGPR, LDS) · arithmetic intensity · streams overlap · `torch.profiler`.

- [ ] Đọc được output profiler và chỉ ra bottleneck

---

## 3. Lịch gợi ý (6 tuần)

| Tuần | Nội dung | Xong |
| :-- | :-- | :-- |
| 1 | Module 1 + setup môi trường | [ ] |
| 2 | Module 2 — architecture | [ ] |
| 3 | Module 3 — libraries/frameworks | [ ] |
| 4 | Module 4 — HIP + porting | [ ] |
| 5 | Module 5 — profiling | [ ] |
| 6 | Ôn tập + thi thử + đặt lịch thi | [ ] |

> Máy hiện tại: **NVIDIA RTX 5070 Laptop 8 GB** → không chạy ROCm native. Dùng **AMD Developer Cloud** (credit MI300X của AI Developer Program) hoặc **AMD AI DevDay** để làm lab chính thức.

---

## 4. Nhật ký học

Ghi mỗi buổi học vào **[Nhật ký.md](Nh%E1%BA%ADt%20k%C3%BD.md)** — hôm nay học gì, lab nào đã làm, chỗ nào còn mơ hồ.

---

## 5. Sau chứng chỉ — BrrrViz

- **A:** giữ ghi chú + lab đủ chắc, dựng môi trường ROCm container ổn định.
- **B:** thử 1–2 HIP kernel cho phần nặng của pipeline (biến đổi hình học, sampling, particle/field).
- **C:** đóng gói thành thư viện (pybind11/C API), đo trên MI300X, tune theo architecture.

Khi bắt đầu sẽ tách thành repo/thư mục riêng.

---

## 6. Nguồn

- ROCm Certified — <https://developer.amd.com/rocm-certified/>
- AMD AI Academy — <https://developer.amd.com/amd-ai-academy/>
- ROCm docs — <https://rocm.docs.amd.com/>
- HIP — <https://rocm.docs.amd.com/projects/HIP/en/latest/>
- ROCm Compute Profiler — <https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/>
