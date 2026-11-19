# GPU — AMD ROCm Learning Path 🚀

Repo học tập + nhật ký thực hành để lấy chứng chỉ **AMD ROCm™ Certified Developer — Associate**, và sau đó mở rộng sang các dự án GPU của riêng mình (hướng tới **[BrrrViz](https://brrrviz.com)**).

> **Mục tiêu ngắn hạn:** pass kỳ thi *Core AMD ROCm™ Software Application Development — Certified Associate* (miễn phí, 5 module, badge Credly, hiệu lực 24 tháng).
> **Mục tiêu dài hạn:** dùng ROCm/HIP cho visualization & compute workloads của BrrrViz.

---

## 1. Chứng chỉ này là gì?

Chương trình chính thức: <https://developer.amd.com/rocm-certified/>

| Cấp độ | Tên | Trạng thái | Nội dung chính |
| :--- | :--- | :--- | :--- |
| **Associate** | Core AMD ROCm™ Software Application Development | ✅ Đang mở (miễn phí) | ROCm setup, Instinct architecture, libraries/frameworks, HIP + CUDA porting, profiling & optimization |
| **Professional** | Advanced AMD ROCm™ Software Development and Production Readiness | 🔜 ~Q1'27 | Multi-GPU (RCCL), inference serving (vLLM/SGLang), distributed training, production tuning |
| **Expert** | Architecture-aware AMD ROCm™ Software Performance Optimization and Scale | 🔜 ~2027 | Capstone, Kubernetes/MLOps, fleet-scale GPU infra, system-level profiling |

**Điều kiện & phần thưởng**

- Tham gia **AMD AI Developer Program** (miễn phí) → học trên **AMD AI Academy** → thi.
- Hình thức: knowledge exam + practical tasks (lab-based, có thể proctored).
- Đạt → **Credly digital badge** + 500 ADP points.
- Hiệu lực **24 tháng**, cửa sổ gia hạn mở trước hạn **180 ngày**.
- Có thể thi trực tiếp tại **AMD AI DevDay** (workshop 4 giờ: 3h lab + 1h exam).

---

## 2. Đăng ký — Step by step (≈15 phút)

1. Tạo **AMD account** tại <https://account.amd.com>.
2. Đăng ký **AMD AI Developer Program**: <https://developer.amd.com/ai-developer-program/>.
3. Xác minh email → vào **AI Academy**: <https://developer.amd.com/amd-ai-academy/>.
4. Enroll khóa **ROCm Certified Associate** (5 module).
5. Học + làm lab theo lộ trình ở mục 3.
6. Đặt lịch thi (online hoặc tại AI DevDay gần nhất).
7. Nhận badge trên **Credly** → gắn vào LinkedIn/CV.

- [ ] Đã tạo AMD account
- [ ] Đã join AI Developer Program
- [ ] Đã enroll ROCm Certified Associate
- [ ] Đã đặt ngày thi: __________

---

## 3. Lộ trình 5 module (checklist tiến độ)

### Module 1 — ROCm Orientation, Setup & GPU Architecture Fundamentals
📄 Ghi chú: [AMD ROCm Fundamental.md](AMD%20ROCm%20Fundamental.md)

**Phải nắm:** CPU (latency) vs GPU (throughput) · ROCm stack (amdgpu driver → ROCr/HSA → HIP/OpenMP/Triton → rocBLAS/MIOpen/RCCL → PyTorch/vLLM) · container hoá (`/dev/kfd`, `/dev/dri`) · `rocm-smi`, `amd-smi`, `hipcc`, `rocprof`.

**Lab**
- [ ] Chạy `rocm/pytorch` container, xác nhận GPU visible
- [ ] `rocm-smi --showuse --showmeminfo vram` đọc được util/VRAM
- [ ] Build & chạy [`labs/00-vector-add`](labs/00-vector-add) bằng `hipcc`
- [ ] In ra `gfx` target đang dùng (`rocminfo | grep gfx`)

### Module 2 — AMD Instinct™ Architecture
📄 Ghi chú: [AMD Instinct Architecture.md](AMD%20Instinct%20Architecture.md) · Thuật ngữ: [Core Glossary.md](Core%20Glossary.md)

**Phải nắm:** MI100 → MI400 (CDNA1→CDNA4/Next), FP64/FP32/BF16/FP16/FP8/FP4 · MI300X = 8 XCD + 256 MB Infinity Cache + 192 GB HBM3 · CU = 4 SIMD × 16 lane = 64 ALU + MFMA · memory hierarchy: VGPR/SGPR → LDS 64 KB + L1 32 KB → L2 4 MB/XCD → HBM3 · **Wave64** (Wave32 trên RDNA) · SIMT + latency hiding · Grid → Workgroup → Wavefront → SIMD · CDNA vs RDNA.

**Lab**
- [ ] Vẽ lại sơ đồ MI300X từ trí nhớ (XCD → CU → SIMD → MFMA)
- [ ] Tính: 1 workgroup 96 thread chiếm mấy wavefront? lãng phí bao nhiêu lane? → **2 wavefront, phí 32 lane**
- [ ] Ước lượng số wavefront resident tối đa theo VGPR/LDS (occupancy)
- [ ] So sánh bảng CDNA vs RDNA bằng lời của mình

### Module 3 — ROCm Libraries & AI Frameworks
**Phải nắm:** `rocBLAS`/`hipBLAS` ↔ cuBLAS · `MIOpen` ↔ cuDNN · `RCCL` ↔ NCCL · `rocFFT`, `rocRAND`, `hipSPARSE`, `rocPRIM` · PyTorch ROCm, TensorFlow, JAX, ONNX Runtime, vLLM · `torch.cuda` API vẫn dùng được trên ROCm.

**Lab**
- [ ] `torch.cuda.is_available() == True`, in `torch.cuda.get_device_name(0)`
- [ ] Benchmark matmul FP32/BF16/FP16 → tính TFLOPS thực tế
- [ ] Fine-tune/inference 1 model nhỏ (Llama/Qwen ≤ 1B) bằng PyTorch ROCm
- [ ] Chạy `rocblas-bench` hoặc `hipblas-bench` cho GEMM
- [ ] (tuỳ chọn) serve model bằng vLLM trên 1 GPU

### Module 4 — HIP Programming & CUDA Porting
**Phải nắm:** HIP ≈ CUDA (`cudaMalloc`→`hipMalloc`, `cudaMemcpy`→`hipMemcpy`, `__global__` giữ nguyên) · `hipcc` (LLVM/Clang → GCN/CDNA/RDNA; trên NVIDIA wrap sang `nvcc`) · `hipMalloc/hipMemcpy/hipFree`, `hipLaunchKernelGGL`, streams & events, `hipDeviceSynchronize` · error checking (`hipGetLastError`) · `__shared__` = LDS, `__syncthreads` · **HIPIFY**: `hipify-perl`, `hipify-clang`.

**Lab**
- [ ] `labs/00-vector-add`: grid-stride loop + đo bandwidth
- [ ] Reduction (sum) dùng `__shared__` + `__syncthreads`, so sánh với naive
- [ ] Matrix transpose tối ưu bằng LDS (tránh bank conflict)
- [ ] Port 1 sample CUDA → HIP và chạy được

### Module 5 — Performance Optimization & Profiling
**Phải nắm:** `rocprof` / `rocprofv3`, **ROCm Compute Profiler** (omniperf), `rocminfo`, roofline model · memory coalescing · LDS bank conflicts · occupancy (VGPR pressure, LDS size) · arithmetic intensity · kernel launch overhead & streams overlap · `torch.profiler`.

**Lab**
- [ ] `rocprofv3 --stats ./vector_add` → đọc kernel time
- [ ] Roofline: workload này memory-bound hay compute-bound?
- [ ] Tối ưu reduction qua 3 phiên bản, ghi lại speedup
- [ ] Profile 1 model PyTorch → tìm top-3 kernel tốn thời gian

---

## 4. Lịch học gợi ý (6 tuần, ~8–10 h/tuần)

| Tuần | Nội dung | Deliverable |
| :--- | :--- | :--- |
| 1 | Module 1 + setup môi trường | vector-add chạy được, ảnh `rocm-smi` |
| 2 | Module 2 (architecture) | sơ đồ MI300X + ghi chú occupancy |
| 3 | Module 3 (libraries/frameworks) | benchmark matmul + note TFLOPS |
| 4 | Module 4 (HIP) | reduction + transpose có LDS |
| 5 | Module 5 (profiling) | báo cáo profile + tối ưu |
| 6 | Ôn tập + mock exam | pass checklist mục 5, đặt lịch thi |

---

## 5. Môi trường thực hành

**Máy hiện tại:** Ubuntu + **NVIDIA RTX 5070 Laptop (8 GB)** → không chạy ROCm native.
→ Dùng để đối chiếu CUDA ↔ HIP và làm quen cú pháp; **lab ROCm thật** chạy trên một trong các môi trường sau:

| Lựa chọn | Ghi chú |
| :--- | :--- |
| **AMD Developer Cloud** | MI300X theo credit của AI Developer Program — khuyến nghị |
| **Docker `rocm/pytorch`** | Cần host AMD GPU + `amdgpu` kernel driver |
| **AMD AI DevDay** | Lab + thi trực tiếp, có máy sẵn |

```bash
# Kiểm tra môi trường
rocminfo | grep -E "Name:|gfx"      # thấy gfx942 (MI300X), gfx90a (MI200)...
rocm-smi --showproductname
hipcc --version
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Container chuẩn
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --shm-size 16G \
  rocm/pytorch:latest
```

Script kiểm tra nhanh: [`scripts/check-rocm-env.sh`](scripts/check-rocm-env.sh)

---

## 6. Checklist trước khi thi

- [ ] Giải thích được ROCm stack từ driver → runtime → libraries → framework
- [ ] Đọc thành thạo output `rocm-smi` / `rocminfo`
- [ ] Viết được kernel HIP từ đầu (vector add, reduction) không cần tra cú pháp
- [ ] Biết map CUDA API → HIP API và dùng `hipify-clang`
- [ ] Đọc được `rocprofv3` và chỉ ra bottleneck
- [ ] Giải thích Wave64, LDS, VGPR, occupancy, roofline bằng lời của mình
- [ ] Làm lại toàn bộ lab trong repo mà không xem đáp án

---

## 7. Sau chứng chỉ — hướng tới BrrrViz

[BrrrViz](https://brrrviz.com) là mục tiêu dài hạn: đưa GPU acceleration vào pipeline visualization/render của mình.

**Phase A — Nền tảng (song song lúc học cert)**
- Giữ mọi lab trong repo này thành code chạy được, có benchmark.
- Chuẩn hoá môi trường: Dockerfile ROCm + CI chạy được trên AMD GPU runner.

**Phase B — Prototype**
- Viết 1–2 HIP kernel cho phần tính toán nặng của pipeline (biến đổi hình học, sampling, tone-mapping, field/particle update).
- So sánh với bản CUDA (nếu có) và bản CPU → ghi lại speedup, roofline.

**Phase C — Tích hợp**
- Đóng gói kernel thành thư viện (pybind11 / C API) và gọi từ app.
- Serving/streaming: cân nhắc vLLM cho phần AI-assisted visualization.
- Đo trên MI300X (cloud) và tune theo architecture.

Ghi chú BrrrViz sẽ nằm trong [`brrrviz/`](brrrviz/) khi bắt đầu.

---

## 8. Cấu trúc repo

```
.
├── README.md                      # lộ trình + tiến độ (file này)
├── AMD ROCm Fundamental.md        # Module 1 notes
├── AMD Instinct Architecture.md   # Module 2 notes
├── Core Glossary.md               # thuật ngữ GPU/ROCm
├── labs/
│   └── 00-vector-add/             # lab HIP đầu tiên (có Makefile)
├── scripts/
│   └── check-rocm-env.sh          # kiểm tra môi trường ROCm
└── brrrviz/                       # (tương lai) prototype cho brrrviz.com
```

## 9. Nguồn tham khảo

- ROCm Certified: <https://developer.amd.com/rocm-certified/>
- AMD AI Academy: <https://developer.amd.com/amd-ai-academy/>
- ROCm docs: <https://rocm.docs.amd.com/>
- HIP programming guide: <https://rocm.docs.amd.com/projects/HIP/en/latest/>
- ROCm Compute Profiler: <https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/>
- AMD Instinct MI300X: <https://www.amd.com/en/products/accelerators/instinct/mi300.html>

---

*Ghi chú cá nhân — cập nhật tiến độ bằng cách tick checkbox ngay trong README.*
