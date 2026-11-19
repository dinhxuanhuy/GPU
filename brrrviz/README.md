# BrrrViz — prototype (tương lai)

Thư mục dành cho phần R&D đưa GPU acceleration vào [brrrviz.com](https://brrrviz.com).

Chưa bắt đầu — mở khoá sau khi có chứng chỉ **ROCm Certified Associate**.

## Ý tưởng ban đầu

- HIP kernel cho các bước nặng: geometry transform, sampling, tone-mapping, particle/field update.
- So sánh 3 backend: CPU (baseline) · CUDA (nếu có) · HIP/ROCm.
- Đo bằng roofline: memory-bound hay compute-bound, speedup thực tế trên MI300X (cloud).

## Khi bắt đầu, tạo

```
brrrviz/
├── README.md          # mục tiêu + kết quả đo
├── kernels/           # *.hip
├── bench/             # script benchmark, output CSV
└── docs/              # phân tích, roofline, quyết định kỹ thuật
```
