# Lab 00 — Vector Add

Lab khởi động: kernel HIP đầu tiên, có đo thời gian và bandwidth.

## Chạy

```bash
cd labs/00-vector-add
make            # build bằng hipcc
make run        # chạy
make resources  # xem VGPR / LDS / scratch usage của kernel
```

Kỳ vọng trên MI300X (~5.3 TB/s HBM3): vài trăm GB/s đến ~2–3 TB/s tuỳ grid size.
Trên GPU nhỏ hơn, con số thấp hơn là bình thường.

## Câu hỏi tự kiểm tra

1. Vì sao `block = 256` là lựa chọn tốt trên CDNA? (gợi ý: Wave64)
2. Nếu đổi `grid = 1`, điều gì xảy ra với bandwidth? Vì sao?
3. Vector add là memory-bound hay compute-bound? Tính arithmetic intensity.
4. Bỏ `hipGetLastError()` sau launch thì bug kiểu gì sẽ bị che mất?
5. `hipEventElapsedTime` đo được gì mà `std::chrono` quanh launch không đo được?

## Bài tập mở rộng

- [ ] Thêm biến thể vector add dùng `float4` (vectorized load/store) → so bandwidth.
- [ ] Chạy `rocprofv3 --stats ./vector_add` và đọc grid/block/wavefront thực tế.
- [ ] Đổi sang grid-stride với `grid = số CU * 4` và so sánh.
