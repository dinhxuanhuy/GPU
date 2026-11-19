#!/usr/bin/env bash
# Kiểm tra nhanh môi trường ROCm trước khi làm lab.
# Dùng: ./scripts/check-rocm-env.sh

set -u

ok()   { printf '  [ok]   %s\n' "$1"; }
warn() { printf '  [warn] %s\n' "$1"; }
fail() { printf '  [fail] %s\n' "$1"; }

echo "== Hệ điều hành =="
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  echo "  ${PRETTY_NAME:-unknown}"
fi
uname -r

echo
echo "== Kernel driver =="
if [ -e /dev/kfd ] && [ -e /dev/dri ]; then
  ok "/dev/kfd và /dev/dri tồn tại"
else
  warn "thiếu /dev/kfd hoặc /dev/dri (chưa có amdgpu driver / không phải máy AMD)"
fi

if lsmod 2>/dev/null | grep -q '^amdgpu'; then
  ok "module amdgpu đã load"
else
  warn "module amdgpu chưa load"
fi

echo
echo "== ROCm userspace =="
if command -v rocminfo >/dev/null 2>&1; then
  ok "rocminfo: $(rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-z]*' || echo 'không thấy gfx target')"
  rocminfo 2>/dev/null | grep -m3 'Name:' | sed 's/^/         /'
else
  fail "không có rocminfo (ROCm chưa cài?)"
fi

if command -v rocm-smi >/dev/null 2>&1; then
  ok "rocm-smi khả dụng"
  rocm-smi --showproductname --showuse --showmeminfo vram 2>/dev/null | sed 's/^/         /' | head -20
else
  warn "không có rocm-smi"
fi

if command -v amd-smi >/dev/null 2>&1; then
  ok "amd-smi khả dụng"
else
  warn "không có amd-smi (tuỳ chọn)"
fi

echo
echo "== Compiler & profiler =="
for tool in hipcc rocprof rocprofv3 hipify-clang hipify-perl; do
  if command -v "$tool" >/dev/null 2>&1; then
    ok "$tool"
  else
    warn "không có $tool"
  fi
done

if command -v hipcc >/dev/null 2>&1; then
  hipcc --version 2>/dev/null | sed 's/^/         /' | head -4
fi

echo
echo "== Python / PyTorch =="
python3 - <<'PY' 2>/dev/null || warn "không import được torch"
import torch
print(f"  [ok]   torch {torch.__version__}  hip={getattr(torch.version, 'hip', None)}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"         device: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"         arch: {getattr(props, 'gcnArchName', 'n/a')}  VRAM: {props.total_memory / 1024**3:.1f} GiB")
PY

echo
echo "Xong. Trên máy NVIDIA, các mục AMD sẽ [fail]/[warn] — dùng AMD Developer Cloud hoặc container ROCm."
