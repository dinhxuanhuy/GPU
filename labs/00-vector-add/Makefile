HIPCC ?= hipcc
TARGET := vector_add
SRC := vector_add.hip
CXXFLAGS := -O3 -std=c++17

# ROCm >= 5.x: tự chọn arch của GPU đang có (gfx942 = MI300X, gfx90a = MI200...).
# Nếu compiler cũ, thay bằng: --offload-arch=gfx942
ARCH ?= native

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(CXXFLAGS) --offload-arch=$(ARCH) -o $@ $<

run: $(TARGET)
	./$(TARGET)

# In resource usage của kernel (VGPR, LDS, scratch) — hữu ích cho Module 5
resources: $(SRC)
	$(HIPCC) $(CXXFLAGS) --offload-arch=$(ARCH) -Rpass-analysis=kernel-resource-usage -c $< -o /dev/null

clean:
	rm -f $(TARGET) *.o

.PHONY: all run resources clean
