#!/bin/bash
# ============================================================
# Script cài đặt môi trường: CUDA 12.6 + GCC 12 + PyTorch + mamba-ssm
# Ubuntu 24.04 | GPU: RTX 3050 | Driver: 556.12
# ============================================================

set -e
echo "====== Bắt đầu cài đặt môi trường ======"

# -----------------------------------------------------------
# BƯỚC 1: Cài GCC/G++ 12 (đặt làm mặc định)
# -----------------------------------------------------------
echo ""
echo "[1/6] Cài đặt GCC/G++ 12..."
sudo apt-get install -y gcc-12 g++-12

# Đặt gcc-12 và g++-12 là mặc định qua update-alternatives
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-12 \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-12

# Nếu có gcc-13, hạ ưu tiên xuống
if command -v gcc-13 &>/dev/null; then
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 5 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-13 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-13 || true
fi

echo "GCC hiện tại: $(gcc --version | head -1)"
echo "G++ hiện tại: $(g++ --version | head -1)"

# -----------------------------------------------------------
# BƯỚC 2: Cài CUDA Toolkit 12.6 (chỉ toolkit, giữ nguyên driver)
# -----------------------------------------------------------
echo ""
echo "[2/6] Tải và cài CUDA Toolkit 12.6..."

# Kiểm tra xem đã có cuda 12.6 chưa
if [ -d "/usr/local/cuda-12.6" ]; then
    echo "CUDA 12.6 đã tồn tại, bỏ qua bước tải về."
else
    # Tải CUDA 12.6 local installer (chỉ toolkit)
    wget -c https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run \
        -O /tmp/cuda_12.6.run
    
    # Cài chỉ toolkit (không cài driver, không cài kernel module)
    sudo sh /tmp/cuda_12.6.run \
        --toolkit \
        --no-drm \
        --no-opengl-libs \
        --override \
        --silent
fi

# -----------------------------------------------------------
# BƯỚC 3: Cấu hình PATH cho CUDA 12.6
# -----------------------------------------------------------
echo ""
echo "[3/6] Cấu hình PATH..."

# Xóa symlink cuda cũ và tạo lại
sudo rm -f /usr/local/cuda
sudo ln -sf /usr/local/cuda-12.6 /usr/local/cuda

# Thêm vào bashrc nếu chưa có
CUDA_PATH_LINE='export PATH=/usr/local/cuda-12.6/bin:$PATH'
CUDA_LD_LINE='export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH'

if ! grep -q "cuda-12.6/bin" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# CUDA 12.6" >> ~/.bashrc
    echo "$CUDA_PATH_LINE" >> ~/.bashrc
    echo "$CUDA_LD_LINE" >> ~/.bashrc
    echo "export CUDA_HOME=/usr/local/cuda-12.6" >> ~/.bashrc
fi

export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.6

echo "NVCC hiện tại: $(nvcc --version | grep release)"

# -----------------------------------------------------------
# BƯỚC 4: Tạo lại virtualenv và cài PyTorch CUDA 12.6
# -----------------------------------------------------------
echo ""
echo "[4/6] Tạo lại virtual environment..."

cd /mnt/f/APPS_PJ/Mamba-SFT

# Xóa venv cũ
if [ -d "venv" ]; then
    echo "Xóa venv cũ..."
    rm -rf venv
fi

python3 -m venv venv
source venv/bin/activate

# Nâng cấp pip
pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------
# BƯỚC 5: Cài PyTorch với CUDA 12.6
# -----------------------------------------------------------
echo ""
echo "[5/6] Cài PyTorch 2.6.0 với CUDA 12.6..."

pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# Kiểm tra PyTorch
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# -----------------------------------------------------------
# BƯỚC 6: Cài mamba-ssm và các dependencies
# -----------------------------------------------------------
echo ""
echo "[6/6] Cài các package ML và mamba-ssm..."

# Cài causal-conv1d trước (dependency của mamba-ssm)
pip install causal-conv1d

# Cài mamba-ssm
pip install mamba-ssm --no-build-isolation

echo ""
echo "====== HOÀN TẤT ======"
echo "Kiểm tra cài đặt:"
python3 -c "
import torch
from mamba_ssm import Mamba
print('torch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('mamba_ssm: OK')

# Test nhanh Mamba
model = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(1, 32, 64).cuda()
y = model(x)
print(f'Mamba test OK - output shape: {y.shape}')
"

echo ""
echo "Để kích hoạt môi trường:"
echo "  source /mnt/f/APPS_PJ/Mamba-SFT/venv/bin/activate"
