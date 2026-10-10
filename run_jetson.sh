#!/bin/bash
# ==============================================================================
# SCRIPT KHỞI ĐỘNG HỆ THỐNG GIÁM SÁT AN TOÀN LAO ĐỘNG TRÊN JETSON ORIN NANO
# ==============================================================================

set -e

# Xác định đường dẫn thư mục gốc của dự án động
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

echo "=============================================================================="
echo "KHỞI ĐỘNG HỆ THỐNG AI VISION GIÁM SÁT AN TOÀN LAO ĐỘNG (JETSON ORIN NANO)"
echo "Thư mục làm việc: ${PROJECT_DIR}"
echo "=============================================================================="

# 1. Kiểm tra Python
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "Lỗi: Không tìm thấy Python trên hệ thống!"
    exit 1
fi

# 2. Kiểm tra phần cứng CUDA trên Jetson
echo "--- KIỂM TRA PHẦN CỨNG VÀ CUDA ---"
${PYTHON_CMD} -c "
import torch
print('Phiên bản PyTorch:', torch.__version__)
print('CUDA khả dụng (GPU Jetson):', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Thiết bị GPU:', torch.cuda.get_device_name(0))
else:
    print('CẢNH BÁO: Không phát hiện GPU CUDA, hệ thống sẽ chạy ở chế độ CPU!')
"

# 3. Tạo các thư mục cần thiết nếu chưa tồn tại
mkdir -p logs/snapshots
mkdir -p model/weights

# 4. Kiểm tra sự tồn tại của các file trọng số
if [ ! -f "model/weights/model_human_detection.pt" ]; then
    echo "CẢNH BÁO: Chưa tìm thấy file trọng số model_human_detection.pt trong model/weights/"
fi

if [ ! -f "model/weights/model_ppe_detection.pt" ]; then
    echo "CẢNH BÁO: Chưa tìm thấy file trọng số model_ppe_detection.pt trong model/weights/"
fi

# 5. Khởi chạy Web Server FastAPI & AI Pipeline
echo "=============================================================================="
echo "BẮT ĐẦU KHỞI CHẠY DỊCH VỤ WEB BACKEND VÀ PIPELINE XỬ LÝ..."
echo "Địa chỉ truy cập Web UI: http://0.0.0.0:8000 (hoặc http://<IP_JETSON>:8000)"
echo "Nhấn Ctrl + C để dừng dịch vụ."
echo "=============================================================================="

exec ${PYTHON_CMD} -m uvicorn services.api_backend:app --host 0.0.0.0 --port 8000
