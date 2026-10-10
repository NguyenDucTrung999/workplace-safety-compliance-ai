import os
import torch
# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN DỰ ÁN (RELATIVE PATH DỰA TRÊN THƯ MỤC GỐC)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR

def load_env_file_pure_python(env_filepath):
    if not os.path.exists(env_filepath):
        return
    try:
        with open(env_filepath, "r", encoding="utf-8") as f:
            for line in f:
                trimmed_line = line.strip()
                if not trimmed_line or trimmed_line.startswith("#"):
                    continue
                if "=" in trimmed_line:
                    key_part, val_part = trimmed_line.split("=", 1)
                    key = key_part.strip()
                    val = val_part.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception:
        pass

# Nạp các biến môi trường từ tệp .env bằng Python thuần
load_env_file_pure_python(os.path.join(BASE_DIR, ".env"))

# ==============================================================================
# CẤU HÌNH THIẾT BỊ PHẦN CỨNG (GPU / CPU)
# ==============================================================================
raw_use_gpu_str = os.getenv("USE_GPU", "True").strip().lower()
USE_GPU = raw_use_gpu_str in ["true", "1", "yes"]
DEVICE = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')

# Tối ưu hóa CUDNN Benchmark & Đa luồng CPU cấp ứng dụng (Không cần quyền sudo)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
else:
    no_cuda_benchmark_flag = False

import cv2
cpu_core_count = os.cpu_count()
if cpu_core_count is not None:
    num_threads_str = str(cpu_core_count)
    os.environ["OMP_NUM_THREADS"] = num_threads_str
    os.environ["MKL_NUM_THREADS"] = num_threads_str
    os.environ["OPENBLAS_NUM_THREADS"] = num_threads_str
    cv2.setNumThreads(cpu_core_count)
    torch.set_num_threads(cpu_core_count)
else:
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    cv2.setNumThreads(4)
    torch.set_num_threads(4)

# Thư mục trọng số mô hình
WEIGHTS_DIR = os.path.join(BASE_DIR, "model", "weights")
HUMAN_MODEL_PATH = os.path.join(WEIGHTS_DIR, "model_human_detection.pt")
PPE_MODEL_PATH = os.path.join(WEIGHTS_DIR, "model_ppe_detection.pt")

# Thư mục nhật ký và cơ sở dữ liệu
LOGS_DIR = os.path.join(BASE_DIR, "logs")
SNAPSHOTS_DIR = os.path.join(LOGS_DIR, "snapshots")
DB_PATH = os.path.join(LOGS_DIR, "violations_log.db")

# ==============================================================================
# CẤU HÌNH NGUỒN VIDEO CAMERA RTSP VÀ MÁY CHỦ WEB
# ==============================================================================
DEFAULT_CAMERA_SOURCE = os.getenv(
    "DEFAULT_CAMERA_SOURCE", 
    "rtsp://admin:vna%40123456@192.168.110.200:554/media/video2"
)
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

