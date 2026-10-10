import os
import sys
import time
import datetime
import numpy as np
import cv2
import psutil
import torch

# Đảm bảo import đúng config nội bộ của dự án
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import DEVICE, USE_GPU, LOGS_DIR
from core_pipeline.pipeline_manager import SafetyPipelineManagerOOP

def run_hardware_benchmark(total_frames_to_test=50):
    # Khởi tạo đường dẫn tệp nhật ký
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file_path = os.path.join(LOGS_DIR, "hardware_benchmark_latency.log")
    
    log_file_handle = open(log_file_path, "w", encoding="utf-8")

    def print_and_log(message_text):
        print(message_text)
        log_file_handle.write(message_text + "\n")
        log_file_handle.flush()

    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_and_log("=================================================================")
    print_and_log("NHẬT KÝ ĐO ĐẠC HIỆU NĂNG VÀ THỜI GIAN XỬ LÝ PHẦN CỨNG")
    print_and_log(f"Thời gian bắt đầu: {current_time_str}")
    print_and_log(f"Thiết bị tính toán: {DEVICE}")
    print_and_log(f"CUDA Khả dụng: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print_and_log(f"Tên GPU: {gpu_name}")
    else:
        print_and_log("Chế độ chạy: CPU")
    print_and_log("=================================================================\n")

    print_and_log("1. Đang khởi tạo Safety Pipeline Manager...")
    init_start_time = time.time()
    pipeline_manager = SafetyPipelineManagerOOP()
    init_duration = time.time() - init_start_time
    print_and_log(f"Khởi tạo Pipeline thành công trong: {init_duration:.2f} giây.\n")

    # Chuẩn bị khung hình kiểm thử
    sample_image_path = os.path.join(_project_root, "test", "test_weight", "anh1.jpg")
    if os.path.exists(sample_image_path):
        test_frame = cv2.imread(sample_image_path)
        print_and_log(f"Sử dụng ảnh test: {sample_image_path} ({test_frame.shape[1]}x{test_frame.shape[0]})")
    else:
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        print_and_log("Không tìm thấy ảnh mẫu, tự động tạo khung hình test 1280x720.")

    print_and_log("\n2. Đang làm nóng mô hình (Warm-up 5 khung hình)...")
    for _ in range(5):
        pipeline_manager.process_frame(test_frame)
    print_and_log("Làm nóng mô hình hoàn tất.\n")

    current_process = psutil.Process(os.getpid())
    process_cpu_percentages = []
    frame_durations_ms = []

    print_and_log(f"3. TIẾN HÀNH XỬ LÝ {total_frames_to_test} KHUNG HÌNH (IN VÀ GHI LOG TRỰC TIẾP):")
    print_and_log("-" * 75)
    print_and_log(f"{'Khung hình':<12} | {'Thời gian (ms)':<15} | {'FPS tức thời':<14} | {'RAM (MB)':<12} | {'CPU (%)':<10}")
    print_and_log("-" * 75)

    for frame_index in range(1, total_frames_to_test + 1):
        frame_start_time = time.perf_counter()
        
        # Xử lý toàn bộ pipeline AI
        annotated_frame, report_dictionary, new_events_list = pipeline_manager.process_frame(test_frame)
        
        frame_end_time = time.perf_counter()
        elapsed_ms = (frame_end_time - frame_start_time) * 1000.0
        frame_durations_ms.append(elapsed_ms)
        instant_fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0

        current_ram_mb = current_process.memory_info().rss / (1024 ** 2)
        current_cpu_percent = current_process.cpu_percent(interval=None)
        process_cpu_percentages.append(current_cpu_percent)

        log_line = (
            f"Frame {frame_index:02d}/{total_frames_to_test:02d}    | "
            f"{elapsed_ms:8.2f} ms     | "
            f"{instant_fps:6.1f} FPS      | "
            f"{current_ram_mb:7.1f} MB  | "
            f"{current_cpu_percent:5.1f}%"
        )
        print_and_log(log_line)

    print_and_log("-" * 75)

    # Thống kê tổng hợp
    total_system_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    system_ram_used_gb = psutil.virtual_memory().used / (1024 ** 3)
    system_ram_percent = psutil.virtual_memory().percent

    average_frame_latency_ms = sum(frame_durations_ms) / len(frame_durations_ms)
    min_frame_latency_ms = min(frame_durations_ms)
    max_frame_latency_ms = max(frame_durations_ms)
    average_fps = 1000.0 / average_frame_latency_ms if average_frame_latency_ms > 0 else 0.0

    valid_cpu_samples = [c for c in process_cpu_percentages if c > 0]
    average_process_cpu_percent = (sum(valid_cpu_samples) / len(valid_cpu_samples)) if len(valid_cpu_samples) > 0 else 0.0

    cuda_allocated_vram_mb = 0.0
    cuda_reserved_vram_mb = 0.0
    if torch.cuda.is_available():
        cuda_allocated_vram_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
        cuda_reserved_vram_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)

    print_and_log("\n=================================================================")
    print_and_log("BÁO CÁO TỔNG KẾT HIỆU NĂNG VÀ ĐỘ MƯỢT")
    print_and_log("=================================================================")
    print_and_log(f"1. TỐC ĐỘ VÀ ĐỘ TRỄ:")
    print_and_log(f"   - Tốc độ trung bình: {average_fps:.2f} FPS")
    print_and_log(f"   - Độ trễ trung bình: {average_frame_latency_ms:.2f} ms")
    print_and_log(f"   - Độ trễ thấp nhất:  {min_frame_latency_ms:.2f} ms")
    print_and_log(f"   - Độ trễ cao nhất:   {max_frame_latency_ms:.2f} ms")
    if average_fps >= 20.0:
        print_and_log("   - Đánh giá: RẤT MƯỢT (Đạt tiêu chuẩn thời gian thực >= 20 FPS)")
    elif average_fps >= 10.0:
        print_and_log("   - Đánh giá: KHÁ ỔN ĐỊNH (Phù hợp trên thiết bị nhúng)")
    else:
        print_and_log("   - Đánh giá: CHẬM (Cần cân nhắc tối ưu thêm)")

    print_and_log(f"\n2. TÀI NGUYÊN BỘ NHỚ RAM:")
    print_and_log(f"   - RAM tiến trình Python chiếm: {current_ram_mb:.2f} MB")
    print_and_log(f"   - Tổng RAM toàn máy: {system_ram_used_gb:.2f} GB / {total_system_ram_gb:.2f} GB ({system_ram_percent:.1f}%)")

    print_and_log(f"\n3. TẢI VI XỬ LÝ CPU:")
    print_and_log(f"   - CPU trung bình sử dụng: {average_process_cpu_percent:.2f}%")

    print_and_log(f"\n4. BỘ NHỚ ĐỒ HỌA GPU (VRAM):")
    if torch.cuda.is_available():
        print_and_log(f"   - VRAM phân bổ thực tế: {cuda_allocated_vram_mb:.2f} MB")
        print_and_log(f"   - VRAM bộ đệm dự phòng: {cuda_reserved_vram_mb:.2f} MB")
    else:
        print_and_log("   - Chạy trên CPU, không dùng GPU VRAM.")
    print_and_log("=================================================================")
    print_and_log(f"Nhật ký đo đạc đã được lưu thành công vào: {log_file_path}\n")

    log_file_handle.close()

if __name__ == "__main__":
    run_hardware_benchmark(total_frames_to_test=50)
