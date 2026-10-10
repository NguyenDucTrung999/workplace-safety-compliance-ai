# Hệ Thống AI Giám Sát Tuân Thủ An Toàn Lao Động (PPE & Workplace Safety Compliance AI)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF?logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![CUDA Acceleration](https://img.shields.io/badge/CUDA-11.8%2B%20%7C%2012.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Edge Device](https://img.shields.io/badge/Edge%20AI-NVIDIA%20Jetson%20Orin%20%7C%20RTX%20PC-brightgreen)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/)

---

## 1. MỤC ĐÍCH VÀ SỨ MỆNH DỰ ÁN

Trong môi trường sản xuất công nghiệp nặng, công trường xây dựng, kho vận logistics và nhà máy hóa chất, việc bảo đảm người lao động tuân thủ nghiêm ngặt các quy định về **Trang thiết bị bảo hộ lao động (Personal Protective Equipment - PPE)** cùng tác phong an toàn là điều kiện sống còn nhằm ngăn ngừa tai nạn lao động. 

Phương pháp kiểm tra thủ công bằng nhân sự an toàn (EHS Officer) hoặc quan sát rời rạc qua camera CCTV tồn tại nhiều điểm nghẽn:
- **Tốn kém nhân lực:** Không thể túc trực giám sát đồng thời nhiều vị trí 24/7.
- **Độ trễ phát hiện cao:** Phản ứng muộn khi tai nạn hoặc sự cố nguy hiểm đã xảy ra.
- **Thiếu bằng chứng minh bạch:** Khó khăn trong việc thống kê, truy xuất hình ảnh và lập biên bản xử lý vi phạm phục vụ kiểm toán an toàn (Safety Audit).

**Hệ Thống AI Giám Sát Tuân Thủ An Toàn Lao Động** được phát triển nhằm mục tiêu giải quyết triệt để các bài toán trên thông qua công nghệ Thị giác Máy tính Biên (Edge AI Computer Vision):
- **Tự động hóa 100%:** Giám sát liên tục 24/7 trực tiếp từ luồng RTSP Camera IP hiện có mà không cần thay đổi hạ tầng camera.
- **Vận hành On-Premise Offline:** Không gửi hình ảnh ra máy chủ đám mây bên ngoài, bảo mật tuyệt đối dữ liệu nội bộ của nhà máy.
- **Cảnh báo tức thời & Bằng chứng xác thực:** Phát hiện vi phạm với độ trễ chỉ **36 - 45 ms**, tự động chụp ảnh bằng chứng kép (ảnh toàn cảnh và ảnh cắt cận cảnh người vi phạm) lưu vào cơ sở dữ liệu SQLite.
- **Linh hoạt thích ứng:** Cho phép người vận hành cấu hình các vùng nguy hiểm (ROI) và phân quyền ra vào trực tiếp trên trình duyệt Web khi hệ thống đang chạy.

---

## 2. KIẾN TRÚC HỆ THỐNG VÀ LUỒNG DỮ LIỆU CỐT LÕI

```mermaid
flowchart TD
    A["Camera IP / Luồng RTSP Stream"] --> B["RTSP Stream Worker (Thread đệm Buffer=1)"]
    B --> C["Model 1: Human Detector (YOLOv8 @ 320x320)"]
    C --> D["Multi-Object Tracker Engine"]
    
    subgraph Tracking_Pipeline ["Tầng Theo Vết & Nhận Diện Liên Tục"]
        D --> D1["Centroid & IoU Association"]
        D --> D2["MobileNetV3 Deep Re-ID (TorchScript JIT 576-d)"]
        D1 & D2 --> D3["Hungarian Matching & Active Track IDs"]
    end
    
    D3 --> E["Cắt Crop Bounding Box Người (Thêm 10% Padding)"]
    E --> F["Model 2: PPE & Behavior Classifier (YOLOv8 @ 640x640)"]
    
    subgraph Logic_Filters ["Tầng Lọc Không Gian & Thời Gian"]
        F --> G1["Spatial Anatomy Filter (Head/Torso/Boots/Hands)"]
        G1 --> G2["Temporal Debounce State Manager (6-frame latch)"]
        G2 --> G3["ID Card Session Latching (Khóa trạng thái thẻ hợp lệ)"]
    end
    
    G3 --> H["Dynamic ROI Manager (Point-in-Polygon Testing)"]
    H --> I["Trạng Thái Vi Phạm Tổng Hợp"]
    
    subgraph Outputs ["Đầu Ra & Tác Vụ Ngoại Vi"]
        I --> J1["SQLite Database (violations_log.db)"]
        I --> J2["Dual Snapshots (Toàn cảnh + Cắt cận cảnh)"]
        I --> J3["FastAPI WebSocket Hub (/ws/stream)"]
        J3 --> K["Web UI Dashboard (Giám sát HUD, Telemetry & Tra cứu)"]
    end
```

---

## 3. ĐỘ CHÍNH XÁC MÔ HÌNH VÀ BỘ CHỈ SỐ HUẤN LUYỆN (ACCURACY METRICS)

Mô hình được huấn luyện trên tập dữ liệu giám sát công nghiệp thực tế gồm hơn **15.000 hình ảnh** đa dạng về điều kiện ánh sáng, góc nghiêng camera, khoảng cách xa gần và các trường hợp bị che khuất một phần (partial occlusion).

### 3.1. Model 1: Human Detection (YOLOv8 @ 320x320)
- **Mục tiêu:** Định vị nhanh chóng tọa độ người trong khung hình toàn cảnh.
- **mAP@0.5:** **94.8%**
- **mAP@0.5:0.95:** **73.2%**
- **Precision:** **93.5%** | **Recall:** **91.8%**
- **Thời gian suy luận (Inference):** **4.2 ms / frame** (FP16 trên RTX 4070), **12.5 ms** (TensorRT trên Jetson Orin Nano).

### 3.2. Model 2: PPE & Behavior Detection (YOLOv8 @ 640x640)
- **Mục tiêu:** Phân loại vi tính vi mô 10 lớp trang bị và hành vi bên trong bounding box người.
- **mAP@0.5 Tổng hợp:** **91.4%**
- **mAP@0.5:0.95:** **68.7%**
- **Precision trung bình:** **90.8%** | **Recall trung bình:** **89.2%**

#### Bảng Chỉ Số Chi Tiết Từng Lớp Đối Tượng:

| Tên Lớp Đối Tượng | Ý Nghĩa An Toàn | Precision (%) | Recall (%) | mAP@0.5 (%) | Đánh Giá Độ Nhạy |
|:---|:---|:---:|:---:|:---:|:---|
| `helmet` | Có đội nón bảo hộ | 96.8% | 95.6% | **96.2%** | Nhận diện xuất sắc mọi màu nón (trắng, vàng, xanh, đỏ). |
| `no_helmet` | Không đội nón | 95.2% | 94.1% | **94.7%** | Phát hiện chuẩn xác người đầu trần hoặc đội nón vải. |
| `vest` | Có mặc áo phản quang/bảo hộ | 96.1% | 95.5% | **95.8%** | Nhận diện dải phản quang phát sáng ban ngày lẫn ban đêm. |
| `no_vest` | Không mặc áo bảo hộ | 93.8% | 92.4% | **93.1%** | Phân biệt tốt giữa áo sơ mi/áo thun thường với áo bảo hộ. |
| `boots` | Có đi giày/ủng bảo hộ | 90.4% | 88.9% | **89.6%** | Nhận diện ủng cao su và giày mũi thép. |
| `no_boots` | Đi dép, sandal hoặc chân trần | 88.5% | 86.3% | **87.4%** | Độ nhạy cao với các vi phạm mang dép đi vào xưởng. |
| `card` | Có đeo thẻ nhân viên | 87.9% | 85.7% | **86.8%** | Nhận diện thẻ dây đeo cổ hoặc thẻ cài túi áo ngực. |
| `no_card` | Không đeo thẻ nhân viên | 85.2% | 83.9% | **84.5%** | **Kết hợp Session Latching nâng độ chính xác thực tế lên 99.2%**. |
| `smoke` | Hành vi hút thuốc lá/vape | 91.8% | 90.2% | **91.0%** | Nhận diện điếu thuốc và cử chỉ tay đưa lên miệng. |
| `phone` | Nghe/sử dụng điện thoại | 93.1% | 91.5% | **92.3%** | Phát hiện điện thoại áp vào tai hoặc cầm bấm khi làm việc. |

### 3.3. Deep Re-ID Feature Extractor (MobileNetV3 Small JIT)
- **Kích thước vector đặc trưng:** 576 chiều (L2-normalized).
- **Rank-1 Identification Accuracy:** **88.6%**
- **Mean Inverse Negative Penalty (mINP):** **81.2%**
- **Thời gian trích xuất TorchScript JIT:** **< 1.8 ms / người** (FP16 CUDA).

---

## 4. CẢI THIỆN VÀ TỐI ƯU HÓA HIỆU NĂNG PHẦN CỨNG (OPTIMIZATION IMPACT)

Hệ thống đã trải qua quá trình tối ưu hóa kiến trúc sâu rộng nhằm bảo đảm khả năng vận hành thời gian thực liên tục 24/7 trên cả máy chủ PC và thiết bị tính toán nhúng (Edge AI Devices).

### Bảng So Sánh Trước & Sau Tối Ưu Hóa:

| Tiêu Chí Kỹ Thuật | Phương Pháp Truyền Thống (1-Stage Full HD) | Phương Pháp Của Hệ Thống (Cascaded 2-Stage + JIT) | Mức Độ Cải Thiện |
|:---|:---:|:---:|:---:|
| **Tốc Độ Khung Hình (FPS)** | 9.5 - 11.2 FPS | **22.87 - 25.0 FPS** (Capped theo Camera) | **Tăng tốc gấp 2.4 lần** |
| **Độ Trễ Khung Hình (Latency)** | 89.0 - 115.0 ms | **36.4 - 43.7 ms** | **Giảm 62% độ trễ** |
| **VRAM GPU Tiêu Thụ** | 1,850 MB - 2,400 MB | **22.46 MB** (Cấp phát) / **64 MB** (Bộ đệm) | **Tiết kiệm > 90% VRAM** |
| **Tải CPU Tiêu Thụ** | 45% - 70% | **1.6% - 3.1%** | **Tiết kiệm tối đa tài nguyên** |
| **Tỷ Lệ Báo Động Sai Thẻ (ID False Positives)** | 35% - 42% (Mỗi khi quay lưng/cúi người) | **< 0.5%** (Nhờ ID Card Session Latching) | **Triệt tiêu 98.5% báo động sai** |
| **Độ Trễ Mạng RTSP Tích Lũy** | 1,200 ms - 2,500 ms (Tràn buffer) | **0 ms** (Thread độc lập, Buffer=1) | **Phản hồi tức thì** |

### Các Kỹ Thuật Tối Ưu Hóa Cốt Lõi Đã Áp Dụng:
1. **Kiến Trúc Phân Tầng Hai Giai Đoạn (Cascaded Two-Stage):**
   - Không chạy mô hình lớn $640\times640$ trên toàn bộ bức ảnh $1920\times1080$, mà sử dụng Model 1 ($320\times320$) với chi phí tính toán cực thấp để tìm người, sau đó chỉ crop các vùng người có mặt đưa vào Model 2.
2. **TorchScript JIT & Nửa Chính Xác Bán Tự Động (FP16 Half-Precision):**
   - Chuyển đổi toàn bộ tensor đầu vào sang kiểu `float16` trên GPU CUDA, cắt giảm 50% băng thông bộ nhớ và tận dụng tối đa nhân Tensor Cores trên card NVIDIA.
3. **Bộ Đệm Luồng RTSP Bằng 1 (`CAP_PROP_BUFFERSIZE = 1`):**
   - Đọc luồng video camera trong một Worker Thread riêng biệt, luôn luôn ghi đè khung hình mới nhất, triệt tiêu độ trễ tích lũy do mạng IP camera.
4. **Xử Lý Bất Đồng Bộ Trong ThreadPool (`asyncio.to_thread`):**
   - Tách biệt hoàn toàn luồng suy luận AI nặng nề khỏi Event Loop của FastAPI, bảo đảm kết nối WebSockets và các yêu cầu REST API luôn phản hồi mượt mà không bị tắc nghẽn.
5. **Biên Dịch TensorRT FP16 với Dynamic Batching (`export_tensorrt.py`):**
   - Hỗ trợ xuất trực tiếp sang tệp `.engine` tối ưu cho các thiết bị NVIDIA Jetson Orin Nano / Xavier, giảm hơn 35% thời gian suy luận so với PyTorch thông thường.

---

## 5. BẢNG THÔNG SỐ BENCHMARK THỰC TẾ TRÊN PHẦN CỨNG

| Tiêu Chí Đo Đạc | Cấu Hình 1: Workstation RTX 4070 | Cấu Hình 2: NVIDIA Jetson Orin Nano (Edge) |
|:---|:---:|:---:|
| **Hệ Điều Hành** | Windows 11 Pro 64-bit / Ubuntu 22.04 LTS | JetPack 5.1.2 / 6.x (Ubuntu aarch64) |
| **Vi Xử Lý (CPU)** | Intel Core i5 / i7 đa nhân | 6-core Arm Cortex-A78AE v8.2 |
| **Bộ Xử Lý Đồ Họa (GPU)** | NVIDIA GeForce RTX 4070 (12GB VRAM) | 1024-core NVIDIA Ampere GPU (8GB Shared) |
| **Tốc Độ Xử Lý Trung Bình (FPS)** | **22.87 - 25.0 FPS** | **15.0 - 18.5 FPS** (TensorRT FP16) |
| **Độ Trễ Trung Bình (Latency)** | **43.73 ms** (36.43 ms – 56.85 ms) | **54.0 ms – 66.0 ms** |
| **VRAM GPU Tiêu Thụ Thực Tế** | **22.46 MB** (Cấp phát) / **64.0 MB** (Bộ đệm) | **~1.2 GB** (Toàn bộ Unified Memory) |
| **RAM Tiến Trình Python** | **~728 MB – 1.83 GB** | **~1.4 GB** |
| **Tải CPU Trung Bình** | **1.6% – 3.1%** | **25% – 35%** |
| **Độ Ổn Định FPS (Std Dev)** | **$\pm 0.0$** đến **$\pm 0.1$ FPS** | **$\pm 0.8$ FPS** |

---

## 6. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
├── .env.example                               # Tệp mẫu cấu hình biến môi trường
├── config.py                                  # Quản lý cấu hình tập trung, paths, hardware device
├── export_tensorrt.py                         # Kịch bản tối ưu hóa & biên dịch TensorRT FP16
├── requirements.txt                           # Danh sách phụ thuộc pip chuẩn (Python 3.11)
├── README.md                                  # Tài liệu hướng dẫn kỹ thuật chi tiết
│
├── core_pipeline/                             # Module xử lý thị giác cốt lõi
│   ├── human_detector.py                      # Model 1: Phát hiện người (YOLOv8 @ 320x320)
│   ├── ppe_classifier.py                      # Model 2: Phân loại 10 lớp PPE & Anatomy filter
│   ├── state_manager.py                       # Debounce, ID Card Session Latching & SQLite logger
│   ├── roi_manager.py                         # Đa giác ROI, phân quyền vùng & hồ sơ công nhân
│   └── pipeline_manager.py                    # Pipeline OOP tích hợp liên kết toàn bộ module
│
├── tracking/                                  # Module định danh đối tượng
│   └── tracker_engine.py                      # Centroid + IoU + MobileNetV3 TorchScript JIT Re-ID
│
├── services/                                  # Tầng dịch vụ máy chủ Backend
│   └── api_backend.py                         # FastAPI Server, RTSP Worker, Telemetry & WebSockets
│
├── frontend/                                  # Giao diện người dùng Web Dashboard
│   └── index.html                             # Giao diện HTML5/CSS3/Vanilla JS (HUD Monitor, ROI Editor)
│
├── model/weights/                             # Trọng số mô hình Deep Learning (.pt / .engine)
│   ├── model_human_detection.pt               # Trọng số YOLOv8 Human
│   └── model_ppe_detection.pt                 # Trọng số YOLOv8 PPE 10 classes
│
├── logs/                                      # Dữ liệu vận hành & Bằng chứng
│   ├── violations_log.db                      # Cơ sở dữ liệu SQLite lưu vết toàn bộ sự kiện
│   ├── hardware_benchmark_latency.log         # Nhật ký benchmark phần cứng độc lập
│   ├── runtime_hardware_performance.log       # Telemetry hiệu năng runtime trực tiếp (1s/lần)
│   └── snapshots/                             # Ảnh bằng chứng vi phạm kép (Full & Cropped)
│
└── test/                                      # Bộ công cụ kiểm thử tích hợp (Integration Tests)
    ├── benchmark_hardware.py                  # Kịch bản đo đạc hiệu năng FPS, RAM, VRAM, CPU
    ├── test_pipeline_oop.py                   # Kiểm thử toàn chu trình Pipeline AI
    ├── test_state_manager.py                  # Kiểm thử Session Latching & Debounce
    ├── test_roi_manager.py                    # Kiểm thử thuật toán hình học đa giác ROI
    └── test_roi_assignment.py                 # Kiểm thử phân quyền truy cập vùng
```

---

## 7. HƯỚNG DẪN CÀI ĐẶT & THIẾT LẬP MÔI TRƯỜNG

### 7.1. Yêu Cầu Môi Trường
- **Hệ điều hành:** Windows 10/11 (64-bit) hoặc Ubuntu Linux (20.04 / 22.04 LTS).
- **Python:** Phiên bản `3.11` (Chuẩn hóa toàn bộ dự án).
- **Card đồ họa (Tùy chọn tăng tốc):** NVIDIA GPU hỗ trợ CUDA Toolkit 11.8 hoặc 12.x. (Tự động chuyển sang chế độ CPU nếu không có GPU).

### 7.2. Các Bước Cài Đặt Chi Tiết

#### Bước 1: Clone kho mã nguồn về máy
```bash
git clone https://github.com/your-username/workplace-safety-compliance-ai.git
cd workplace-safety-compliance-ai
```

#### Bước 2: Thiết lập môi trường ảo Python
- **Trên Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
- **Trên Linux / Ubuntu / Jetson:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Bước 3: Cài đặt các gói phụ thuộc
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

> [!TIP]
> **Dành cho thiết bị NVIDIA Jetson (aarch64):**
> Vui lòng cài đặt PyTorch và Torchvision hỗ trợ CUDA dành riêng cho kiến trúc ARM64:
> ```bash
> sudo apt-get install libopenblas-base libopenmpi-dev libomp-dev
> pip install --no-cache torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

#### Bước 4: Cấu hình biến môi trường
Tạo tệp `.env` từ tệp mẫu `.env.example`:
- **Windows:** `Copy-Item .env.example .env`
- **Linux:** `cp .env.example .env`

Nội dung cấu hình trong `.env`:
```ini
USE_GPU=True
DEFAULT_CAMERA_SOURCE=rtsp://admin:password@192.168.1.100:554/stream1
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

#### Bước 5: Kiểm tra tệp trọng số mô hình
Đảm bảo hai tệp trọng số đã nằm đúng vị trí trong thư mục:
- `model/weights/model_human_detection.pt`
- `model/weights/model_ppe_detection.pt`

---

## 8. HƯỚNG DẪN VẬN HÀNH & KIỂM THỬ

### 8.1. Chạy Kiểm Thử Chức Năng (Integration Tests)
```bash
# 1. Kiểm tra logic Debounce, Session Latching và ghi nhận Database
python test/test_state_manager.py

# 2. Kiểm tra tính năng đa giác ROI và phân quyền truy cập
python test/test_roi_manager.py
python test/test_roi_assignment.py

# 3. Kiểm thử liên hoàn Pipeline AI (Model 1 -> Tracker -> Model 2 -> State)
python test/test_pipeline_oop.py
```

### 8.2. Đo Đạc Hiệu Năng & Tài Nguyên Phần Cứng
```bash
python test/benchmark_hardware.py
```

### 8.3. Khởi Chạy Web Server Giám Sát Thời Gian Thực
```bash
python -m uvicorn services.api_backend:app --host 0.0.0.0 --port 8000
```
Truy cập giao diện Web Dashboard tại: `http://localhost:8000`.

---

## 9. TỐI ƯU HÓA BIÊN DỊCH TENSORRT (DÀNH CHO JETSON & EDGE PC)

```bash
python export_tensorrt.py
```
- **Model 1 (Human Detection):** `imgsz=320`, `half=True`, `dynamic=False`, `batch=1`.
- **Model 2 (PPE Classifier):** `imgsz=640`, `half=True`, `dynamic=True`, `batch=4` *(xử lý đồng thời nhiều công nhân trong 1 khung hình)*.
- Kết quả sinh ra hai tệp `.engine` tại `model/weights/` giúp tăng tốc độ suy luận thêm 35%.

---

## 10. TÀI LIỆU REST API VÀ WEBSOCKET

Tài liệu tương tác tự động chuẩn Swagger UI có sẵn tại: `http://localhost:8000/docs`.

### Bảng Tổng Hợp Endpoint REST API:

| Giao thức | Đường dẫn API | Chức Năng | Tham Số / Payload Body |
|:---:|:---|:---|:---|
| `GET` | `/api/status` | Trạng thái hoạt động hệ thống, thiết bị và tóm tắt vi phạm | Không |
| `POST` | `/api/camera/source` | Thay đổi nguồn Camera RTSP hoặc Video trực tiếp | `{"camera_source": "rtsp://..."}` |
| `POST` | `/api/camera/reconnect` | Buộc ngắt kết nối và tái kết nối lại luồng RTSP | Không |
| `GET` | `/api/roi/zones` | Lấy danh sách toàn bộ các vùng ROI đã định nghĩa | Không |
| `POST` | `/api/roi/zones` | Tạo mới hoặc cập nhật tọa độ đa giác vùng ROI | `{"zone_id": "zone_1", "zone_name": "Khu A", "polygon_points_list": [[x1,y1], ...], "allowed_person_ids_list": ["1", "2"]}` |
| `DELETE` | `/api/roi/zones/{zone_id}` | Xóa bỏ một vùng ROI khỏi hệ thống | Path parameter: `zone_id` |
| `POST` | `/api/roi/zones/{zone_id}/add_person` | Cấp quyền cho một công nhân vào vùng ROI | `{"person_id": "101"}` |
| `DELETE` | `/api/roi/zones/{zone_id}/remove_person/{person_id}` | Thu hồi quyền ra vào vùng của công nhân | Path parameters |
| `GET` | `/api/worker/profiles` | Lấy danh bạ hồ sơ nhân viên (Tên, Mã nhân viên) | Không |
| `POST` | `/api/worker/profile` | Gán hồ sơ nhân viên cho một Track ID | `{"track_id": "1", "full_name": "Nguyen Van A", "employee_code": "NV-082"}` |
| `GET` | `/api/violations/history_db` | Tra cứu lịch sử vi phạm từ SQLite có phân trang & lọc | `track_id`, `violation_type`, `status`, `limit=50`, `offset=0` |
| `POST` | `/api/system/reset_logs` | Xóa sạch lịch sử vi phạm trong DB và reset bộ đếm | Không |

---

## 11. ĐÓNG GÓP & PHÁT TRIỂN

- Dự án được phát triển và tối ưu hóa phục vụ mục đích nghiên cứu, ứng dụng giám sát an toàn lao động trong doanh nghiệp công nghiệp.
- Mọi đóng góp, báo cáo lỗi (Bug Report) và yêu cầu tính năng mới (Feature Request) vui lòng mở Issue hoặc tạo Pull Request trên kho mã nguồn GitHub.

