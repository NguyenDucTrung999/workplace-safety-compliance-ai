# KẾ HOẠCH KIẾN TRÚC & QUY CHUẨN NGHIỆP VỤ HỆ THỐNG GIÁM SÁT AN TOÀN LAO ĐỘNG (JETSON ORIN NANO)

## I. SƠ ĐỒ LUỒNG DỮ LIỆU TỔNG THỂ (8 BƯỚC PIPELINE)

```
[Luồng RTSP Camera] 
        │
        ▼
[1. RTSPCameraStreamWorker] (Thread đọc khung hình độc lập)
        │
        ▼
[2. Model 1 HumanDetectorOOP] (YOLOv8 + ByteTrack SOTA)
        │
        ▼
[3. TrackerEngineOOP] (Hybrid IoU + Memory Buffer 60-90 frames)
        │
        ▼
[4. Crop & 10% Expansion Padding] (Nới rộng khung cắt 10% các hướng)
        │
        ▼
[5. Model 2 PPEClassifierOOP] (YOLOv8 10-Class PPE)
        │
        ▼
[6. ROIManagerOOP] (Kiểm tra xâm nhập theo tọa độ chân & White-list)
        │
        ▼
[7. SafetyStateManagerOOP] (Máy trạng thái FSM: Khai báo & Khắc phục vi phạm)
        │
        ▼
[8. Bounding Box Rendering & WebSocket Base64] (Vẽ khung Vàng/Xanh & Stream qua Web)
```

---

## II. CHI TIẾT YÊU CẦU NGHIỆP VỤ & KỸ THUẬT THEO TỪNG BƯỚC

### BƯỚC 1: QUẢN LÝ LUỒNG ĐỌC CAMERA RTSP BẤT ĐỒNG BỘ (`RTSPCameraStreamWorker`)
- Khởi chạy một Tiến trình độc lập (Daemon Thread) đọc liên tục các khung hình từ nguồn RTSP `rtsp://admin:vna%40123456@192.168.110.200:554/media/video2` (hoặc Webcam `0` / Ảnh tĩnh `anh1.jpg`).
- Sử dụng cơ chế khóa Thread Lock (`threading.Lock()`) bảo vệ biến `latest_frame` tránh hiện tượng tranh chấp tài nguyên (Race Condition).
- Hỗ trợ đổi nguồn camera động thời gian thực qua API `POST /api/camera/source` và Reset đường truyền qua API `POST /api/camera/reconnect`.
- Cơ chế tự động kết nối lại (Auto Reconnect) sau mỗi 1.0 giây nếu đường truyền mạng RTSP bị rớt.

### BƯỚC 2: PHÁT HIỆN ĐỐI TƯỢNG NGƯỜI BẰNG MODEL 1 (`HumanDetectorOOP`)
- Tải trọng số Model 1 `model_human_detection.pt` chuẩn YOLOv8.
- Kích thước ảnh đầu vào `imgsz=1280`, Ngưỡng tin cậy `conf=0.5`.
- Tích hợp công cụ ByteTrack SOTA (`tracker="bytetrack.yaml"`, `persist=True`) trực tiếp trong `self.model.track()`.
- Trả về danh sách đối tượng người dưới dạng `[x_min, y_min, x_max, y_max, box_confidence, byte_track_id]`.

### BƯỚC 3: THEO VẾT ĐỐI TƯỢNG ĐA MỤC TIÊU HYBRID (`TrackerEngineOOP`)
- Áp dụng Thuật toán Theo vết 4 cấp độ:
  1. Khớp theo ByteTrack ID mapping thu được từ Model 1.
  2. Khớp theo Ma trận diện tích đè IoU Overlap với ngưỡng `iou_threshold >= 0.30`.
  3. Khớp bổ trợ theo Khoảng cách tâm Euclid `distance_threshold < 120.0` pixels.
  4. Lưu ký ức vết dài hạn `max_disappeared_frames = 60` frames (~2-3 giây).
- Giữ nguyên ID cố định cho từng công nhân ngay cả khi họ di chuyển nhanh, bị rớt frame RTSP, hoặc tạm thời bị khuất bóng.
- Bộ đếm ID `next_track_id` duy nhất và tăng dần từ 1.

### BƯỚC 4: CẮT ẢNH CÔNG NHÂN NỚI RỘNG 10% PADDING (`crop_person_with_padding`)
- Tính toán `box_width = x_max - x_min` và `box_height = y_max - y_min`.
- Nới rộng khung cắt thêm 10% mỗi chiều: `padding_x = box_width * 0.10`, `padding_y = box_height * 0.10`.
- Kẹp biên an toàn trong phạm vi chiều rộng và chiều cao ảnh gốc (`max(0, ...)` và `min(image_size, ...)`).
- Đảm bảo ảnh cắt thu đủ vùng đầu (chứa mũ bảo hộ) và vùng sát đất (chứa giày bảo hộ).

### BƯỚC 5: PHÂN LOẠI 10 NHÃN TRANG THIẾT BỊ BẢO HỘ BẰNG MODEL 2 (`PPEClassifierOOP`)
- Tải trọng số Model 2 `model_ppe_detection.pt` chuẩn YOLOv8 với ngưỡng `conf=0.5`.
- Nhận diện 10 nhãn bảo hộ và hành vi: `helmet on`, `no helmet`, `vest`, `no vest`, `boots`, `no boots`, `id_card`, `no id_card`, `smoking`, `phone_usage`.
- Trả về danh sách chi tiết các nhãn bảo hộ phát hiện được trên vùng ảnh cắt của từng công nhân.

### BƯỚC 6: ĐÁNH GIÁ VÙNG ROI XÂM NHẬP TRÁI PHÉP (`ROIManagerOOP`)
- Xác định vị trí điểm đứng thực tế của công nhân theo tâm chân: `foot_x_center = (x_min + x_max) / 2.0`, `foot_y_bottom = y_max`.
- Kiểm tra điểm đứng `(foot_x_center, foot_y_bottom)` nằm trong đa giác ROI bằng `cv2.pointPolygonTest(...) >= 0`.
- Kiểm tra điều kiện Danh sách trắng `allowed_person_ids_list`. Nếu công nhân đứng trong vùng mà ID không thuộc danh sách cho phép, kích hoạt vi phạm `UNAUTHORIZED_INTRUSION`.

### BƯỚC 7: QUẢN LÝ VÒNG ĐỜI VI PHẠM BẰNG MÁY TRẠNG THÁI FSM (`SafetyStateManagerOOP`)
- **Quy tắc Bằng chứng Thực tế 100% (Strict Explicit Evidence Rule)**: Chỉ báo vi phạm khi Model 2 trả về nhãn `no_...` rõ ràng (`no helmet`, `no vest`, `no boots`, `no id_card`, `smoking`, `phone_usage` hoặc xâm nhập vùng ROI). Tuyệt đối không tự ý gán vi phạm giả cho thẻ `id_card` khi không soi thấy thẻ. Nếu không có nhãn vi phạm, công nhân được đánh giá là AN TOÀN (`[SAFE]`).
- Quản lý 6 loại vi phạm chính: `no_helmet`, `no_vest`, `no_boots`, `no_id_card`, `smoking`, `phone_usage`.
- Quản lý vòng đời trạng thái FSM cho từng công nhân: Ghi nhận thời điểm bắt đầu vi phạm (`start_timestamp`), giữ trạng thái `ACTIVE`, và tự động chuyển sang `RESOLVED` khi công nhân khắc phục.

### BƯỚC 8: TRỰC QUAN HÓA KHUNG HÌNH & HIỂN THỊ BẢNG AN TOÀN THEO CÔNG NHÂN (`api_backend.py` & `index.html`)
- Công nhân có vi phạm: Vẽ khung màu VÀNG BGR `(0, 255, 255)` với nhãn xếp chồng `ID:X [VIOLATION: type1, type2]`.
- Công nhân tuân thủ an toàn: Vẽ khung màu XANH LÁ BGR `(0, 255, 0)` với nhãn `ID:X [SAFE]`.
- Bảng Giám sát An toàn hiển thị theo cơ chế **Person-Centric State Logging (Một dòng duy nhất cho mỗi Công nhân)**:
  - Hệ thống nhận diện Công nhân trước ở Bước 1 & 2.
  - Mỗi Công nhân sở hữu **ĐÚNG 1 DÒNG DUY NHẤT** trên Bảng Giám sát đại diện cho ID của họ.
  - Khi Công nhân thay đổi trạng thái (vi phạm mới, vi phạm gộp, hoặc khắc phục xong), hệ thống **CẬP NHẬT TRỰC TIẾP NỘI DUNG NẰM TRONG DÒNG/THẺ ĐÓ** (Thời gian mới nhất, Chuỗi vi phạm, Badge trạng thái `ĐANG VI PHẠM` màu Đỏ / `AN TOÀN` màu Xanh).
  - Không sinh ra các dòng rác trùng lặp liên tục mỗi khi thời gian nhảy.
