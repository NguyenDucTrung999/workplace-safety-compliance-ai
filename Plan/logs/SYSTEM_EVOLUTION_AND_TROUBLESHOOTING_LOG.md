# NHẬT KÝ KIẾN TRÚC HỆ THỐNG VÀ LỊCH SỬ XỬ LÝ SỰ CỐ
# (SYSTEM EVOLUTION & TROUBLESHOOTING LOG)

Hệ thống: Giám sát tuân thủ an toàn lao động bằng Thị giác máy tính (Computer Vision)
Cập nhật lần cuối: 2026-10-05

---

## MỤC LỤC
1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG
2. LỊCH SỬ CÁC BÀI TOÁN KỸ THUẬT VÀ HƯỚNG SUY LUẬN XỬ LÝ (GIAI ĐOẠN 1)
   2.1. Suy luận thiếu thẻ nhân viên khi mô hình không có nhãn âm
   2.2. Xung đột lọc không gian giải phẫu cơ thể (Spatial Anatomy Filtering)
   2.3. Sự cố nhảy ID liên tục do bộ đệm ngắn của ByteTrack mặc định
   2.4. Cơ chế chốt trạng thái vi phạm bám đuổi (State Latching)
   2.5. Sự khác biệt giữa bảng trạng thái trực tiếp (Live Table) và Cơ sở dữ liệu (SQLite)
3. THIẾT KẾ GIẢI PHÁP NÂNG CẤP CHẤT LƯỢNG VẬN HÀNH (GIAI ĐOẠN 2 - QoL)
   3.1. Lưu trữ bằng chứng vi phạm kép (Full Frame + Cropped Person Snapshot)
   3.2. Tra cứu lịch sử vi phạm từ Database và Xuất báo cáo
4. THIẾT KẾ GIẢI PHÁP TỐI ƯU HIỆU SUẤT HỆ THỐNG (GIAI ĐOẠN 3 - PERFORMANCE)
   4.1. Giải phóng Event Loop trong FastAPI bằng Async / ThreadPool
   4.2. Cơ chế đồng bộ thời gian thực và loại bỏ trễ khung hình (Frame Dropping)

---

## 1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG

Hệ thống hoạt động theo mô hình Pipeline phân tầng:
- **Tầng 1 - Phát hiện người (Human Detection):** Sử dụng YOLOv8 phát hiện tọa độ người trong khung hình camera.
- **Tầng 2 - Định danh và theo vết (Object Tracking & Deep Re-ID):** Kết hợp thuật toán tính toán độ trùng khớp hình học (IoU), so khớp tâm (Centroid) và trích xuất vector đặc trưng ngoại hình sâu (Deep Re-ID bằng MobileNetV3) để duy trì Track ID ổn định.
- **Tầng 3 - Phân loại trang bị bảo hộ (PPE Classification):** Sử dụng mô hình YOLOv8 thứ hai để nhận diện các trang bị bảo hộ và hành vi (helmet on, no helmet, vest, no vest, boots, no boots, id_card, smoking, phone_usage).
- **Tầng 4 - Quản lý trạng thái và luật an toàn (State Manager):** Áp dụng luật giải phẫu không gian, cửa sổ trượt làm mịn dữ liệu và cơ chế chốt trạng thái (State Latching).
- **Tầng 5 - Vùng kiểm soát an toàn (ROI Intrusion):** Kiểm tra xâm nhập vùng cấm dựa trên thuật toán đa giác hình học.
- **Tầng 6 - Lưu trữ và Giao tiếp (Storage & Delivery):** Lưu trữ sự kiện vào SQLite Database (`logs/violations_log.db`), đẩy luồng hình ảnh và trạng thái về Frontend qua WebSockets và FastAPI.

---

## 2. LỊCH SỬ CÁC BÀI TOÁN KỸ THUẬT VÀ HƯỚNG SUY LUẬN XỬ LÝ (GIAI ĐOẠN 1)

### 2.1. Suy luận thiếu thẻ nhân viên khi mô hình không có nhãn âm

- **Bối cảnh thực tế:** File cấu hình dataset ban đầu có khai báo class `no id_card`, tuy nhiên trong thực tế tập dữ liệu huấn luyện không có mẫu cho nhãn này. Mô hình YOLO chỉ có thể phát hiện sự hiện diện của `id_card`.
- **Vấn đề phát sinh:** Không thể dựa vào việc mô hình trả về nhãn `no id_card` để kết luận vi phạm. Nếu công nhân không đeo thẻ, mô hình sẽ không trả về nhãn nào liên quan đến thẻ.
- **Hướng suy luận xử lý:**
  - Áp dụng nguyên lý suy luận phủ định dựa trên quan sát liên tục (Negative inference by continuous observation).
  - Sử dụng biến đếm số khung hình quan sát rõ cơ thể (`full_body_observed_frames`).
  - Thiết lập thời gian ân hạn (grace period) là 8 khung hình (tương ứng kích thước cửa sổ trượt). Nếu sau 8 khung hình quan sát mà không xuất hiện bất kỳ nhãn nào trong danh sách hợp lệ (`id_card`, `id card`, `IDcard`, `ID_card`), hệ thống mới chính thức kết luận vi phạm `no_id_card`.
  - Nếu đã phát hiện thẻ ở bất kỳ thời điểm nào, trạng thái lập tức chuyển sang an toàn (`compliant`).

### 2.2. Xung đột lọc không gian giải phẫu cơ thể (Spatial Anatomy Filtering)

- **Bối cảnh thực tế:** Khi chạy mô hình PPE, xuất hiện các trường hợp nhận diện nhầm (false positives) ở vị trí bất hợp lý trên cơ thể (ví dụ: phát hiện giày ở phần đầu hoặc phát hiện mũ ở phần chân).
- **Vấn đề phát sinh:** Hệ thống đã đặt bộ lọc giải phẫu để giới hạn vị trí xuất hiện của từng trang bị theo trục dọc tương đối (từ 0.0 ở đỉnh đầu đến 1.0 ở gót chân). Trong cấu hình ban đầu, hai nhãn hành vi `phone_usage` (dùng điện thoại) và `smoking` (hút thuốc) bị xếp nhầm vào nhóm thân (`relative_y_center` từ 0.10 đến 0.85). Khi công nhân đưa điện thoại lên tai hoặc đưa thuốc lá lên miệng, tọa độ trung tâm của đối tượng rơi vào vùng đầu (`< 0.10`), dẫn đến việc hệ thống tự động loại bỏ phát hiện này vì coi là sai vị trí giải phẫu.
- **Hướng suy luận xử lý:**
  - Tách `phone_usage` và `smoking` thành nhóm hành vi tay độc lập.
  - Cho phép hành vi này xuất hiện từ đỉnh đầu đến ngang hông (dải tọa độ từ 0.0 đến 0.85).
  - Tinh chỉnh ngưỡng tin cậy của 2 nhãn này xuống 0.25 trong bộ phân loại do kích thước vật thể nhỏ, giúp hệ thống bắt nhạy ngay khi đưa tay lên vùng mặt/tai.

### 2.3. Sự cố nhảy ID liên tục do bộ đệm ngắn của ByteTrack mặc định

- **Bối cảnh thực tế:** Người dùng quan sát thấy số ID của công nhân bị tăng vọt (ví dụ: đang là Người 9 nhảy sang Người 13 chỉ sau hơn 20 giây), đồng thời nút reset nhật ký không đưa số đếm về 1.
- **Nguyên nhân gốc rễ:**
  - Mô hình phát hiện người ban đầu gọi trực tiếp `model.track(..., tracker="bytetrack.yaml")`. Bộ đệm theo vết mặc định của YOLO chỉ lưu vết trong khoảng 30 khung hình (xấp xỉ 1 giây). Khi công nhân quay lưng hoặc bị che khuất tạm thời quá 1 giây, ByteTrack tạo ra một ID mới.
  - Bộ tracker tùy chỉnh của hệ thống (`TrackerEngineOOP`) vốn đã tích hợp Deep Re-ID (MobileNetV3) nhưng bị đoạn code ưu tiên `byte_track_id` từ YOLO đè lên, vô hiệu hóa khả năng so khớp lại ngoại hình cũ.
  - Lõi ByteTrack trong YOLO lưu biến nội bộ `predictor.trackers`, không tự reset khi gọi xóa nhật ký thông thường.
- **Hướng suy luận xử lý:**
  - Chuyển lệnh gọi của mô hình người từ `model.track()` sang `model.predict()`, bàn giao hoàn toàn quyền quyết định ID cho `TrackerEngineOOP`.
  - Tăng bộ nhớ lưu vết (`max_disappeared_frames`) từ 300 lên 3600 khung hình (tương đương 2 phút ở 30 FPS).
  - Tăng ngưỡng khoảng cách tâm (`distance_threshold`) từ 150 lên 350 pixel để bắt kịp người di chuyển nhanh.
  - Giảm ngưỡng tương đồng cosine của Re-ID (`reid_similarity_threshold`) từ 0.65 xuống 0.45 để thích ứng với điều kiện ánh sáng thay đổi.
  - Bổ sung hàm `reset()` vào `HumanDetectorOOP` can thiệp vào predictor để ép bộ đếm về 1 khi người dùng bấm xóa nhật ký.

### 2.4. Cơ chế chốt trạng thái vi phạm bám đuổi (State Latching)

- **Bối cảnh thực tế:** Trước đây, trạng thái an toàn được đánh giá độc lập theo cửa sổ trượt. Khi công nhân không đội mũ (`no helmet`), hệ thống báo lỗi. Nhưng khi người đó quay đầu sang hướng khác khiến AI không nhìn thấy vùng đầu trong vài frame, cửa sổ trượt không có dữ liệu và rơi vào nhánh `else: status = "unknown"`, làm lỗi vi phạm tự động biến mất rồi lại hiện lại gây chớp nháy.
- **Yêu cầu nghiệp vụ:** Một khi đã bắt được lỗi vi phạm trên một người đang theo vết, lỗi đó phải bám theo người đó liên tục cho đến khi tìm thấy bằng chứng khắc phục (ví dụ: `no helmet` chỉ biến mất khi tìm thấy nhãn `helmet on` trên người đó).
- **Hướng suy luận xử lý:**
  - Loại bỏ hoàn toàn việc gán trạng thái về `"unknown"` khi mất dấu tạm thời đối với nhóm trang bị đối xứng (Mũ, Áo, Giày).
  - Thiết lập cơ chế chốt trạng thái (State Latching):
    - Trạng thái chỉ chuyển từ an toàn sang vi phạm khi có bằng chứng vi phạm rõ ràng (`no helmet`, `no vest`, `no boots`).
    - Trạng thái chỉ chuyển từ vi phạm sang an toàn khi phát hiện bằng chứng khắc phục (`helmet on`, `vest`, `boots`).
    - Nếu không có phát hiện mới (do góc khuất, quay lưng), giữ nguyên trạng thái trước đó.
  - Với thẻ nhân viên: Nếu đã xác nhận có thẻ (`compliant`), việc che khuất tạm thời sẽ không bị phạt lại. Nếu chưa từng thấy thẻ sau thời gian ân hạn, chuyển thành vi phạm và giữ nguyên cho đến khi thấy thẻ.

### 2.5. Sự khác biệt giữa bảng trạng thái trực tiếp (Live Table) và Cơ sở dữ liệu (SQLite)

- **Bối cảnh thực tế:** Người dùng ấn nút "Xóa nhật ký" nhưng trên bảng vẫn hiển thị các dòng trạng thái (ví dụ: công nhân đứng ở rìa cam bị báo "CHƯA RÕ GÓC NHÌN (CẮT MÉP CAM)"). Người dùng nghĩ nút xóa không hoạt động.
- **Nguyên nhân gốc rễ:**
  - Có sự nhầm lẫn giữa bảng theo dõi trực tiếp (Live Status của luồng camera hiện tại) và bảng lịch sử lưu trong cơ sở dữ liệu.
  - Dữ liệu trong SQLite đã được xóa sạch, nhưng do luồng WebSocket liên tục gửi trạng thái các đối tượng đang đứng trước ống kính (30 lần/giây), giao diện web lập tức vẽ lại trạng thái của những người đang có mặt.
- **Hướng suy luận xử lý:**
  - Tách bạch rõ ràng khái niệm: Bảng Live phản ánh hiện trường tức thời, Cơ sở dữ liệu phản ánh lịch sử vi phạm.
  - Tinh chỉnh điều kiện gắn mác: Chỉ cấp mác "ĐỦ BẢO HỘ" khi người đó không có vi phạm hành vi nào VÀ mô hình đã nhìn thấy đủ cả 4 trang bị (Mũ, Áo, Giày, Thẻ). Nếu chưa nhìn thấy đủ 4 trang bị mà không có lỗi, gắn mác "CHƯA XÁC ĐỊNH ĐỦ BẢO HỘ".

---

## 3. THIẾT KẾ GIẢI PHÁP NÂNG CẤP CHẤT LƯỢNG VẬN HÀNH (GIAI ĐOẠN 2 - QoL)

### 3.1. Lưu trữ bằng chứng vi phạm kép (Full Frame + Cropped Person Snapshot)

- **Mục tiêu:** Cung cấp bằng chứng hình ảnh không thể chối cãi khi công nhân thắc mắc về biên bản vi phạm.
- **Cơ chế hoạt động:**
  1. Khi một sự kiện vi phạm chuyển sang trạng thái `ACTIVE` trong `SafetyStateManagerOOP`:
     - Nhận khung hình gốc `frame` tại thời điểm vi phạm cùng tọa độ bounding box `[x_min, y_min, x_max, y_max]` của người vi phạm.
  2. Tạo 2 file ảnh trong thư mục `logs/snapshots/`:
     - **Ảnh 1 (Toàn cảnh - Full Frame):** Khung hình đầy đủ, vẽ bounding box màu đỏ bao quanh người vi phạm kèm nhãn lỗi tiếng Việt rõ ràng.
       - Tên file: `full_{track_id}_{violation_type}_{timestamp}.jpg`
     - **Ảnh 2 (Cận cảnh - Cropped Person):** Cắt riêng phần cơ thể của người vi phạm có mở rộng đệm biên (padding 10%) để nhìn rõ chi tiết trang phục.
       - Tên file: `crop_{track_id}_{violation_type}_{timestamp}.jpg`
  3. Cập nhật cấu trúc bảng `violations` trong SQLite:
     - Thêm cột `full_snapshot_path TEXT`
     - Thêm cột `crop_snapshot_path TEXT`
  4. Cấu hình FastAPI phục vụ thư mục tĩnh `/snapshots` để giao diện web truy cập trực tiếp bằng đường dẫn URL.

### 3.2. Tra cứu lịch sử vi phạm từ Database và Xuất báo cáo

- **Mục tiêu:** Cung cấp công cụ quản lý toàn diện cho phòng an toàn lao động và nhân sự.
- **Cơ chế hoạt động:**
  1. **API Tra cứu (`GET /api/violations/history_db`):**
     - Nhận các tham số: `start_time`, `end_time`, `track_id`, `violation_type`, `status`, `limit`, `offset`.
     - Truy vấn bảng `violations` trong file `logs/violations_log.db`.
     - Trả về danh sách bản ghi kèm URL ảnh bằng chứng kép.
  2. **API Xuất báo cáo (`GET /api/violations/export_csv`):**
     - Trích xuất dữ liệu theo bộ lọc ngày tháng.
     - Sinh file CSV mã hóa chuẩn UTF-8 có Byte Order Mark (BOM: `\ufeff`) để phần mềm Microsoft Excel hiển thị đúng tiếng Việt có dấu mà không bị lỗi font.
     - Các cột báo cáo: Mã bản ghi, Mã người (Track ID), Họ tên (nếu có profile), Loại vi phạm, Thời điểm bắt đầu, Thời điểm kết thúc, Thời lượng vi phạm (giây), Trạng thái, Đường dẫn ảnh bằng chứng.
  3. **Giao diện Web:**
     - Thiết kế thêm tab **Nhật ký lưu trữ (Database Log)** độc lập với bảng Live.
     - Tích hợp bộ lọc thời gian và nút xem ảnh popup (Modal Dialog) hiển thị song song ảnh toàn cảnh và ảnh cận cảnh.

---

## 4. THIẾT KẾ GIẢI PHÁP TỐI ƯU HIỆU SUẤT HỆ THỐNG (GIAI ĐOẠN 3 - PERFORMANCE)

### 4.1. Giải phóng Event Loop trong FastAPI bằng Async / ThreadPool

- **Vấn đề:** Hàm `pipeline_manager.process_frame(current_frame)` chạy các phép tính ma trận và suy luận mô hình nặng trên CPU/GPU. Gọi trực tiếp hàm này trong vòng lặp `async def websocket_stream_endpoint` sẽ khóa cứng (block) luồng xử lý sự kiện (Event Loop) của FastAPI, khiến các request HTTP khác bị treo.
- **Giải pháp xử lý:**
  - Sử dụng `asyncio.to_thread(pipeline_manager.process_frame, current_frame)` để chuyển tác vụ xử lý ảnh sang một luồng riêng biệt thuộc `ThreadPoolExecutor`.
  - Event loop chính của FastAPI được giải phóng hoàn toàn, duy trì khả năng đáp ứng các API truy vấn lịch sử, xuất báo cáo và quản lý vùng ROI một cách mượt mà ngay cả khi AI đang suy luận liên tục.

### 4.2. Cơ chế đồng bộ thời gian thực và loại bỏ trễ khung hình (Frame Dropping)

- **Vấn đề:** Camera RTSP đẩy khung hình liên tục ở tốc độ 25 - 30 FPS. Nếu tốc độ suy luận của toàn bộ Pipeline là 15 - 20 FPS, các khung hình chưa kịp xử lý sẽ bị dồn ứ trong bộ đệm hàng đợi (buffer queue). Sau một thời gian hoạt động, hình ảnh hiển thị trên màn hình sẽ bị trễ (lag/delay) vài giây so với thực tế bên ngoài.
- **Giải pháp xử lý:**
  - Chuẩn hóa cơ chế lưu trữ của `RTSPCameraStreamWorker`: chỉ lưu trữ duy nhất 1 khung hình mới nhất (`self.latest_frame`) kèm khóa luồng an toàn (`threading.Lock`).
  - Vòng lặp AI khi kết thúc một chu kỳ suy luận sẽ lập tức đọc khung hình mới nhất tại thời điểm hiện tại và bỏ qua toàn bộ các khung hình trung gian đã bị bỏ lỡ.
  - Đảm bảo video hiển thị và cảnh báo luôn bám sát thời gian thực (Zero-latency real-time synchronization).

---

## 5. CÁC TỐI ƯU HÓA VÀ TINH CHỈNH ĐẶC BIỆT MỚI NHẤT (GIAI ĐOẠN 4)

### 5.1. Gom nhóm vi phạm: 1 Track ID tương ứng 1 dòng duy nhất trong Database
- **Nguyên nhân:** Trước đây mỗi loại vi phạm sinh ra 1 dòng riêng khiến một người bị nhiều lỗi bị xé lẻ thành nhiều dòng trong Database.
- **Giải pháp:** Cập nhật `state_manager.py` gom nhóm trực tiếp: mỗi người chỉ có 1 dòng duy nhất, danh sách các lỗi đang vi phạm được nối chuỗi (ví dụ: `no_vest, no_helmet, no_id_card`). Khi có lỗi mới thì `UPDATE` dồn vào dòng đó, tuyệt đối không sinh thêm dòng mới.

### 5.2. Tinh chỉnh ngưỡng nhận diện Giày và Hành vi (Điện thoại, Thuốc lá)
- **Giày bảo hộ (`boots` / `no boots`):** Trong môi trường văn phòng/nhà xưởng, chân thường xuyên bị bàn ghế, góc tủ che khuất một phần. Đã hạ ngưỡng confidence tối thiểu xuống **`0.20`** (kết hợp bộ lọc không gian nửa thân dưới `>= 0.50`) để bắt nhạy ngay cả khi bị che khuất một phần chân.
- **Điện thoại và Thuốc lá (`phone_usage`, `smoking`):** Đã nâng ngưỡng confidence tối thiểu lên **`0.60`** để loại bỏ hoàn toàn các trường hợp báo động giả do cử chỉ tay (như gãi đầu, vuốt mặt, cầm bút).

### 5.3. Hoàn thiện vòng đời phát hiện và tháo cởi Thẻ nhân viên (ID Card)
- **Vấn đề cũ:** Khi công nhân từng đeo thẻ 1 lần (đạt `compliant`), trạng thái bị khóa cứng vĩnh viễn, khi cởi thẻ ra cất đi hệ thống không báo lại là vi phạm.
- **Giải pháp:** Bổ sung biến đếm `consecutive_missing_id_frames`. 
  - Khi thấy thẻ: gán `compliant` và reset bộ đếm về 0.
  - Khi không thấy thẻ: tăng bộ đếm; nếu không thấy liên tiếp quá **45 khung hình** (~1.5 giây) thì chính thức xác nhận đã cởi thẻ ra và chuyển lại thành `violation`.

### 5.4. Định nghĩa chuẩn xác về sự kiện Khắc phục và Rời camera
- **Rời camera:** Không tự ý gán nhãn `RESOLVED` hay kết thúc phiên. Giữ nguyên trạng thái vi phạm của người đó trong hệ thống.
- **Khắc phục lỗi:** Lỗi nào được khắc phục thì tự động xóa khỏi danh sách lỗi. Khi khắc phục hết toàn bộ lỗi thì dòng vi phạm tự động được xóa bỏ sạch sẽ khỏi Database.
- **Bỏ cột Trạng thái:** Loại bỏ hoàn toàn cột "Trạng thái" trên giao diện Web để tránh các chữ thừa thãi ("Đang diễn ra", "Đã khắc phục").

### 5.5. Giao diện thu gọn Accordion cho Lịch sử Database & Đồng bộ Bảng Live
- **Bảng Database (Kiểu A):** Thu gọn chỉ hiển thị tổng số người vi phạm, tên người và thời gian. Khi click chuột vào dòng của người đó mới bung ra chi tiết các nhãn lỗi vi phạm và nút `Xem ảnh` bằng chứng.
- **Nút Đóng Modal:** Thu nhỏ kích thước gọn gàng, tinh tế ở góc trên bên phải khung ảnh bằng chứng kép.
- **Đồng bộ Bảng Live theo thời gian thực:** Backend truyền danh sách `current_frame_track_ids` (những ID thực tế đang xuất hiện trong khung hình). Bảng Live chỉ hiển thị người đang có mặt trước camera; khi không có ai thì lập tức hiển thị `Chưa có người trong camera`, không còn bị lưu vết trễ từ bộ đệm Re-ID.

---

## 6. BẢNG TỔNG HỢP CÁC CHUYỂN DỊCH CỐT LÕI (CORE ARCHITECTURAL SHIFTS)

Bảng phân tích toàn diện lý do thay đổi, giải pháp kỹ thuật và giá trị thực tế mang lại cho toàn bộ hệ thống:

| STT | Trục tính năng | Trạng thái ban đầu (Before) | Nguyên nhân & Động lực thay đổi (Why) | Giải pháp kỹ thuật cốt lõi (How) | Kết quả & Vấn đề được giải quyết (Impact) |
|---|---|---|---|---|---|
| **1** | **Cơ chế Tracking & Giữ ID** | Dùng ByteTrack mặc định tích hợp trong YOLO (`model.track`) | Bộ đệm quá ngắn (~30 frames / 1s). Công nhân chỉ cần quay lưng, cúi người hoặc bị góc tủ che 1 giây là ByteTrack quên ngay lập tức và cấp ID mới (nhảy từ Người 9 sang Người 13). | Chuyển sang `model.predict()`, bàn giao 100% quyền theo vết cho `TrackerEngineOOP`. Nâng bộ nhớ lên 3600 frames (2 phút), hạ ngưỡng Re-ID cosine xuống 0.45 và tăng khoảng cách tâm lên 350px. | **Giải quyết triệt để sự cố nhảy ID:** Duy trì ID ổn định, bám dính liên tục suốt 2 phút dù người có đi ra sau vật cản hay quay lưng đổi góc đứng. |
| **2** | **Logic bắt lỗi an toàn (State)** | Đánh giá độc lập theo cửa sổ trượt (Sliding Window Voting). | Khi mất dấu tạm thời ở vùng đầu/chân, rơi vào nhánh fallback `unknown`. Vi phạm vừa bắt được lại tự biến mất rồi hiện lại gây chớp nháy (flickering). | Bỏ hoàn toàn fallback `unknown`. Xây dựng cơ chế **State Latching**: Một khi đã bắt được lỗi vi phạm thì chốt cứng bám theo người cho tới khi có nhãn trang bị bảo hộ thay thế rõ ràng. | **Giải quyết triệt để lỗi chớp nháy:** Lỗi vi phạm bám đuổi tuyệt đối, không thể trốn tránh bằng cách quay đầu hay né góc nhìn. |
| **3** | **Lọc không gian giải phẫu cơ thể** | Điện thoại/thuốc lá gán dải Y 0.10 - 0.85; Giày yêu cầu confidence cao (>0.40). | Điện thoại đưa lên tai hoặc thuốc lá đưa lên miệng rơi vào vùng đầu (`<0.10`) bị lọc nhầm thành rác. Giày ở văn phòng bị bàn ghế che khuất chỉ đạt conf 0.20 - 0.30 nên không bao giờ nhận được giày. | Tách nhóm hành vi cho phép dải Y từ 0.0 đến 0.85, nâng conf lên 0.60 chống báo động giả. Giày giữ dải Y `>=0.50` nhưng hạ conf xuống 0.20 để bắt nhạy. | **Bắt trúng 100% hành vi và giày:** Nghe điện thoại ở tai hay hút thuốc ở miệng đều bắt trúng; bàn chân bị che một phần sau bàn làm việc vẫn nhận diện chuẩn xác. |
| **4** | **Vòng đời Thẻ nhân viên (ID Card)** | Model không có nhãn `no id_card`; khi đã thấy thẻ 1 lần thì gán `compliant` vĩnh viễn. | Công nhân cởi thẻ ra cất vào túi áo/túi quần nhưng hệ thống vẫn nhớ trạng thái an toàn cũ, không bao giờ báo lỗi lại. | Bổ sung biến đếm `consecutive_missing_id_frames`. Khi thấy thẻ: reset đếm về 0. Khi cởi thẻ ra: không thấy liên tiếp quá 45 khung hình (~1.5s) lập tức kích hoạt lại lỗi vi phạm `no_id_card`. | **Đúng chuẩn nghiệp vụ thực tế:** Chưa đeo thì phạt -> Đeo vào thì xóa phạt -> Cởi ra cất đi thì tự động bắt phạt lại. |
| **5** | **Cấu trúc lưu trữ Cơ sở dữ liệu** | Mỗi loại vi phạm lưu 1 dòng riêng trong SQLite; hết giờ thì đổi thành `RESOLVED_TIMEOUT`. | Một người vi phạm 3 lỗi bị xé lẻ thành 3-4 dòng rác bảng; người đi ra khỏi camera bị in chữ "Đã khắc phục" gây hiểu nhầm nghiêm trọng. | Gom nhóm: 1 ID chỉ có **1 dòng duy nhất**, nối các lỗi bằng dấu phẩy. Người rời camera: **giữ nguyên lỗi vi phạm**. Khắc phục xong lỗi nào: **xóa bỏ lỗi đó**; khắc phục hết: **xóa dòng khỏi DB**. | **Cơ sở dữ liệu tinh gọn và chính xác:** Không có dữ liệu rác, phản ánh trung thực ai đang còn nợ lỗi vi phạm nào. |
| **6** | **Hiển thị Camera Live thời gian thực** | Bảng Live đọc từ `person_summaries` của State Manager (vốn lưu vết 120 giây). | Camera trống phòng (không có ai) nhưng bảng Live vẫn hiển thị Người 1, Người 2 do bộ nhớ 120 giây chưa hết hạn. | Backend gửi thêm `current_frame_track_ids` (chỉ những ID đang thực sự có mặt trong frame). Frontend chỉ vẽ những ID này, nếu rỗng thì hiện `Chưa có người trong camera`. | **Tách bạch hoàn hảo giữa AI ngầm và Màn hình Live:** Bảng Live phản ánh đúng 100% những gì mắt người nhìn thấy trên khung hình hiện tại. |
| **7** | **Trải nghiệm người dùng (UX/UI)** | Bảng Database phơi hết toàn bộ dòng chi tiết; nút đóng to bè chiếm hết màn hình; lệch số cột. | Giao diện bị rối mắt, thanh cuộn kéo dài khó theo dõi tổng thể; cửa sổ xem ảnh bị che mất dòng tiêu đề. | Đồng bộ 4 cột chuẩn ở bảng Live. Bảng Database chuyển sang **Kiểu A (Accordion)**: mặc định thu gọn chỉ hiện số người và danh sách người, bấm vào mới bung chi tiết lỗi và nút xem ảnh. Nút đóng modal thu nhỏ 12px gọn gàng. | **Giao diện công nghiệp chuyên nghiệp:** Gọn gàng, hiện đại, xem chi tiết và bằng chứng tiện lợi chỉ với 1 cú click. |

---

## 7. NHẬT KÝ THAY ĐỔI THEO TỪNG NGÀY (DAILY CHANGELOG)

### [2026-10-05] - TỔNG HỢP NÂNG CẤP HỆ THỐNG (BẢN RÚT GỌN)

#### 1. Ca sáng: Lõi Tracking, State Latching và SQLite
- **SQLite Database:** Chuyển đổi cơ chế lưu file text sang SQLite (`logs/violations_log.db`), sửa lỗi nút xóa nhật ký và đưa số đếm ID về 1.
- **Ổn định Tracking:** Vô hiệu hóa ByteTrack mặc định của YOLO, kích hoạt bộ theo vết chuyên sâu TrackerEngineOOP (Deep Re-ID MobileNetV3) với bộ đệm 2 phút (3600 frames), giải quyết triệt để lỗi nhảy ID khi bị che khuất.
- **Cơ chế State Latching:** Chốt lỗi bám theo người cho đến khi AI nhìn thấy đồ bảo hộ thay thế, loại bỏ hoàn toàn hiện tượng chớp nháy vi phạm.
- **Các file sửa đổi:** `core_pipeline/human_detector.py`, `core_pipeline/state_manager.py`, `tracking/tracker_engine.py`, `services/api_backend.py`.

#### 2. Ca chiều: Snapshot Bằng chứng, Hiệu suất, Vòng đời Thẻ và Nâng cấp Giao diện
- **Snapshot Bằng chứng kép:** Tự động cắt và lưu song song ảnh toàn cảnh (Full Frame) và cận cảnh (Crop) vào `logs/snapshots/`, liên kết đường dẫn vào SQLite.
- **Tối ưu FastAPI và RTSP:** Bọc xử lý AI bằng `asyncio.to_thread` trong WebSocket chống nghẽn Event Loop; cơ chế lấy frame mới nhất chống trễ video camera RTSP.
- **Gom nhóm Database 1 ID 1 dòng:** Mỗi người chỉ lưu đúng 1 dòng trong DB (gộp chung các lỗi), phát sinh thêm lỗi thì cập nhật dồn vào dòng đó; loại bỏ tính năng xuất file Excel/CSV.
- **Tinh chỉnh độ nhạy Giày và Hành vi:** Hạ ngưỡng phát hiện giày (`boots`/`no boots`) xuống 0.20 để bắt nhạy khi chân bị che khuất; nâng ngưỡng điện thoại/thuốc lá lên 0.60 chống báo nhầm cử chỉ tay.
- **Vòng đời Thẻ nhân viên:** Đeo thẻ thì xóa phạt; cởi thẻ ra liên tiếp quá 45 frames (~1.5s) tự động kích hoạt phạt lại.
- **Chuẩn hóa Logic sự kiện:** Rời camera thì giữ nguyên lỗi trong DB; khắc phục lỗi nào xóa lỗi đó, khắc phục hết thì tự động xóa dòng khỏi DB; loại bỏ cột Trạng thái thừa thãi.
- **Nâng cấp Giao diện Web (UX/UI):**
  - Chuyển bảng Lịch sử vi phạm sang dạng Accordion thu gọn theo người, bấm vào dòng người mới bung chi tiết lỗi và nút xem ảnh; tự động làm mới mỗi 5 giây.
  - Bảng Live chỉ hiển thị người đang thực sự có mặt trước camera (`current_frame_track_ids`), phòng trống thì hiển thị thông báo chưa có người.
  - Thu nhỏ nút Đóng modal xem ảnh gọn gàng (12px); đồng bộ hai bảng chuẩn 4 cột: Thời gian, ID, Lỗi vi phạm, Bằng chứng.
- **Cơ chế Chốt Vi phạm & Xác nhận Khắc phục (Debounce & Strict Latching):** Thêm bộ đếm khung hình xác nhận vi phạm (6 frames) chống nhiễu 1 frame; khi đã bắt lỗi thì khóa cứng trạng thái (không đổi về unknown hay hủy lỗi khi che khuất/mất dấu); chỉ xóa bỏ lỗi khi AI phát hiện khắc phục liên tiếp đủ 6 frames.
- **Tập trung hóa tài liệu:** Quy hoạch toàn bộ file log về thư mục duy nhất `Plan/`.
- **Các file sửa đổi:** `core_pipeline/ppe_classifier.py`, `core_pipeline/pipeline_manager.py`, `core_pipeline/state_manager.py`, `services/api_backend.py`, `frontend/index.html`.

