import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO

# =======================================================================
# KHU VỰC CÀI ĐẶT ĐƯỜNG DẪN (BẠN HÃY THAY ĐỔI Ở ĐÂY)
# =======================================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(base_dir))

# 1. Đường dẫn đến file mô hình (.pt) của bạn
trained_model_path = os.path.join(project_root, "model", "weights", "model_human_detection.pt")

# 2. Đầu vào video: Bạn có thể điền đường dẫn file video (.mp4) hoặc đường dẫn luồng mạng (rtsp://...)
video_source_path = "rtsp://admin:vna%40123456@192.168.110.200:554/media/video2"

# 3. Thư mục chứa file log (script sẽ tự tạo nếu chưa có)
output_directory = base_dir

# =======================================================================
# KHU VỰC XỬ LÝ LÕI
# =======================================================================

def run_video_detection():
    # Kiểm tra và tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    # Tạo đường dẫn cho file log
    log_file_path = os.path.join(output_directory, "detection_log.txt")
    
    print("Đang tải mô hình YOLO...")
    model = YOLO(trained_model_path)
    
    print(f"Đang kết nối luồng video: {video_source_path}")
    video_capture = cv2.VideoCapture(video_source_path)
    
    if not video_capture.isOpened():
        print("Lỗi: Không thể mở được video hoặc luồng tín hiệu mạng.")
        return

    frame_counter = 0
    
    # Mở file log ở chế độ ghi nối tiếp (append) để không làm mất dữ liệu cũ
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        # Ghi một mốc thời gian đánh dấu mỗi lần bạn chạy lại code
        log_file.write(f"\n--- BẮT ĐẦU PHIÊN CHẠY MỚI: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        
        while True:
            success, frame = video_capture.read()
            
            # Nếu video kết thúc hoặc mất kết nối thì dừng vòng lặp (Đã thêm cơ chế chống lỗi frame is None)
            if success == False or frame is None:
                print("Đã đọc hết video hoặc mất kết nối mạng.")
                break
                
            frame_counter += 1
            
            # Gán khung hình hiển thị mặc định là ảnh gốc
            display_frame = frame
            
            # Điều kiện: Cứ 3 frame thì mới xử lý 1 lần
            if frame_counter % 1 == 0:
                # Bắt đầu tính giờ
                start_time = time.time()
                
                # Tham số verbose=False để terminal đỡ bị trôi nhanh
                prediction_results = model.predict(source=frame, conf=0.5, verbose=False)
                first_result = prediction_results[0]
                
                # Kết thúc tính giờ và đếm đối tượng
                end_time = time.time()
                processing_time = end_time - start_time
                detected_objects_count = len(first_result.boxes)
                
                # Nếu mô hình phát hiện ra ít nhất 1 đối tượng, tiến hành ghi log
                if detected_objects_count > 0:
                    # Lấy thời gian thực để ghi vào log
                    timestamp_string = datetime.now().strftime("%H:%M:%S")
                    
                    # Chuẩn bị dòng thông tin để ghi log
                    log_message = (
                        f"Thời điểm: {timestamp_string} | "
                        f"Frame số: {frame_counter} | "
                        f"Số lượng nhận diện: {detected_objects_count} đối tượng | "
                        f"Thời gian xử lý: {processing_time:.4f} giây\n"
                    )
                    
                    # Ghi dòng thông tin vào file txt và in ra terminal để bạn dễ theo dõi
                    log_file.write(log_message)
                    log_file.flush() # Ép lưu file ngay lập tức
                    print(log_message.strip())
                    
                # Cập nhật khung hình hiển thị bằng khung hình đã được YOLO vẽ nhận diện
                display_frame = first_result.plot()
                
            # Mở cửa sổ hiển thị camera trực tiếp (Bấm phím 'q' để thoát an toàn)
            cv2.imshow("Camera Live", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Đã đóng camera.")
                break
                    
    # Giải phóng bộ nhớ video sau khi xử lý xong
    video_capture.release()
    cv2.destroyAllWindows()
    print(f"Quá trình hoàn tất. File log đã được lưu an toàn tại: {log_file_path}")

if __name__ == "__main__":
    run_video_detection()