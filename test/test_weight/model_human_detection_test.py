import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# =======================================================================
# KHU VỰC CÀI ĐẶT ĐƯỜNG DẪN (BẠN CHỈ CẦN THAY ĐỔI Ở ĐÂY)
# =======================================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(base_dir))

trained_model_path = os.path.join(project_root, "model", "weights", "model_human_detection.pt")
test_image_path = os.path.join(base_dir, "anh2.jpg")

# =======================================================================
# KHU VỰC XỬ LÝ VÀ HIỂN THỊ KẾT QUẢ
# =======================================================================

def test_yolo_model(model_path, image_path):
    # Kiểm tra xem các file có tồn tại không để phòng tránh lỗi
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy file mô hình tại {model_path}")
        return
        
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy file ảnh tại {image_path}")
        return

    print("Đang tải mô hình...")
    # Tải mô hình từ trọng số mà bạn đã huấn luyện
    model = YOLO(model_path)
    
    print("Đang tiến hành nhận diện...")
    # Chạy dự đoán. Tham số conf=0.5 có nghĩa là chỉ hiển thị các kết quả có độ tin cậy từ 50% trở lên
    prediction_results = model.predict(source=image_path, conf=0.5, save=False)
    
    # Lấy kết quả đầu tiên (vì chúng ta chỉ đưa vào 1 ảnh)
    first_result = prediction_results[0]
    
    # Vẽ các khung nhận diện (bounding boxes) lên ảnh gốc
    # Hàm plot() trả về ma trận điểm ảnh (numpy array) theo hệ màu BGR của OpenCV
    result_image_array_bgr = first_result.plot()
    
    # Chuyển đổi hệ màu từ BGR sang RGB để thư viện Matplotlib hiển thị màu sắc chuẩn xác
    result_image_array_rgb = cv2.cvtColor(result_image_array_bgr, cv2.COLOR_BGR2RGB)
    
    # Hiển thị ảnh ngay trên giao diện Notebook
    plt.figure(figsize=(12, 12))
    plt.imshow(result_image_array_rgb)
    plt.axis("off")  # Tắt khung hiển thị tọa độ trục x, y để nhìn ảnh rõ hơn
    plt.show()

if __name__ == "__main__":
    test_yolo_model(trained_model_path, test_image_path)