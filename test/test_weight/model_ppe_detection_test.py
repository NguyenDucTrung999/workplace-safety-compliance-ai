import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# =======================================================================
# KHU VỰC CÀI ĐẶT ĐƯỜNG DẪN (BẠN CHỈ CẦN THAY ĐỔI Ở ĐÂY)
# =======================================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(base_dir))

trained_model_path = os.path.join(project_root, "model", "weights", "model_ppe_detection.pt")
test_image_path = os.path.join(base_dir, "anh1.jpg")

# Ngưỡng độ tin cậy nhận diện (>= 50%)
confidence_threshold = 0.5

# =======================================================================
# KHU VỰC XỬ LÝ VÀ HIỂN THỊ KẾT QUẢ
# =======================================================================

def test_ppe_weight_model(model_path, image_path):
    # Kiểm tra sự tồn tại của các tệp đầu vào
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy tệp mô hình PPE tại {model_path}")
        return
        
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy tệp ảnh kiểm thử tại {image_path}")
        return

    print("Đang tải Mô hình 2 (PPE Detection)...")
    model = YOLO(model_path)
    
    # IN RA DANH SÁCH CLASS ĐỂ KIỂM TRA TRỰC TIẾP
    print("--------------------------------------------------")
    print(f"Danh sách các Class của mô hình đang load: {model.names}")
    print("--------------------------------------------------")
    
    print("Đang tiến hành nhận diện trang thiết bị và hành vi...")
    prediction_results = model.predict(source=image_path, conf=confidence_threshold, save=False)
    
    first_result = prediction_results[0]
    
    # In ra danh sách các đối tượng phát hiện được
    detected_boxes = first_result.boxes
    print(f"Tổng số đối tượng PPE phát hiện được: {len(detected_boxes)}")
    
    for box in detected_boxes:
        class_id = int(box.cls[0])
        class_name = first_result.names[class_id]
        box_confidence = float(box.conf[0])
        print(f"  - Phát hiện: {class_name} (Độ tin cậy: {box_confidence:.2f})")
    
    # Vẽ khung nhận diện lên ảnh
    result_image_bgr = first_result.plot()
    result_image_rgb = cv2.cvtColor(result_image_bgr, cv2.COLOR_BGR2RGB)
    
    # Hiển thị kết quả bằng Matplotlib
    plt.figure(figsize=(12, 12))
    plt.imshow(result_image_rgb)
    plt.axis("off")
    plt.title(f"PPE Detection Result - Total Objects: {len(detected_boxes)}")
    plt.show()

if __name__ == "__main__":
    test_ppe_weight_model(trained_model_path, test_image_path)