import os
import cv2
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from core_pipeline.pipeline_manager import SafetyPipelineManagerOOP

def test_user_uploaded_image():
    user_image_path = "/home/cazofi/.gemini/antigravity/brain/87bed50a-e5b5-4178-90fe-25024e0d9055/.user_uploaded/media_1790993684503.jpg"
    output_save_path = os.path.join(project_root, "test", "test_weight", "user_image_result.jpg")

    if not os.path.exists(user_image_path):
        print(f"Lỗi: Không tìm thấy file ảnh tại {user_image_path}")
        return

    print("Đang khởi tạo Pipeline Manager...")
    pipeline_manager = SafetyPipelineManagerOOP()

    print(f"Đang đọc ảnh người dùng gửi: {user_image_path}")
    raw_frame = cv2.imread(user_image_path)

    annotated_frame, report_dictionary, new_events_list = pipeline_manager.process_frame(raw_frame)

    print("--------------------------------------------------")
    print("=== KẾT QUẢ XỬ LÝ ẢNH NGƯỜI DÙNG TẢI LÊN ===")
    print(f"Tổng số đối tượng người được theo vết: {report_dictionary['total_tracked_persons']}")
    
    for person in report_dictionary["persons_detail"]:
        print(f"  - Track ID {person['track_id']}: Tọa độ {person['bounding_box']}, Độ tin cậy: {person['confidence']:.2f}")
        for ppe in person["ppe_detections"]:
            print(f"      + Bảo hộ phát hiện: {ppe['class_name']} (Độ tin cậy: {ppe['confidence']:.2f})")

    print("--------------------------------------------------")
    print(f"Báo cáo Trạng thái State Manager: {report_dictionary['state_summary']}")
    print(f"Các sự kiện mới phát hiện: {new_events_list}")

    cv2.imwrite(output_save_path, annotated_frame)
    print(f"Ảnh kết quả vẽ nhận diện đã được lưu tại: {output_save_path}")

if __name__ == "__main__":
    test_user_uploaded_image()
