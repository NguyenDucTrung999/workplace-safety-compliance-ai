import os
import cv2
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core_pipeline.pipeline_manager import SafetyPipelineManagerOOP

def test_pipeline_on_image():
    test_image_path = os.path.join(project_root, "test", "test_weight", "anh1.jpg")
    output_save_path = os.path.join(project_root, "test", "test_weight", "pipeline_oop_result.jpg")

    if not os.path.exists(test_image_path):
        print(f"Lỗi: Không tìm thấy file ảnh kiểm thử tại {test_image_path}")
        return

    print("Đang khởi tạo Lớp Quản lý Pipeline OOP Tích hợp (Detector + Tracker + PPE + StateManager + ROIManager)...")
    pipeline_manager = SafetyPipelineManagerOOP()

    pipeline_manager.roi_manager.add_zone(
        zone_id="ZONE_TEST_01",
        zone_name="Khu vuc kiem tra an toan",
        polygon_points_list=[(100, 50), (1200, 50), (1200, 800), (100, 800)],
        allowed_person_ids_list=["1"]
    )

    print(f"Đang đọc ảnh kiểm thử: {test_image_path}")
    raw_frame = cv2.imread(test_image_path)

    print("Đang tiến hành chạy liên hoàn Pipeline (mô phỏng luồng 6 frames để kích hoạt bộ đếm xác nhận)...")
    for frame_index in range(6):
        annotated_frame, report_dictionary, new_events_list = pipeline_manager.process_frame(raw_frame)

    print("--------------------------------------------------")
    print("=== KẾT QUẢ XỬ LÝ PIPELINE TỔNG HỢP ===")
    print(f"Tổng số đối tượng người được theo vết: {report_dictionary['total_tracked_persons']}")
    
    for person in report_dictionary["persons_detail"]:
        print(f"  - Track ID {person['track_id']}: Tọa độ {person['bounding_box']}, Độ tin cậy: {person['confidence']:.2f}")
        for ppe in person["ppe_detections"]:
            print(f"      + Bảo hộ phát hiện: {ppe['class_name']} (Độ tin cậy: {ppe['confidence']:.2f})")
        for roi_eval in person["roi_evaluations"]:
            print(f"      + Vùng ROI [{roi_eval['zone_name']}]: Xâm nhập = {roi_eval['is_inside']}, Vi phạm vùng = {roi_eval['zone_violations']}")

    print("--------------------------------------------------")
    print(f"Báo cáo Trạng thái State Manager: {report_dictionary['state_summary']}")
    print(f"Các sự kiện mới phát hiện: {new_events_list}")

    cv2.imwrite(output_save_path, annotated_frame)
    print(f"Ảnh kết quả vẽ nhận diện đã được lưu tại: {output_save_path}")

if __name__ == "__main__":
    test_pipeline_on_image()
