import os
import yaml
from ultralytics import YOLO

def build_ppe_yaml_configuration(base_input_directory="/kaggle/input"):
    """
    Tự động quét bộ dữ liệu PPE Dataset trên Kaggle 
    và đúc file YAML đường dẫn tuyệt đối cho 10 Class chuẩn.
    """
    print("Đang quét dữ liệu PPE Dataset trên máy chủ Kaggle...")
    
    target_dataset_directory = None
    
    # Quét tìm thư mục chứa dữ liệu dataset_ppe_10class
    for root, dirs, files in os.walk(base_input_directory):
        if "dataset_ppe_10class" in dirs:
            target_dataset_directory = os.path.join(root, "dataset_ppe_10class")
            break
        elif "train" in dirs and ("valid" in dirs or "val" in dirs):
            target_dataset_directory = root
            break

    if not target_dataset_directory:
        print("Lỗi: Không tìm thấy thư mục dataset_ppe_10class trong /kaggle/input/")
        return None

    print(f"Đã xác định vị trí dữ liệu tại: {target_dataset_directory}")

    # Xác định đường dẫn ảnh tuyệt đối cho các tập dữ liệu
    train_images_path = os.path.join(target_dataset_directory, "train", "images")
    validation_images_path = os.path.join(target_dataset_directory, "valid", "images")
    test_images_path = os.path.join(target_dataset_directory, "test", "images")

    # Kiểm tra nếu thư mục không phân chia cấp images
    if not os.path.exists(train_images_path):
        train_images_path = os.path.join(target_dataset_directory, "train")
        
    if not os.path.exists(validation_images_path):
        validation_images_path = os.path.join(target_dataset_directory, "valid")

    # Danh sách 10 Class chuẩn của Mô hình 2 PPE
    class_names_list = [
        'boots', 
        'helmet on', 
        'no boots', 
        'no helmet', 
        'no vest', 
        'vest', 
        'id_card', 
        'no id_card', 
        'smoking', 
        'phone_usage'
    ]

    # Kiểm tra xem có thể đọc trực tiếp từ data.yaml gốc hay không
    original_yaml_path = os.path.join(target_dataset_directory, "data.yaml")
    if os.path.exists(original_yaml_path):
        try:
            with open(original_yaml_path, "r", encoding="utf-8") as file:
                yaml_data = yaml.safe_load(file)
                if "names" in yaml_data:
                    class_names_list = yaml_data["names"]
        except Exception as exception_error:
            print(f"Cảnh báo: Sử dụng danh sách 10 class mặc định ({exception_error}).")

    # Tạo nội dung file YAML mới tuyệt đối
    yaml_content = {
        "train": train_images_path,
        "val": validation_images_path,
        "test": test_images_path,
        "nc": len(class_names_list),
        "names": class_names_list
    }

    fixed_yaml_path = "/kaggle/working/ppe_dataset_fixed.yaml"
    with open(fixed_yaml_path, "w", encoding="utf-8") as file:
        yaml.dump(yaml_content, file, default_flow_style=False, sort_keys=False)

    print(f"Đã đúc thành công file cấu hình tại: {fixed_yaml_path}")
    print(f"Số lượng Class: {len(class_names_list)}")
    print(f"Danh sách Class: {class_names_list}")
    
    return fixed_yaml_path


def train_ppe_model(data_configuration_path, model_weights_path, output_project_directory):
    if not data_configuration_path:
        print("Lỗi: Đường dẫn file cấu hình rỗng.")
        return

    model = YOLO(model_weights_path)

    training_arguments = {
        "data": data_configuration_path,
        "epochs": 100,
        "patience": 30,
        "batch": 16,
        "imgsz": 1280,
        "save": True,
        "save_period": -1,
        "cache": False,
        "device": "0",
        "workers": 2,
        "project": output_project_directory,
        "name": "yolov8_ppe_training_run",
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "auto",
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        "single_cls": False,
        "lr0": 0.001,
        "mosaic": 0.1,
        "mixup": 0.0
    }

    print("Đang khởi chạy huấn luyện Mô hình 2 (PPE Detection)...")
    model.train(**training_arguments)
    print(f"Hoàn tất huấn luyện. Kết quả lưu tại: {output_project_directory}")


if __name__ == "__main__":
    print("=== BẮT ĐẦU CHẠY HUẤN LUYỆN MÔ HÌNH 2 PPE TRÊN KAGGLE ===")

    # 1. Tự động quét và tạo file YAML chuẩn cho PPE Dataset 10 Class
    ppe_yaml_path = build_ppe_yaml_configuration("/kaggle/input")

    # 2. Huấn luyện mô hình YOLOv8n với file YAML vừa đúc
    if ppe_yaml_path:
        train_ppe_model(
            data_configuration_path=ppe_yaml_path,
            model_weights_path="yolov8n.pt",
            output_project_directory="/kaggle/working/training_results_ppe"
        )