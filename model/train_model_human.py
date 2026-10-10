import os
import yaml
from ultralytics import YOLO

def train_yolo_model(data_configuration_path, model_weights_path, output_project_directory):
    if not os.path.exists(data_configuration_path):
        print(f"Lỗi: Không tìm thấy file cấu hình tại {data_configuration_path}")
        return
        
    model = YOLO(model_weights_path)
    
    training_arguments = {
        "data": data_configuration_path,
        "epochs": 100,
        "patience": 50,
        "batch": 16,
        "imgsz": 640,
        "save": True,
        "save_period": -1,
        "cache": False,
        "device": "0",
        "workers": 4,
        "project": output_project_directory,
        "name": "yolov8_training_run",
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "auto",
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        "single_cls": False,
        "lr0": 0.001  

    }
    
    model.train(**training_arguments)
    print(f"Hoàn tất. Kết quả lưu tại: {output_project_directory}")

def prepare_kaggle_yaml(original_yaml_path):
    # Hàm xử lý lỗi đường dẫn tương đối và lỗi thư mục chỉ đọc (read-only) trên Kaggle
    if not os.path.exists(original_yaml_path):
        print(f"Lỗi: Không tìm thấy {original_yaml_path}")
        return original_yaml_path
        
    with open(original_yaml_path, "r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)
        
    base_directory = os.path.dirname(original_yaml_path)
    
    yaml_data["train"] = os.path.join(base_directory, "train", "images")
    yaml_data["val"] = os.path.join(base_directory, "valid", "images")
    yaml_data["test"] = os.path.join(base_directory, "test", "images")
    
    fixed_yaml_path = "/kaggle/working/data_fixed.yaml"
    with open(fixed_yaml_path, "w", encoding="utf-8") as file:
        yaml.dump(yaml_data, file, default_flow_style=False, sort_keys=False)
        
    return fixed_yaml_path

if __name__ == "__main__":
    
    kaggle_original_yaml = "/kaggle/input/personworker/dataset_yolo/data.yaml"
    
    kaggle_data_yaml_path = prepare_kaggle_yaml(kaggle_original_yaml)
    
    pretrained_weights = "yolov8n.pt"
    
    kaggle_output_directory = "/kaggle/working/training_results"
    
    train_yolo_model(
        data_configuration_path=kaggle_data_yaml_path,
        model_weights_path=pretrained_weights,
        output_project_directory=kaggle_output_directory
    )