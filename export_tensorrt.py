import os
import sys
import torch
from ultralytics import YOLO

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import HUMAN_MODEL_PATH, PPE_MODEL_PATH

def export_model_to_tensorrt(weights_path, imgsz, dynamic=False, batch=1):
    if not os.path.exists(weights_path):
        print(f"Lỗi: Không tìm thấy file trọng số {weights_path}")
        return None

    print(f"\n==================================================================")
    print(f"BẮT ĐẦU BIÊN DỊCH TENSORRT: {os.path.basename(weights_path)}")
    print(f"Kích thước imgsz: {imgsz}, Half FP16: True, Dynamic: {dynamic}, Batch: {batch}")
    print(f"==================================================================")

    model = YOLO(weights_path)
    try:
        engine_path = model.export(
            format="engine",
            imgsz=imgsz,
            half=True,
            dynamic=dynamic,
            batch=batch,
            device=0,
            verbose=True
        )
        print(f"Biên dịch thành công! File Engine lưu tại: {engine_path}")
        return engine_path
    except Exception as export_error:
        print(f"Lỗi trong quá trình export TensorRT: {export_error}")
        return None

def main():
    if not torch.cuda.is_available():
        print("Lỗi: Không phát hiện GPU CUDA trên thiết bị để biên dịch TensorRT!")
        return

    print("Thiết bị GPU nhận diện:", torch.cuda.get_device_name(0))
    print("Phiên bản PyTorch:", torch.__version__)
    
    # 1. Model 1: Human Detection (320x320, batch=1)
    export_model_to_tensorrt(HUMAN_MODEL_PATH, imgsz=320, dynamic=False, batch=1)
    
    # 2. Model 2: PPE Classifier (640x640, dynamic batching cho nhiều người)
    export_model_to_tensorrt(PPE_MODEL_PATH, imgsz=640, dynamic=True, batch=4)

if __name__ == "__main__":
    main()
