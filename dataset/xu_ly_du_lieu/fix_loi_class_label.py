
import os

def patch_labelimg_source_code():
    # Đường dẫn file mã nguồn lõi yolo_io.py trong Anaconda của bạn
    target_file_path = "/home/cazofi/anaconda3/envs/labelImg/lib/python3.8/site-packages/libs/yolo_io.py"
    
    if not os.path.exists(target_file_path):
        print(f"Lỗi: Không tìm thấy file mã nguồn tại {target_file_path}")
        return
        
    with open(target_file_path, "r", encoding="utf-8") as file:
        content = file.read()
        
    # Đoạn mã nguồn cũ của labelImg gây văng app
    old_code_snippet = "label = self.classes[int(class_index)]"
    
    # Đoạn mã nguồn mới đã được gia cố an toàn
    new_code_snippet = """class_index_integer = int(class_index)
        if 0 <= class_index_integer < len(self.classes):
            label = self.classes[class_index_integer]
        else:
            label = f"class_{class_index_integer}" """
            
    if old_code_snippet in content:
        updated_content = content.replace(old_code_snippet, new_code_snippet)
        with open(target_file_path, "w", encoding="utf-8") as file:
            file.write(updated_content)
        print("Đã vá thành công mã nguồn yolo_io.py của labelImg!")
    else:
        print("Mã nguồn yolo_io.py đã được vá trước đó hoặc đã ở trạng thái an toàn.")

def restore_full_classes_txt(dataset_directory):
    classes_content = """boots
helmet on
no boots
no helmet
no vest
vest
id_card
no id_card
smoking
phone_usage"""

    target_file_path = os.path.join(dataset_directory, "train", "labels", "classes.txt")
    os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
    with open(target_file_path, "w", encoding="utf-8") as file:
        file.write(classes_content)
    print(f"Đã cập nhật lại đủ 10 class vào: {target_file_path}")

if __name__ == "__main__":
    patch_labelimg_source_code()
    dataset_directory = "/home/cazofi/data/datamodel2/ppe.v1i.yolov8 (1)"
    restore_full_classes_txt(dataset_directory)