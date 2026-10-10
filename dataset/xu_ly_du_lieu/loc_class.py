import os
import glob
import re

def diagnose_dataset_classes(dataset_directory):
    splits = ["train", "valid", "test"]
    line_pattern = re.compile(r"^\s*(\d+)\s+")
    
    found_class_ids = set()
    
    for split in splits:
        labels_directory = os.path.join(dataset_directory, split, "labels")
        
        if not os.path.exists(labels_directory):
            continue
            
        text_files = glob.glob(os.path.join(labels_directory, "*.txt"))
        
        for text_path in text_files:
            if os.path.basename(text_path) == "classes.txt":
                continue
                
            with open(text_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
                
            for line in lines:
                match = line_pattern.match(line)
                if match:
                    found_class_ids.add(int(match.group(1)))
                    
    sorted_ids = sorted(list(found_class_ids))
    print("=== KẾT QUẢ CHẨN ĐOÁN DỮ LIỆU ===")
    print(f"Các ID class thực tế đang có trong file .txt: {sorted_ids}")
    
    if len(sorted_ids) > 0:
        max_id = max(sorted_ids)
        print(f"Chỉ số ID lớn nhất tìm thấy là: {max_id}")
    else:
        max_id = 0
        print("Không tìm thấy ID nào.")
        
    # Kiểm tra số dòng của file classes.txt hiện tại
    classes_txt_path = os.path.join(dataset_directory, "train", "labels", "classes.txt")
    if os.path.exists(classes_txt_path):
        with open(classes_txt_path, "r", encoding="utf-8") as file:
            current_classes = [line.strip() for line in file.readlines() if line.strip()]
            
        print(f"File classes.txt hiện tại đang có: {len(current_classes)} dòng (từ ID 0 đến ID {len(current_classes) - 1}).")
        
        if max_id >= len(current_classes):
            print(f"PHÁT HIỆN NGUYÊN NHÂN LỖI: Trong file .txt có chứa ID {max_id}, nhưng file classes.txt chỉ khai báo tới ID {len(current_classes) - 1}!")
        else:
            print("Cấu hình số dòng trong classes.txt đã khớp với dữ liệu.")
    else:
        print("Không tìm thấy file classes.txt tại đường dẫn train/labels!")

if __name__ == "__main__":
    dataset_directory = "/home/cazofi/data/datamodel2/ppe.v1i.yolov8 (1)"
    diagnose_dataset_classes(dataset_directory)