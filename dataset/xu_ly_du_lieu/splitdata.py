import os
import glob
import random
import shutil
import yaml

def split_dataset_and_create_yaml(source_images_directory, source_labels_directory, output_directory, class_names):
    # 1. Cài đặt tỷ lệ phân chia (90% Train, 5% Valid, 5% Test)
    train_ratio = 0.90
    validation_ratio = 0.05
    # test_ratio tự động lấy phần còn lại (0.05)
    
    # Lấy danh sách tất cả các ảnh
    image_extensions = [".jpg", ".jpeg", ".png"]
    all_image_paths = []
    
    for extension in image_extensions:
        all_image_paths.extend(glob.glob(os.path.join(source_images_directory, "*" + extension)))
        
    valid_pairs = []
    
    # Kiểm tra chỉ lấy những ảnh có file nhãn .txt tương ứng
    for image_path in all_image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(source_labels_directory, base_name + ".txt")
        
        if os.path.exists(label_path):
            valid_pairs.append((image_path, label_path))
            
    if len(valid_pairs) == 0:
        print("Lỗi: Không tìm thấy cặp ảnh và nhãn nào hợp lệ. Vui lòng kiểm tra lại đường dẫn.")
        return
        
    # 2. Xáo trộn ngẫu nhiên danh sách dữ liệu cố định với seed 42
    random.seed(42)
    random.shuffle(valid_pairs)
    
    total_files = len(valid_pairs)
    train_count = int(total_files * train_ratio)
    validation_count = int(total_files * validation_ratio)
    
    # 3. Phân chia danh sách theo chỉ số
    train_pairs = valid_pairs[0 : train_count]
    validation_pairs = valid_pairs[train_count : train_count + validation_count]
    test_pairs = valid_pairs[train_count + validation_count : total_files]
    
    # 4. Khởi tạo cấu trúc thư mục đích
    splits = ["train", "valid", "test"]
    for split in splits:
        os.makedirs(os.path.join(output_directory, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_directory, split, "labels"), exist_ok=True)
        
    # Hàm hỗ trợ sao chép cặp ảnh và nhãn
    def copy_pairs_to_split(pairs_list, split_name):
        for image_source, label_source in pairs_list:
            image_name = os.path.basename(image_source)
            label_name = os.path.basename(label_source)
            
            image_destination = os.path.join(output_directory, split_name, "images", image_name)
            label_destination = os.path.join(output_directory, split_name, "labels", label_name)
            
            shutil.copy2(image_source, image_destination)
            shutil.copy2(label_source, label_destination)
            
    # 5. Thực hiện sao chép file
    print("Đang sao chép tập train (90%)...")
    copy_pairs_to_split(train_pairs, "train")
    
    print("Đang sao chép tập valid (5%)...")
    copy_pairs_to_split(validation_pairs, "valid")
    
    print("Đang sao chép tập test (5%)...")
    copy_pairs_to_split(test_pairs, "test")
    
    # 6. Tạo file cấu hình data.yaml chuẩn 10 class
    yaml_path = os.path.join(output_directory, "data.yaml")
    yaml_content = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": len(class_names),
        "names": class_names
    }
    
    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.dump(yaml_content, file, default_flow_style=None, sort_keys=False)
        
    print("--------------------------------------------------")
    print("Quá trình phân chia dữ liệu và khởi tạo data.yaml hoàn tất!")
    print(f"Tổng số cặp ảnh-nhãn hợp lệ xử lý: {total_files}")
    print(f"Số lượng tập Train (90%): {len(train_pairs)}")
    print(f"Số lượng tập Valid (5%): {len(validation_pairs)}")
    print(f"Số lượng tập Test (5%): {len(test_pairs)}")
    print(f"File data.yaml chuẩn 10 class được sinh tại: {yaml_path}")


if __name__ == "__main__":
    # 1. Thư mục ảnh nguồn (lấy từ ảnh chụp màn hình bạn gửi)
    source_images_directory = "/home/cazofi/data/datamodel2/ppe.v1i.yolov8 (1)/train/images"
    
    # 2. Thư mục nhãn nguồn (lấy từ ảnh chụp màn hình bạn gửi)
    source_labels_directory = "/home/cazofi/data/datamodel2/ppe.v1i.yolov8 (1)/train/labels"
    
    # 3. Thư mục đích lưu bộ dữ liệu đã chia 90-5-5 hoàn chỉnh
    output_directory = "/home/cazofi/data/datamodel2/dataset_ppe_10class"
    
    # 4. Danh sách 10 Class chuẩn của bạn
    class_names = [
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
    
    split_dataset_and_create_yaml(
        source_images_directory, 
        source_labels_directory, 
        output_directory, 
        class_names
    )