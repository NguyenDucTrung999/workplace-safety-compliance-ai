import os
import glob

def remove_orphan_labels(dataset_directory):
    splits = ["train", "valid", "test"]
    image_extensions = [".jpg", ".jpeg", ".png"]
    
    deleted_orphan_labels_count = 0
    
    for split in splits:
        labels_directory = os.path.join(dataset_directory, split, "labels")
        images_directory = os.path.join(dataset_directory, split, "images")
        
        if not os.path.exists(labels_directory):
            continue
            
        text_files = glob.glob(os.path.join(labels_directory, "*.txt"))
        
        for text_path in text_files:
            file_name = os.path.basename(text_path)
            
            # CƠ CHẾ BẢO VỆ: Bỏ qua file classes.txt để script không xóa nhầm file này
            if file_name == "classes.txt":
                continue
                
            base_name = os.path.splitext(file_name)[0]
            
            image_found = False
            
            # Kiểm tra xem có ảnh nào tồn tại khớp với tên của label không
            for extension in image_extensions:
                image_path = os.path.join(images_directory, base_name + extension)
                if os.path.exists(image_path):
                    image_found = True
                    break
            
            # Nếu duyệt qua các đuôi ảnh mà vẫn không thấy file ảnh, tiến hành xóa file label mồ côi
            if image_found == False:
                os.remove(text_path)
                deleted_orphan_labels_count += 1

    print("Quá trình kiểm tra và dọn dẹp nhãn mồ côi hoàn tất.")
    print(f"Tổng số file nhãn (label) bị xóa do không có ảnh tương ứng là: {deleted_orphan_labels_count}")

if __name__ == "__main__":
    dataset_directory = "/home/cazofi/data/datamodel2/ppe.v1i.yolov8 (1)"
    remove_orphan_labels(dataset_directory)