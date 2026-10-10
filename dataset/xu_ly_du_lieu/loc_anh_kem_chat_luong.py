import os
import glob

def remove_small_images_and_labels(dataset_directory, size_threshold_kb=65):
    # Quy đổi ngưỡng KB sang Bytes
    size_threshold_bytes = size_threshold_kb * 1024
    
    splits = ["train"]
    image_extensions = [".jpg", ".jpeg", ".png"]
    
    deleted_small_images_count = 0
    deleted_small_labels_count = 0

    for split in splits:
        images_directory = os.path.join(dataset_directory, split, "images")
        labels_directory = os.path.join(dataset_directory, split, "labels")
        
        if not os.path.exists(images_directory):
            continue
            
        # Lấy danh sách tất cả các file ảnh theo đuôi mở rộng
        image_files = []
        for extension in image_extensions:
            image_files.extend(glob.glob(os.path.join(images_directory, "*" + extension)))
            
        for image_path in image_files:
            image_size_bytes = os.path.getsize(image_path)
            
            # Nếu dung lượng ảnh nhỏ hơn ngưỡng cài đặt
            if image_size_bytes < size_threshold_bytes:
                # 1. Xóa file ảnh
                os.remove(image_path)
                deleted_small_images_count += 1
                
                # 2. Tìm và xóa file label (.txt) tương ứng nếu có
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                label_path = os.path.join(labels_directory, base_name + ".txt")
                
                if os.path.exists(label_path):
                    os.remove(label_path)
                    deleted_small_labels_count += 1

    print(f"Quá trình dọn dẹp các ảnh dưới {size_threshold_kb} KB hoàn tất.")
    print(f"Tổng số ảnh bị xóa: {deleted_small_images_count}")
    print(f"Tổng số nhãn bị xóa tương ứng: {deleted_small_labels_count}")

if __name__ == "__main__":
    # Chỉ cần truyền đường dẫn gốc của dataset
    dataset_directory = "/home/cazofi/data/datamodel2/ppe.v1i.yolov8 (1)/train/images"
    remove_small_images_and_labels(dataset_directory, size_threshold_kb=65)