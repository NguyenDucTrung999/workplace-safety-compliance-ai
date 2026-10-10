import os
import glob
import cv2
from ultralytics import YOLO

# =======================================================================
# KHU VỰC CÀI ĐẶT ĐƯỜNG DẪN VÀ THÔNG SỐ
# =======================================================================

# 1. Đường dẫn đến mô hình nhận diện người Model 1 (trích xuất từ file test mẫu của bạn)
trained_model_path = "/home/cazofi/VNAtech_intern/tuan-thu-an-toan-lao-dong-mr-trung/model/weights/model_human_detection.pt"

# 2. Thư mục chứa các ảnh cần duyệt
input_images_directory = "/home/cazofi/data/datamodel2/anhtuvideo"

# 3. Thư mục lưu kết quả các ảnh đối tượng người đã được cắt (crop)
output_cropped_directory = "/home/cazofi/data/datamodel2/anhtuquaynew"

# Ngưỡng độ tin cậy của mô hình nhận diện (>= 50%)
confidence_threshold = 0.5

# Tỷ lệ padding nới rộng khung (10%)
padding_ratio = 0.10

# =======================================================================
# KHU VỰC XỬ LÝ LÕI
# =======================================================================

def crop_detected_persons():
    # Kiểm tra sự tồn tại của tệp mô hình
    if not os.path.exists(trained_model_path):
        print(f"Lỗi: Không tìm thấy file mô hình tại {trained_model_path}")
        return

    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_cropped_directory):
        os.makedirs(output_cropped_directory)

    # Lấy danh sách tất cả các ảnh từ thư mục nguồn
    image_extensions = [".jpg", ".jpeg", ".png"]
    all_image_paths = []

    for extension in image_extensions:
        all_image_paths.extend(glob.glob(os.path.join(input_images_directory, "*" + extension)))

    if len(all_image_paths) == 0:
        print(f"Lỗi: Không tìm thấy file ảnh nào trong thư mục {input_images_directory}")
        return

    print(f"Đang tải mô hình Model 1 từ: {trained_model_path}")
    model = YOLO(trained_model_path)

    print(f"Tìm thấy {len(all_image_paths)} ảnh cần quét nhận diện.")
    total_cropped_persons_count = 0

    for image_path in all_image_paths:
        base_image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        original_image = cv2.imread(image_path)
        if original_image is None:
            continue

        image_height, image_width = original_image.shape[:2]

        # Chạy nhận diện bằng mô hình YOLO Model 1
        prediction_results = model.predict(source=original_image, conf=confidence_threshold, verbose=False)
        first_result = prediction_results[0]

        detected_boxes = first_result.boxes
        person_index = 0

        for box in detected_boxes:
            # Lấy tọa độ gốc (x_min, y_min, x_max, y_max)
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

            box_width = x_max - x_min
            box_height = y_max - y_min

            # Tính toán khoảng nới rộng 10%
            padding_x = box_width * padding_ratio
            padding_y = box_height * padding_ratio

            # Mở rộng 4 cạnh và ép ranh giới không vượt quá mép bức ảnh
            padded_x_min = max(0, int(x_min - padding_x))
            padded_y_min = max(0, int(y_min - padding_y))
            padded_x_max = min(image_width, int(x_max + padding_x))
            padded_y_max = min(image_height, int(y_max + padding_y))

            # Trích xuất vùng ảnh đối tượng người đã được nới rộng
            cropped_person_image = original_image[padded_y_min:padded_y_max, padded_x_min:padded_x_max]

            # Kiểm tra ảnh cắt hợp lệ trước khi tiến hành lưu
            if cropped_person_image.size > 0:
                person_index += 1
                total_cropped_persons_count += 1

                output_filename = f"{base_image_name}_person_{person_index}.jpg"
                output_filepath = os.path.join(output_cropped_directory, output_filename)

                cv2.imwrite(output_filepath, cropped_person_image)

        print(f"Đã xử lý ảnh {base_image_name}: Cắt được {person_index} người.")

    print("--------------------------------------------------")
    print("Quá trình cắt ảnh đối tượng hoàn tất!")
    print(f"Tổng số ảnh người đã cắt và lưu thành công: {total_cropped_persons_count}")
    print(f"Tất cả kết quả được lưu tại: {output_cropped_directory}")

if __name__ == "__main__":
    crop_detected_persons()