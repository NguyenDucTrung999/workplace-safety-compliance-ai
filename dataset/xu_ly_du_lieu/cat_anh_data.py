import os
import glob
import cv2

def extract_frames_from_video_folder(videos_directory, output_directory, frame_interval=10):
    # 1. Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 2. Lấy danh sách tất cả các file video có trong thư mục nguồn
    video_extensions = [".mp4", ".avi", ".mkv", ".mov"]
    all_video_files = []

    for extension in video_extensions:
        all_video_files.extend(glob.glob(os.path.join(videos_directory, "*" + extension)))

    if len(all_video_files) == 0:
        print(f"Lỗi: Không tìm thấy file video nào trong thư mục {videos_directory}")
        return

    print(f"Tìm thấy tổng cộng {len(all_video_files)} file video cần xử lý.")

    total_extracted_images_count = 0

    # 3. Duyệt và xử lý từng file video
    for video_path in all_video_files:
        video_filename = os.path.basename(video_path)
        video_base_name = os.path.splitext(video_filename)[0]

        print(f"Đang xử lý video: {video_filename}")

        video_capture = cv2.VideoCapture(video_path)
        
        if not video_capture.isOpened():
            print(f"Lỗi: Không thể mở file video {video_filename}")
            continue

        frame_counter = 0
        extracted_from_this_video_count = 0

        while True:
            success, frame = video_capture.read()

            if success == False or frame is None:
                break

            frame_counter += 1

            # Điều kiện: Cứ đúng 3 frame thì cắt và lưu 1 ảnh
            if frame_counter % frame_interval == 0:
                extracted_from_this_video_count += 1
                total_extracted_images_count += 1 

                # Tên file ảnh gồm tên video gốc + số thứ tự frame
                output_image_name = f"{video_base_name}_frame_{frame_counter:06d}.jpg"
                output_image_path = os.path.join(output_directory, output_image_name)

                # Ghi file ảnh ra đĩa
                cv2.imwrite(output_image_path, frame)

        video_capture.release()
        print(f"Hoàn thành video {video_filename}: đã cắt được {extracted_from_this_video_count} ảnh.")

    print("--------------------------------------------------")
    print("Quá trình trích xuất hoàn tất!")
    print(f"Tổng số ảnh thu được từ tất cả video: {total_extracted_images_count}")
    print(f"Tất cả ảnh đã được lưu tại thư mục: {output_directory}")

if __name__ == "__main__":
    videos_directory = "/home/cazofi/Videos"
    output_directory = "/home/cazofi/data/datamodel2/anhtuvideo"
    extract_frames_from_video_folder(videos_directory, output_directory, frame_interval=10)