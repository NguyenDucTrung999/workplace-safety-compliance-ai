import os
import glob
import re

def fix_class_ids(target_directory):
    # Biểu thức chính quy (regex) để lấy riêng ID ở đầu dòng (group 1) 
    # và phần khoảng trắng cùng toàn bộ tọa độ đằng sau (group 2)
    line_pattern = re.compile(r"^\s*(\d+)(\s+.*)")
    
    modified_files_count = 0
    modified_lines_count = 0
    
    # Lấy toàn bộ danh sách file txt trong thư mục chỉ định
    text_files = glob.glob(os.path.join(target_directory, "*.txt"))
    
    for text_path in text_files:
        with open(text_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            
        new_lines = []
        file_was_modified = False
        
        for line in lines:
            # Bỏ qua nếu là dòng trống
            if not line.strip():
                continue
                
            match = line_pattern.match(line)
            
            if match:
                class_id = match.group(1)
                rest_of_line = match.group(2)
                
                # Cập nhật id 43 và id 1 về thành 0
                if class_id == "43" or class_id == "1":
                    new_line = "0" + rest_of_line
                    
                    # Đảm bảo giữ đúng định dạng xuống dòng của YOLO
                    if not new_line.endswith("\n"):
                        new_line = new_line + "\n"
                        
                    new_lines.append(new_line)
                    file_was_modified = True
                    modified_lines_count += 1
                else:
                    # Nếu là ID khác (chẳng hạn đã là 0), giữ nguyên
                    new_lines.append(line)
            else:
                # Nếu không khớp định dạng YOLO thì giữ nguyên
                new_lines.append(line)
                
        # Nếu có sự thay đổi thì mới tiến hành ghi đè file
        if file_was_modified == True:
            with open(text_path, "w", encoding="utf-8") as file:
                file.writelines(new_lines)
            modified_files_count += 1
            
    print("Quá trình sửa lỗi nhãn đã hoàn tất.")
    print(f"Tổng số file text đã được ghi đè sửa lỗi là: {modified_files_count}")
    print(f"Tổng số dòng nhãn đã được chuyển về ID 0 là: {modified_lines_count}")

if __name__ == "__main__":
    target_directory = "/home/cazofi/data/model1data/all_labels"
    fix_class_ids(target_directory)