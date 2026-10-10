import os
import glob
import xml.etree.ElementTree as ET
import yaml

def convert_xml_to_yolo(target_directory):
    # 1. Tìm file data.yaml ở thư mục gốc của dataset để không phải dự đoán ID
    # Từ thư mục /train/newlabel, đi ngược lên 2 cấp để lấy thư mục gốc
    dataset_directory = os.path.dirname(os.path.dirname(target_directory))
    yaml_path = os.path.join(dataset_directory, "data.yaml")
    
    if not os.path.exists(yaml_path):
        print(f"Lỗi: Không tìm thấy file {yaml_path} để đọc danh sách classes.")
        return

    with open(yaml_path, "r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)
        
    class_names_list = yaml_data.get("names", [])
    
    # Xây dựng từ điển map tên class sang ID
    class_name_to_id = {}
    if isinstance(class_names_list, list):
        for index, name in enumerate(class_names_list):
            class_name_to_id[name] = str(index)
    elif isinstance(class_names_list, dict):
        for index, name in class_names_list.items():
            class_name_to_id[name] = str(index)
            
    print(f"Đã tự động đọc được danh sách classes từ data.yaml: {class_name_to_id}")

    # 2. Bắt đầu chuyển đổi
    xml_files = glob.glob(os.path.join(target_directory, "*.xml"))
    converted_count = 0
    
    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Lấy kích thước ảnh gốc từ file XML
        size_element = root.find("size")
        if size_element is None:
            continue
            
        image_width = float(size_element.find("width").text)
        image_height = float(size_element.find("height").text)
        
        if image_width == 0 or image_height == 0:
            continue
            
        yolo_lines = []
        
        # Duyệt qua các object có trong XML
        for object_element in root.findall("object"):
            class_name = object_element.find("name").text
            
            # Nếu class name trong xml không tồn tại ở data.yaml thì bỏ qua
            if class_name not in class_name_to_id:
                continue
                
            class_id = class_name_to_id[class_name]
            
            # Lấy tọa độ bounding box
            bndbox = object_element.find("bndbox")
            x_min = float(bndbox.find("xmin").text)
            y_min = float(bndbox.find("ymin").text)
            x_max = float(bndbox.find("xmax").text)
            y_max = float(bndbox.find("ymax").text)
            
            # Chuyển đổi tọa độ sang định dạng chuẩn hóa YOLO (0 đến 1)
            x_center = ((x_min + x_max) / 2.0) / image_width
            y_center = ((y_min + y_max) / 2.0) / image_height
            box_width = (x_max - x_min) / image_width
            box_height = (y_max - y_min) / image_height
            
            # Định dạng dòng văn bản YOLO: class_id x_center y_center width height
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"
            yolo_lines.append(yolo_line)
            
        # Tạo file .txt lưu cùng vị trí để chứa nội dung vừa chuyển đổi
        base_name = os.path.splitext(xml_path)[0]
        txt_path = base_name + ".txt"
        
        with open(txt_path, "w", encoding="utf-8") as file:
            file.writelines(yolo_lines)
            
        # Xóa file .xml cũ để thay thế hoàn toàn
        os.remove(xml_path)
        converted_count += 1
        
    print("Quá trình chuyển đổi hoàn tất.")
    print(f"Tổng số file XML đã được chuyển thành TXT và thay thế thành công: {converted_count}")

if __name__ == "__main__":
    target_directory = "/home/cazofi/data/model1data/ppe.v1i.yolov8/train/newlabel"
    convert_xml_to_yolo(target_directory)