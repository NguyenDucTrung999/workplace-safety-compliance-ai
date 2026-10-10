import cv2
import numpy as np

class ROIManagerOOP:
    def __init__(self):
        self.roi_zones_dictionary = {}
        self.worker_profiles_dictionary = {}
        self.track_assigned_zones_dictionary = {}

    def assign_track_zones(self, track_id, allowed_zone_ids_list):
        track_id_str = str(track_id).strip()
        self.track_assigned_zones_dictionary[track_id_str] = [str(z).strip() for z in allowed_zone_ids_list]
        print(f"Đã phân công vùng cho ID {track_id_str}: {self.track_assigned_zones_dictionary[track_id_str]}")

    def set_person_profile(self, track_id, full_name, employee_code=None):
        track_id_str = str(track_id).strip()
        self.worker_profiles_dictionary[track_id_str] = {
            "full_name": full_name.strip(),
            "employee_code": employee_code.strip() if employee_code else ""
        }
        print(f"Đã gán hồ sơ nhân viên [{full_name}] cho ID {track_id_str}")

    def get_person_display_name(self, track_id):
        track_id_str = str(track_id).strip()
        if track_id_str in self.worker_profiles_dictionary:
            profile = self.worker_profiles_dictionary[track_id_str]
            full_name = profile["full_name"]
            emp_code = profile.get("employee_code", "")
            if emp_code:
                return f"{full_name} ({emp_code}, ID:{track_id_str})"
            else:
                return f"{full_name} (ID:{track_id_str})"
        else:
            return f"Người {track_id_str}"

    def add_zone(self, zone_id, zone_name, polygon_points_list, allowed_person_ids_list=None, zone_type=None, required_ppe_list=None):
        if allowed_person_ids_list is None:
            allowed_person_ids_list = []
        else:
            allowed_person_ids_list = list(allowed_person_ids_list)

        if required_ppe_list is None:
            required_ppe_list = []
        else:
            required_ppe_list = list(required_ppe_list)

        numpy_polygon = np.array(polygon_points_list, dtype=np.int32)
        
        self.roi_zones_dictionary[zone_id] = {
            "zone_id": zone_id,
            "zone_name": zone_name,
            "polygon_points": numpy_polygon,
            "raw_points": polygon_points_list,
            "allowed_person_ids_list": allowed_person_ids_list,
            "zone_type": zone_type,
            "required_ppe_list": required_ppe_list
        }
        print(f"Đã tạo vùng ROI [{zone_id}] - {zone_name} với danh sách cho phép: {allowed_person_ids_list}")

    def remove_zone(self, zone_id):
        if zone_id in self.roi_zones_dictionary:
            removed_zone = self.roi_zones_dictionary.pop(zone_id)
            print(f"Đã xóa vùng ROI [{zone_id}]")
            return removed_zone
        else:
            print(f"Không tìm thấy vùng ROI [{zone_id}] để xóa")
            return None

    def _is_person_allowed(self, track_id_str, allowed_list):
        if not allowed_list:
            return True
        if (track_id_str in allowed_list) or (f"Người {track_id_str}" in allowed_list) or (f"ID {track_id_str}" in allowed_list):
            return True
        profile = self.worker_profiles_dictionary.get(track_id_str)
        if profile:
            emp_name = profile.get("full_name", "")
            emp_code = profile.get("employee_code", "")
            if (emp_name and emp_name in allowed_list) or (emp_code and emp_code in allowed_list):
                return True
        return False

    def add_allowed_person_to_zone(self, zone_id, person_id):
        if zone_id in self.roi_zones_dictionary:
            person_str = str(person_id).strip()
            allowed_list = self.roi_zones_dictionary[zone_id]["allowed_person_ids_list"]
            if person_str not in allowed_list:
                allowed_list.append(person_str)
                print(f"Đã thêm [{person_str}] vào {zone_id}")
            return True
        else:
            print(f"Không tìm thấy vùng [{zone_id}]")
            return False

    def remove_allowed_person_from_zone(self, zone_id, person_id):
        if zone_id in self.roi_zones_dictionary:
            person_str = str(person_id).strip()
            allowed_list = self.roi_zones_dictionary[zone_id]["allowed_person_ids_list"]
            if person_str in allowed_list:
                allowed_list.remove(person_str)
                print(f"Đã xóa [{person_str}] khỏi {zone_id}")
            return True
        else:
            return False

    def evaluate_person_roi(self, track_id, bounding_box, ppe_detections_list):
        x_min, y_min, x_max, y_max = bounding_box
        foot_x_center = float((x_min + x_max) / 2.0)
        foot_y_bottom = float(y_max)
        check_point = (foot_x_center, foot_y_bottom)

        roi_evaluations_list = []
        track_id_str = str(track_id)

        for zone_id, zone_data in self.roi_zones_dictionary.items():
            numpy_polygon = zone_data["polygon_points"]
            test_result = cv2.pointPolygonTest(numpy_polygon, check_point, measureDist=False)

            if test_result >= 0:
                allowed_list = zone_data["allowed_person_ids_list"]
                zone_type = zone_data.get("zone_type", None)
                required_ppe_list = zone_data.get("required_ppe_list", [])
                zone_violations = []

                display_name = self.get_person_display_name(track_id)

                if track_id_str in self.track_assigned_zones_dictionary:
                    assigned_zones = self.track_assigned_zones_dictionary[track_id_str]
                    if zone_id not in assigned_zones:
                        zone_violations.append({
                            "violation_type": "UNAUTHORIZED_ZONE_ACCESS",
                            "description": f"{display_name} đi sai vùng phân công tại {zone_data['zone_name']}"
                        })

                if zone_type == "RESTRICTED_ZONE":
                    if not self._is_person_allowed(track_id_str, allowed_list):
                        zone_violations.append({
                            "violation_type": "RESTRICTED_ZONE_INTRUSION",
                            "description": f"{display_name} xâm nhập Vùng cấm {zone_data['zone_name']}"
                        })
                elif zone_type == "MANDATORY_PPE_ZONE":
                    detected_class_names = [ppe_item["class_name"] for ppe_item in ppe_detections_list]
                    for req_item in required_ppe_list:
                        if req_item == "vest":
                            if ("vest" not in detected_class_names) or ("no vest" in detected_class_names):
                                zone_violations.append({
                                    "violation_type": "MISSING_VEST_IN_ZONE",
                                    "description": f"{display_name} thiếu áo phản quang trong {zone_data['zone_name']}"
                                })
                        elif req_item == "helmet on":
                            if ("helmet on" not in detected_class_names) or ("no helmet" in detected_class_names):
                                zone_violations.append({
                                    "violation_type": "MISSING_HELMET_IN_ZONE",
                                    "description": f"{display_name} thiếu mũ bảo hộ trong {zone_data['zone_name']}"
                                })
                        else:
                            if req_item not in detected_class_names:
                                zone_violations.append({
                                    "violation_type": "MISSING_MANDATORY_PPE",
                                    "description": f"{display_name} thiếu {req_item} trong {zone_data['zone_name']}"
                                })
                else:
                    if len(allowed_list) > 0 and not self._is_person_allowed(track_id_str, allowed_list):
                        zone_violations.append({
                            "violation_type": "UNAUTHORIZED_INTRUSION",
                            "description": f"{display_name} xâm nhập trái phép {zone_data['zone_name']}"
                        })

                roi_evaluations_list.append({
                    "zone_id": zone_id,
                    "zone_name": zone_data["zone_name"],
                    "is_inside": True,
                    "zone_violations": zone_violations
                })

        return roi_evaluations_list

    def draw_roi_zones(self, frame):
        if frame is None or frame.size == 0:
            return frame

        annotated_frame = frame.copy()

        for zone_id, zone_data in self.roi_zones_dictionary.items():
            numpy_polygon = zone_data["polygon_points"]
            border_color = (0, 210, 255)

            cv2.polylines(annotated_frame, [numpy_polygon], isClosed=True, color=border_color, thickness=2)
            
            first_point = numpy_polygon[0]
            label_text = f"{zone_data['zone_name']}"
            cv2.putText(annotated_frame, label_text, (int(first_point[0]), int(first_point[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, border_color, 2)

        return annotated_frame
