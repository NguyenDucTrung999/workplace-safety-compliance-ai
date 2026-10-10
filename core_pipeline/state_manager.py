import time
import datetime
import os
import sys
import sqlite3
import cv2
import threading

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import DB_PATH, SNAPSHOTS_DIR

class PersonSafetyState:
    def __init__(self, track_id, initial_timestamp=None):
        if initial_timestamp is None:
            initial_timestamp = time.time()
        else:
            initial_timestamp = float(initial_timestamp)

        self.track_id = track_id
        self.first_seen_timestamp = initial_timestamp
        self.last_seen_timestamp = initial_timestamp
        self.is_boundary_cutoff = False
        
        self.current_ppe_status = {
            "helmet": "unknown",
            "vest": "unknown",
            "boots": "unknown",
            "id_card": "unknown",
            "smoking": False,
            "phone_usage": False
        }
        
        self.violation_confirmation_threshold_frames = 6
        self.remediation_confirmation_threshold_frames = 6

        self.consecutive_no_helmet_frames = 0
        self.consecutive_helmet_on_frames = 0

        self.consecutive_no_vest_frames = 0
        self.consecutive_vest_frames = 0

        self.consecutive_no_boots_frames = 0
        self.consecutive_boots_frames = 0

        self.full_body_observed_frames = 0
        self.id_card_grace_period_frames = 35
        self.consecutive_id_card_frames = 0
        self.consecutive_missing_id_frames = 0
        self.id_card_removal_threshold_frames = 45

        self.consecutive_smoking_frames = 0
        self.consecutive_no_smoking_frames = 0
        self.consecutive_phone_frames = 0
        self.consecutive_no_phone_frames = 0
        
        self.ppe_detections_history = []
        self.history_window_size = 8
        self.current_roi_evaluations = []
        self.active_violations_dictionary = {}
        self.resolved_violations_list = []

    def _update_symmetric_ppe_item(self, item_name, pos_label, neg_label, current_classes, neg_attr, pos_attr):
        consec_neg = getattr(self, neg_attr)
        consec_pos = getattr(self, pos_attr)

        if neg_label in current_classes:
            consec_neg += 1
            consec_pos = 0
        elif pos_label in current_classes:
            consec_pos += 1
            consec_neg = 0
        else:
            if self.current_ppe_status[item_name] != "violation":
                consec_neg = 0
            consec_pos = 0

        curr_status = self.current_ppe_status[item_name]
        if curr_status == "violation":
            if consec_pos >= self.remediation_confirmation_threshold_frames:
                self.current_ppe_status[item_name] = "compliant"
                consec_neg = 0
        elif curr_status == "compliant":
            if consec_neg >= self.violation_confirmation_threshold_frames:
                self.current_ppe_status[item_name] = "violation"
                consec_pos = 0
        else:
            if consec_neg >= self.violation_confirmation_threshold_frames:
                self.current_ppe_status[item_name] = "violation"
            elif consec_pos >= 2:
                self.current_ppe_status[item_name] = "compliant"

        setattr(self, neg_attr, consec_neg)
        setattr(self, pos_attr, consec_pos)

    def update_ppe_status(self, ppe_detections_list, is_boundary_cutoff=False, current_timestamp=None, roi_evaluations=None):
        if current_timestamp is None:
            current_timestamp = time.time()
        else:
            current_timestamp = float(current_timestamp)
            
        self.last_seen_timestamp = current_timestamp
        self.is_boundary_cutoff = is_boundary_cutoff
        self.current_roi_evaluations = roi_evaluations if roi_evaluations is not None else []

        if not is_boundary_cutoff:
            self.full_body_observed_frames += 1
        
        # Spatial anatomy filtering
        current_frame_class_confidences = {}

        for ppe_item in ppe_detections_list:
            class_name = ppe_item["class_name"]
            box_confidence = float(ppe_item.get("confidence", 0.5))
            relative_y_center = ppe_item.get("relative_y_center", None)

            is_spatially_valid = True
            if relative_y_center is not None:
                if class_name in ["helmet on", "no helmet"]:
                    is_spatially_valid = (relative_y_center <= 0.45)
                elif class_name in ["vest", "no vest", "id_card", "no id_card", "no id card", "IDcard", "ID_card"]:
                    is_spatially_valid = (0.10 <= relative_y_center <= 0.85)
                elif class_name in ["smoking", "phone_usage"]:
                    is_spatially_valid = (relative_y_center <= 0.85)
                elif class_name in ["boots", "no boots"]:
                    is_spatially_valid = (relative_y_center >= 0.50)

            if is_spatially_valid:
                current_frame_class_confidences[class_name] = max(
                    current_frame_class_confidences.get(class_name, 0.0), 
                    box_confidence
                )

        opposing_pairs = [("no helmet", "helmet on"), ("no vest", "vest"), ("no boots", "boots")]
        for neg_lbl, pos_lbl in opposing_pairs:
            if neg_lbl in current_frame_class_confidences and pos_lbl in current_frame_class_confidences:
                if current_frame_class_confidences[neg_lbl] > current_frame_class_confidences[pos_lbl] + 0.05:
                    del current_frame_class_confidences[pos_lbl]
                else:
                    del current_frame_class_confidences[neg_lbl]

        current_frame_class_names = set(current_frame_class_confidences.keys())

        self.ppe_detections_history.append(current_frame_class_names)
        if len(self.ppe_detections_history) > self.history_window_size:
            self.ppe_detections_history.pop(0)

        class_appearance_counts = {}
        total_history_frames = len(self.ppe_detections_history)

        for frame_classes in self.ppe_detections_history:
            for c_name in frame_classes:
                class_appearance_counts[c_name] = class_appearance_counts.get(c_name, 0) + 1

        confirmed_class_names = set()
        majority_threshold = max(1, int(total_history_frames * 0.40))

        for c_name, count in class_appearance_counts.items():
            if count >= majority_threshold:
                confirmed_class_names.add(c_name)

        self._update_symmetric_ppe_item(
            "helmet", "helmet on", "no helmet", current_frame_class_names,
            "consecutive_no_helmet_frames", "consecutive_helmet_on_frames"
        )
        self._update_symmetric_ppe_item(
            "vest", "vest", "no vest", current_frame_class_names,
            "consecutive_no_vest_frames", "consecutive_vest_frames"
        )
        self._update_symmetric_ppe_item(
            "boots", "boots", "no boots", current_frame_class_names,
            "consecutive_no_boots_frames", "consecutive_boots_frames"
        )

        valid_id_names = ["id_card", "id card", "IDcard", "ID_card"]
        has_id_card_detected = any(v in current_frame_class_names for v in valid_id_names)

        if has_id_card_detected:
            self.consecutive_id_card_frames += 1
            self.consecutive_missing_id_frames = 0
        else:
            self.consecutive_missing_id_frames += 1
            self.consecutive_id_card_frames = 0

        if self.current_ppe_status["id_card"] == "violation":
            if self.consecutive_id_card_frames >= self.remediation_confirmation_threshold_frames:
                self.current_ppe_status["id_card"] = "compliant"
                self.consecutive_missing_id_frames = 0
        elif self.current_ppe_status["id_card"] == "compliant":
            # Session latching: Khóa cố định compliant để tránh báo động giả khi quay lưng hoặc che khuất
            pass
        else:
            if self.consecutive_id_card_frames >= 2:
                self.current_ppe_status["id_card"] = "compliant"
            elif self.full_body_observed_frames >= self.id_card_grace_period_frames:
                self.current_ppe_status["id_card"] = "violation"

        if "smoking" in current_frame_class_names:
            self.consecutive_smoking_frames += 1
            self.consecutive_no_smoking_frames = 0
            if self.consecutive_smoking_frames >= 2:
                self.current_ppe_status["smoking"] = True
        else:
            self.consecutive_no_smoking_frames += 1
            self.consecutive_smoking_frames = 0

        if "phone_usage" in current_frame_class_names:
            self.consecutive_phone_frames += 1
            self.consecutive_no_phone_frames = 0
            if self.consecutive_phone_frames >= 2:
                self.current_ppe_status["phone_usage"] = True
        else:
            self.consecutive_no_phone_frames += 1
            self.consecutive_phone_frames = 0

        new_events_list = self._evaluate_violation_events(current_timestamp)
        return new_events_list

    def _evaluate_violation_events(self, current_timestamp):
        new_events_list = []

        is_roi_unauthorized = False
        for roi_eval in self.current_roi_evaluations:
            for z_viol in roi_eval.get("zone_violations", []):
                if z_viol.get("violation_type") == "UNAUTHORIZED_INTRUSION":
                    is_roi_unauthorized = True

        violations_check_map = {
            "no_helmet": self.current_ppe_status["helmet"] == "violation",
            "no_vest": self.current_ppe_status["vest"] == "violation",
            "no_boots": self.current_ppe_status["boots"] == "violation",
            "no_id_card": self.current_ppe_status["id_card"] == "violation",
            "smoking": self.current_ppe_status["smoking"],
            "phone_usage": self.current_ppe_status["phone_usage"],
            "UNAUTHORIZED_INTRUSION": is_roi_unauthorized
        }

        for violation_type_name, is_violating in violations_check_map.items():
            if is_violating:
                if violation_type_name not in self.active_violations_dictionary:
                    start_time_string = datetime.datetime.fromtimestamp(current_timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    violation_record = {
                        "track_id": self.track_id,
                        "violation_type": violation_type_name,
                        "start_timestamp": current_timestamp,
                        "start_time_string": start_time_string,
                        "status": "ACTIVE"
                    }
                    self.active_violations_dictionary[violation_type_name] = violation_record
                    new_events_list.append(violation_record)
                else:
                    existing_record = self.active_violations_dictionary[violation_type_name]
                    existing_record["last_detected_timestamp"] = current_timestamp
            else:
                if violation_type_name in self.active_violations_dictionary:
                    resolved_record = self.active_violations_dictionary.pop(violation_type_name)
                    end_time_string = datetime.datetime.fromtimestamp(current_timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    resolved_record["end_timestamp"] = current_timestamp
                    resolved_record["end_time_string"] = end_time_string
                    resolved_record["duration_seconds"] = round(current_timestamp - resolved_record["start_timestamp"], 2)
                    resolved_record["status"] = "RESOLVED"
                    self.resolved_violations_list.append(resolved_record)
                    
                    resolution_event = {
                        "track_id": self.track_id,
                        "violation_type": violation_type_name,
                        "event_type": "VIOLATION_RESOLVED",
                        "resolution_record": resolved_record
                    }
                    new_events_list.append(resolution_event)

        return new_events_list


class SafetyStateManagerOOP:
    def __init__(self, max_disappeared_seconds=120, db_path=None, snapshots_dir=None):
        if db_path is None:
            db_path = DB_PATH
        else:
            db_path = str(db_path)

        if snapshots_dir is None:
            snapshots_dir = SNAPSHOTS_DIR
        else:
            snapshots_dir = str(snapshots_dir)

        self.tracked_person_states = {}
        self.max_disappeared_seconds = max_disappeared_seconds
        self.active_db_rows = {}
        
        os.makedirs(os.path.abspath(snapshots_dir), exist_ok=True)
        self.snapshots_dir = snapshots_dir

        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                violation_type TEXT,
                start_timestamp REAL,
                start_time_string TEXT,
                end_timestamp REAL,
                end_time_string TEXT,
                duration_seconds REAL,
                status TEXT,
                full_snapshot_path TEXT,
                crop_snapshot_path TEXT
            )
        ''')
        cursor.execute("PRAGMA table_info(violations)")
        columns = [row[1] for row in cursor.fetchall()]
        if "full_snapshot_path" not in columns:
            cursor.execute("ALTER TABLE violations ADD COLUMN full_snapshot_path TEXT")
        if "crop_snapshot_path" not in columns:
            cursor.execute("ALTER TABLE violations ADD COLUMN crop_snapshot_path TEXT")

        conn.commit()
        conn.close()

    def _create_snapshots(self, original_frame, bounding_box, track_id, violation_type, timestamp_float):
        if original_frame is None or original_frame.size == 0 or bounding_box is None:
            return None, None

        image_height, image_width = original_frame.shape[:2]
        x_min, y_min, x_max, y_max = [int(coord) for coord in bounding_box]
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(image_width, x_max), min(image_height, y_max)

        time_tag = datetime.datetime.fromtimestamp(timestamp_float).strftime("%Y%m%d_%H%M%S")
        full_filename = f"full_id{track_id}_{violation_type}_{time_tag}.jpg"
        crop_filename = f"crop_id{track_id}_{violation_type}_{time_tag}.jpg"

        full_filepath = os.path.join(self.snapshots_dir, full_filename)
        crop_filepath = os.path.join(self.snapshots_dir, crop_filename)

        full_frame_annotated = original_frame.copy()
        cv2.rectangle(full_frame_annotated, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(
            full_frame_annotated,
            f"VIOLATION: {violation_type} (Person {track_id})",
            (x_min, max(15, y_min - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
        cv2.imwrite(full_filepath, full_frame_annotated)

        box_width = x_max - x_min
        box_height = y_max - y_min
        pad_x = int(box_width * 0.10)
        pad_y = int(box_height * 0.10)

        crop_x_min = max(0, x_min - pad_x)
        crop_y_min = max(0, y_min - pad_y)
        crop_x_max = min(image_width, x_max + pad_x)
        crop_y_max = min(image_height, y_max + pad_y)

        if crop_x_max > crop_x_min and crop_y_max > crop_y_min:
            cropped_image = original_frame[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            cv2.imwrite(crop_filepath, cropped_image)
        else:
            cv2.imwrite(crop_filepath, original_frame)

        return full_filename, crop_filename

    def update(self, persons_detail_list, current_timestamp=None, original_frame=None):
        if current_timestamp is None:
            current_timestamp = time.time()
        else:
            current_timestamp = float(current_timestamp)

        all_new_events_list = []
        active_track_ids = set()
        person_bounding_boxes_map = {}

        for person_detail in persons_detail_list:
            track_id = person_detail["track_id"]
            active_track_ids.add(track_id)
            ppe_detections_list = person_detail.get("ppe_detections", [])
            is_boundary_cutoff = person_detail.get("is_boundary_cutoff", False)
            roi_evaluations = person_detail.get("roi_evaluations", [])
            bounding_box = person_detail.get("bounding_box", None)
            if bounding_box is not None:
                person_bounding_boxes_map[track_id] = bounding_box

            if track_id not in self.tracked_person_states:
                person_state = PersonSafetyState(track_id=track_id, initial_timestamp=current_timestamp)
                self.tracked_person_states[track_id] = person_state
            else:
                person_state = self.tracked_person_states[track_id]

            new_events = person_state.update_ppe_status(
                ppe_detections_list=ppe_detections_list, 
                is_boundary_cutoff=is_boundary_cutoff, 
                current_timestamp=current_timestamp,
                roi_evaluations=roi_evaluations
            )
            all_new_events_list.extend(new_events)

        self._cleanup_stale_states(active_track_ids, current_timestamp)
        if len(all_new_events_list) > 0:
            frame_snapshot_copy = original_frame.copy() if original_frame is not None else None
            boxes_map_copy = person_bounding_boxes_map.copy() if person_bounding_boxes_map is not None else {}
            events_copy = list(all_new_events_list)
            db_thread = threading.Thread(
                target=self._sync_events_to_db,
                args=(events_copy,),
                kwargs={"original_frame": frame_snapshot_copy, "bounding_boxes_map": boxes_map_copy},
                daemon=True
            )
            db_thread.start()

        summary_report = self.get_summary_report()
        return summary_report, all_new_events_list

    def _sync_events_to_db(self, all_new_events_list, original_frame=None, bounding_boxes_map=None):
        if bounding_boxes_map is None:
            bounding_boxes_map = {}

        # Thu thập các track_id có sự kiện thay đổi
        affected_track_ids = set()
        for event in all_new_events_list:
            t_id = event.get("track_id")
            if t_id is not None:
                affected_track_ids.add(t_id)

        if len(affected_track_ids) == 0:
            return

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            for track_id in affected_track_ids:
                if track_id not in self.tracked_person_states:
                    continue

                person_state = self.tracked_person_states[track_id]
                current_active_violations = list(person_state.active_violations_dictionary.keys())

                if len(current_active_violations) > 0:
                    joined_violations_string = ", ".join(current_active_violations)
                    earliest_start_timestamp = min(
                        rec["start_timestamp"] for rec in person_state.active_violations_dictionary.values()
                    )
                    start_time_string = datetime.datetime.fromtimestamp(earliest_start_timestamp).strftime("%Y-%m-%d %H:%M:%S")

                    if track_id in self.active_db_rows:
                        db_id = self.active_db_rows[track_id]
                        cursor.execute('''
                            UPDATE violations
                            SET violation_type = ?, status = ?
                            WHERE id = ?
                        ''', (joined_violations_string, "ACTIVE", db_id))
                    else:
                        full_snapshot_file = None
                        crop_snapshot_file = None

                        if original_frame is not None and track_id in bounding_boxes_map:
                            person_box = bounding_boxes_map[track_id]
                            full_snapshot_file, crop_snapshot_file = self._create_snapshots(
                                original_frame=original_frame,
                                bounding_box=person_box,
                                track_id=track_id,
                                violation_type=joined_violations_string.replace(", ", "_"),
                                timestamp_float=earliest_start_timestamp
                            )

                        cursor.execute('''
                            INSERT INTO violations (
                                track_id,
                                violation_type,
                                start_timestamp,
                                start_time_string,
                                status,
                                full_snapshot_path,
                                crop_snapshot_path
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            track_id,
                            joined_violations_string,
                            earliest_start_timestamp,
                            start_time_string,
                            "ACTIVE",
                            full_snapshot_file,
                            crop_snapshot_file
                        ))
                        self.active_db_rows[track_id] = cursor.lastrowid
                else:
                    if track_id in self.active_db_rows:
                        db_id = self.active_db_rows.pop(track_id)
                        cursor.execute("DELETE FROM violations WHERE id = ?", (db_id,))

            conn.commit()
        finally:
            conn.close()

    def _cleanup_stale_states(self, active_track_ids, current_timestamp):
        stale_track_ids = []
        for track_id, person_state in self.tracked_person_states.items():
            if track_id not in active_track_ids:
                elapsed_seconds = current_timestamp - person_state.last_seen_timestamp
                if elapsed_seconds > self.max_disappeared_seconds:
                    stale_track_ids.append(track_id)

        if len(stale_track_ids) > 0:
            for track_id in stale_track_ids:
                # Khi người rời khỏi camera: Không thay đổi bất kỳ trạng thái nào trong DB, giữ nguyên lỗi vi phạm đã ghi nhận
                self.active_db_rows.pop(track_id, None)
                del self.tracked_person_states[track_id]

    def get_summary_report(self):
        total_tracked = len(self.tracked_person_states)
        total_violations = 0
        person_summaries_list = []

        for track_id, person_state in self.tracked_person_states.items():
            active_violations_count = len(person_state.active_violations_dictionary)
            total_violations += active_violations_count
            
            last_seen_time_str = datetime.datetime.fromtimestamp(person_state.last_seen_timestamp).strftime("%H:%M:%S")
            person_summary = {
                "track_id": track_id,
                "current_ppe_status": person_state.current_ppe_status,
                "active_violations": list(person_state.active_violations_dictionary.values()),
                "resolved_violations_count": len(person_state.resolved_violations_list),
                "last_seen_timestamp": person_state.last_seen_timestamp,
                "last_seen_time_string": last_seen_time_str,
                "is_boundary_cutoff": person_state.is_boundary_cutoff
            }
            person_summaries_list.append(person_summary)

        return {
            "total_tracked_persons": total_tracked,
            "total_active_violations": total_violations,
            "person_summaries": person_summaries_list
        }

    def reset(self):
        self.tracked_person_states.clear()
        self.active_db_rows.clear()

    def clear_all_logs_in_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM violations")
        conn.commit()
        conn.close()
