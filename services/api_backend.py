import sys
import os
import time
import datetime
import cv2
import asyncio
import base64
import threading
import sqlite3
import csv
import io
import psutil
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core_pipeline.pipeline_manager import SafetyPipelineManagerOOP
from config import DEFAULT_CAMERA_SOURCE, SNAPSHOTS_DIR, LOGS_DIR

class RTSPCameraStreamWorker:
    def __init__(self, camera_source=None):
        self.camera_source = camera_source
        self.latest_frame = None
        self.frame_id = 0
        self.is_running = False
        self.reconnect_requested = False
        self.lock = threading.Lock()
        self.capture_thread = None

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.capture_thread.start()
            print(f"Đã khởi chạy luồng đọc Camera RTSP: {self.camera_source}")

    def set_camera_source(self, new_camera_source):
        with self.lock:
            self.camera_source = new_camera_source
            self.latest_frame = None
            self.reconnect_requested = True
        print(f"Đã cập nhật nguồn Camera RTSP mới: {new_camera_source}")

    def reconnect(self):
        with self.lock:
            self.latest_frame = None
            self.reconnect_requested = True
        print(f"Yêu cầu reset kết nối Camera RTSP: {self.camera_source}")

    def _update_loop(self):
        video_capture = None

        while self.is_running:
            with self.lock:
                current_source = self.camera_source
                should_reconnect = self.reconnect_requested
                if should_reconnect:
                    self.reconnect_requested = False

            if should_reconnect:
                if video_capture is not None:
                    video_capture.release()
                    video_capture = None

            if current_source is not None and current_source != "":
                if video_capture is None:
                    print(f"Đang thử kết nối tới Nguồn Camera RTSP: {current_source}")
                    try:
                        if isinstance(current_source, int) or (isinstance(current_source, str) and current_source.isdigit()):
                            video_capture = cv2.VideoCapture(int(current_source))
                        else:
                            video_capture = cv2.VideoCapture(current_source)
                        if video_capture is not None and video_capture.isOpened():
                            video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception as error_msg:
                        print(f"Lỗi mở Nguồn RTSP: {str(error_msg)}")
                        video_capture = None

                if video_capture is not None and video_capture.isOpened():
                    grabbed_success, frame = video_capture.read()
                    if grabbed_success:
                        with self.lock:
                            self.latest_frame = frame
                            self.frame_id += 1
                        time.sleep(0.005)
                    else:
                        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        grabbed_again, loop_frame = video_capture.read()
                        if grabbed_again:
                            with self.lock:
                                self.latest_frame = loop_frame
                                self.frame_id += 1
                        else:
                            print("Không thể đọc khung hình từ nguồn RTSP, đang thử kết nối lại...")
                            video_capture.release()
                            video_capture = None
                            time.sleep(1.0)
                else:
                    time.sleep(1.0)
            else:
                if video_capture is not None:
                    video_capture.release()
                    video_capture = None
                time.sleep(0.5)

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.frame_id
            else:
                return None, 0

    def stop(self):
        self.is_running = False
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)
        print("Đã dừng luồng đọc Camera RTSP.")

# Khởi tạo lớp ứng dụng FastAPI
app = FastAPI(title="Industrial Safety Compliance Supervision System")

# Đường dẫn tới thư mục frontend và thư mục snapshots
frontend_directory = os.path.join(project_root, "frontend")
if os.path.exists(frontend_directory):
    app.mount("/static", StaticFiles(directory=frontend_directory), name="static")
else:
    os.makedirs(frontend_directory, exist_ok=True)

snapshots_directory = SNAPSHOTS_DIR
os.makedirs(snapshots_directory, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=snapshots_directory), name="snapshots")

from collections import deque
import math

class RuntimeHardwarePerformanceLogger:
    def __init__(self, log_interval_seconds=1.0):
        self.log_interval_seconds = log_interval_seconds
        self.process = psutil.Process(os.getpid())
        self.is_running = False
        self.monitor_thread = None
        self.latest_latency_ms = 0.0
        self.latest_fps = 25.0
        self.frame_counter = 0
        self.last_log_timestamp = time.time()
        self.last_frame_timestamp = None
        self.smoothed_fps = 25.0
        self.lock = threading.Lock()
        
        # Cửa sổ trượt lưu trữ mẫu FPS trong 60 giây gần nhất: danh sách tuple (timestamp, fps)
        self.sliding_window_1m = deque()
        
        # Đường dẫn tệp nhật ký hiệu năng thời gian thực
        self.log_file_path = os.path.join(LOGS_DIR, "runtime_hardware_performance.log")
        
        # Reset file log mỗi lần khởi chạy máy chủ backend
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            start_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"=== NHẬT KÝ THEO DÕI HIỆU NĂNG THỜI GIAN THỰC (BẮT ĐẦU: {start_str}) ===\n")
            f.write("Format: [Thời gian] FPS | FPS Avg(1m) | Độ lệch FPS (±) | Độ trễ (ms) | CPU (%) | RAM (MB) | GPU VRAM\n")
            f.write("-" * 95 + "\n")
            
        print(f"Đã khởi tạo tệp nhật ký hiệu năng thời gian thực: {self.log_file_path}")

    def update_frame_time(self, frame_duration_ms):
        current_perf_now = time.perf_counter()
        wall_clock_now = time.time()
        with self.lock:
            self.latest_latency_ms = frame_duration_ms
            
            if self.last_frame_timestamp is not None:
                delta_seconds = current_perf_now - self.last_frame_timestamp
                if delta_seconds > 0.001:
                    raw_stream_fps = 1.0 / delta_seconds
                else:
                    raw_stream_fps = 25.0
            else:
                raw_stream_fps = 25.0
            self.last_frame_timestamp = current_perf_now

            # Khống chế FPS tức thời theo tốc độ camera (tối đa 25.0 FPS)
            capped_fps = min(25.0, raw_stream_fps)
            
            # Làm mịn bằng Exponential Moving Average (EMA) để số hiển thị ổn định, không giật lắc
            self.smoothed_fps = (0.80 * self.smoothed_fps) + (0.20 * capped_fps)
            self.latest_fps = self.smoothed_fps
            self.frame_counter += 1
            
            self.sliding_window_1m.append((wall_clock_now, self.latest_fps))
            
            # Loại bỏ các mẫu cũ hơn 60 giây
            cutoff_time = wall_clock_now - 60.0
            while self.sliding_window_1m and self.sliding_window_1m[0][0] < cutoff_time:
                self.sliding_window_1m.popleft()

    def get_current_metrics(self):
        current_now = time.time()
        with self.lock:
            fps_val = self.latest_fps
            latency_val = self.latest_latency_ms
            
            cutoff_time = current_now - 60.0
            while self.sliding_window_1m and self.sliding_window_1m[0][0] < cutoff_time:
                self.sliding_window_1m.popleft()

            window_samples = list(self.sliding_window_1m)

        if len(window_samples) >= 2:
            earliest_timestamp = window_samples[0][0]
            elapsed_time_seconds = current_now - earliest_timestamp
            
            # FPS trung bình chuẩn = Tổng số khung hình chia cho tổng thời gian thực trôi qua (khống chế tối đa 25 FPS)
            if elapsed_time_seconds > 0.5:
                avg_fps_1m = (len(window_samples) - 1) / elapsed_time_seconds
                avg_fps_1m = min(25.0, avg_fps_1m)
            else:
                avg_fps_1m = fps_val

            # Tính độ lệch giữa các khung hình (chỉ xét các giá trị FPS hợp lý trong khoảng 1 - 60 FPS)
            valid_fps_values = [item[1] for item in window_samples if 1.0 <= item[1] <= 60.0]
            if len(valid_fps_values) > 0:
                mean_fps = sum(valid_fps_values) / len(valid_fps_values)
                variance = sum((x - mean_fps) ** 2 for x in valid_fps_values) / len(valid_fps_values)
                fps_std_1m = math.sqrt(variance)
            else:
                fps_std_1m = 0.0
        else:
            avg_fps_1m = fps_val
            fps_std_1m = 0.0

        cpu_percent = self.process.cpu_percent(interval=None)
        process_ram_mb = self.process.memory_info().rss / (1024 ** 2)
        system_ram_percent = psutil.virtual_memory().percent

        cuda_vram_mb = 0.0
        if torch.cuda.is_available():
            # Đo tổng VRAM thực tế đã cấp phát trên card màn hình (bao gồm cả bộ đệm và trọng số mô hình)
            cuda_vram_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)

        return {
            "fps": round(fps_val, 1),
            "avg_fps_1m": round(avg_fps_1m, 1),
            "fps_std_1m": round(fps_std_1m, 1),
            "latency_ms": round(latency_val, 1),
            "cpu_percent": round(cpu_percent, 1),
            "ram_mb": round(process_ram_mb, 1),
            "system_ram_percent": round(system_ram_percent, 1),
            "cuda_vram_mb": round(cuda_vram_mb, 1),
            "has_cuda": torch.cuda.is_available()
        }

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self._logging_loop, daemon=True)
            self.monitor_thread.start()

    def _logging_loop(self):
        print_counter = 0
        while self.is_running:
            time.sleep(self.log_interval_seconds)
            metrics = self.get_current_metrics()
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            gpu_str = f"{metrics['cuda_vram_mb']} MB" if metrics["has_cuda"] else "CPU Mode"
            log_entry = (
                f"[{timestamp_str}] "
                f"FPS: {metrics['fps']:4.1f} | "
                f"FPS Avg(1m): {metrics['avg_fps_1m']:4.1f} | "
                f"Độ lệch: ±{metrics['fps_std_1m']:3.1f} | "
                f"Latency: {metrics['latency_ms']:5.1f} ms | "
                f"CPU: {metrics['cpu_percent']:4.1f}% | "
                f"RAM: {metrics['ram_mb']:6.1f} MB | "
                f"GPU VRAM: {gpu_str}\n"
            )

            # Ghi trực tiếp vào file log 1 giây 1 lần
            try:
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write(log_entry)
            except Exception:
                pass

            # Cứ mỗi 5 giây in 1 dòng tóm tắt ra màn hình terminal
            print_counter += 1
            if print_counter % 5 == 0:
                print(f"[HARDWARE MONITOR] FPS: {metrics['fps']:.1f} | Avg 1m: {metrics['avg_fps_1m']:.1f} (±{metrics['fps_std_1m']:.1f}) | Latency: {metrics['latency_ms']:.1f}ms | CPU: {metrics['cpu_percent']:.1f}% | RAM: {metrics['ram_mb']:.1f}MB | GPU: {gpu_str}")

# Khởi tạo Pipeline Manager duy nhất, Worker đọc Camera và Hardware Logger
pipeline_manager = SafetyPipelineManagerOOP()
camera_worker = RTSPCameraStreamWorker(camera_source=DEFAULT_CAMERA_SOURCE)
camera_worker.start()

hardware_logger = RuntimeHardwarePerformanceLogger(log_interval_seconds=1.0)
hardware_logger.start()

violation_events_history_list = []

class ROIZoneModel(BaseModel):
    zone_id: str
    zone_name: str
    polygon_points_list: List[List[int]]
    allowed_person_ids_list: Optional[List[str]] = []

class AddPersonModel(BaseModel):
    person_id: str

class WorkerProfileModel(BaseModel):
    track_id: str
    full_name: str
    employee_code: Optional[str] = ""

class CameraSourceModel(BaseModel):
    camera_source: str

@app.get("/", response_class=HTMLResponse)
async def get_index_page():
    index_html_path = os.path.join(frontend_directory, "index.html")
    if os.path.exists(index_html_path):
        with open(index_html_path, "r", encoding="utf-8") as file_handle:
            content = file_handle.read()
            return HTMLResponse(content=content)
    else:
        return HTMLResponse(content="<h2>Lỗi: Chưa tìm thấy file frontend/index.html</h2>")

@app.get("/api/status")
async def get_system_status():
    summary_report = pipeline_manager.state_manager.get_summary_report()
    return JSONResponse(content={
        "status": "ONLINE",
        "device": "NVIDIA Jetson Orin Nano",
        "camera_source": camera_worker.camera_source,
        "timestamp": time.time(),
        "summary_report": summary_report
    })

@app.post("/api/camera/source")
async def set_camera_source_endpoint(payload: CameraSourceModel):
    camera_worker.set_camera_source(payload.camera_source)
    return JSONResponse(content={"message": f"Đã cập nhật nguồn Camera RTSP mới: {payload.camera_source}"})

@app.post("/api/camera/reconnect")
async def reconnect_camera_endpoint():
    camera_worker.reconnect()
    return JSONResponse(content={"message": "Đã reset kết nối camera RTSP thành công!"})

@app.get("/api/roi/zones")
async def get_roi_zones():
    zones_list = []
    for zone_id, zone_data in pipeline_manager.roi_manager.roi_zones_dictionary.items():
        zones_list.append({
            "zone_id": zone_id,
            "zone_name": zone_data["zone_name"],
            "polygon_points_list": zone_data["raw_points"],
            "allowed_person_ids_list": zone_data["allowed_person_ids_list"]
        })
    return JSONResponse(content={"zones": zones_list})

@app.post("/api/roi/zones")
async def create_or_update_roi_zone(zone_payload: ROIZoneModel):
    pipeline_manager.roi_manager.add_zone(
        zone_id=zone_payload.zone_id,
        zone_name=zone_payload.zone_name,
        polygon_points_list=zone_payload.polygon_points_list,
        allowed_person_ids_list=zone_payload.allowed_person_ids_list
    )
    return JSONResponse(content={"message": f"Đã tạo {zone_payload.zone_name} thành công!"})

@app.delete("/api/roi/zones/{zone_id}")
async def delete_roi_zone(zone_id: str):
    removed_zone = pipeline_manager.roi_manager.remove_zone(zone_id)
    if removed_zone is not None:
        return JSONResponse(content={"message": f"Đã xóa [{zone_id}]"})
    else:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy [{zone_id}] để xóa")

@app.post("/api/roi/zones/{zone_id}/add_person")
async def add_person_to_zone(zone_id: str, payload: AddPersonModel):
    success = pipeline_manager.roi_manager.add_allowed_person_to_zone(zone_id, payload.person_id)
    if success:
        return JSONResponse(content={"message": f"Đã thêm Người {payload.person_id} vào {zone_id}"})
    else:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy {zone_id}")

@app.delete("/api/roi/zones/{zone_id}/remove_person/{person_id}")
async def remove_person_from_zone(zone_id: str, person_id: str):
    success = pipeline_manager.roi_manager.remove_allowed_person_from_zone(zone_id, person_id)
    if success:
        return JSONResponse(content={"message": f"Đã xóa Người {person_id} khỏi {zone_id}"})
    else:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy {zone_id}")

@app.get("/api/worker/profiles")
async def get_worker_profiles():
    profiles_dict = pipeline_manager.roi_manager.worker_profiles_dictionary
    return JSONResponse(content={"worker_profiles": profiles_dict})

@app.post("/api/worker/profile")
async def set_worker_profile(payload: WorkerProfileModel):
    pipeline_manager.roi_manager.set_person_profile(
        track_id=payload.track_id,
        full_name=payload.full_name,
        employee_code=payload.employee_code
    )
    display_name = pipeline_manager.roi_manager.get_person_display_name(payload.track_id)
    return JSONResponse(content={"message": f"Đã gán hồ sơ thành công: {display_name}"})

@app.get("/api/violations/history")
async def get_violations_history():
    return JSONResponse(content={"violations_history": violation_events_history_list})

@app.get("/api/violations/history_db")
async def get_violations_history_from_db(
    track_id: Optional[int] = None,
    violation_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    db_path = pipeline_manager.state_manager.db_path
    if not os.path.exists(db_path):
        return JSONResponse(content={"total": 0, "records": []})

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query_conditions = []
    query_parameters = []

    if track_id is not None:
        query_conditions.append("track_id = ?")
        query_parameters.append(track_id)
    if violation_type is not None and violation_type != "":
        query_conditions.append("violation_type = ?")
        query_parameters.append(violation_type)
    if status is not None and status != "":
        query_conditions.append("status = ?")
        query_parameters.append(status)

    where_clause = ""
    if len(query_conditions) > 0:
        where_clause = "WHERE " + " AND ".join(query_conditions)

    count_query = f"SELECT COUNT(*) FROM violations {where_clause}"
    cursor.execute(count_query, query_parameters)
    total_count = cursor.fetchone()[0]

    select_query = f'''
        SELECT 
            id, track_id, violation_type, start_timestamp, start_time_string,
            end_timestamp, end_time_string, duration_seconds, status,
            full_snapshot_path, crop_snapshot_path
        FROM violations
        {where_clause}
        ORDER BY id DESC
        LIMIT ? OFFSET ?
    '''
    cursor.execute(select_query, query_parameters + [limit, offset])
    rows = cursor.fetchall()
    conn.close()

    records_list = []
    for row in rows:
        records_list.append({
            "id": row[0],
            "track_id": row[1],
            "violation_type": row[2],
            "start_timestamp": row[3],
            "start_time_string": row[4],
            "end_timestamp": row[5],
            "end_time_string": row[6],
            "duration_seconds": row[7],
            "status": row[8],
            "full_snapshot_path": row[9],
            "crop_snapshot_path": row[10]
        })

    return JSONResponse(content={
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "records": records_list
    })

@app.post("/api/system/reset_logs")
async def reset_system_logs_endpoint():
    global violation_events_history_list
    violation_events_history_list.clear()
    pipeline_manager.state_manager.clear_all_logs_in_db()
    pipeline_manager.reset()
    return JSONResponse(content={"message": "Đã xóa toàn bộ nhật ký vi phạm (kể cả Database) và reset trạng thái hệ thống từ đầu!"})

@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSockets Client connected!")
    
    test_image_path = os.path.join(project_root, "test", "test_weight", "anh1.jpg")
    fallback_frame = cv2.imread(test_image_path) if os.path.exists(test_image_path) else None
    last_processed_frame_id = -1

    try:
        while True:
            frame_from_camera, current_frame_id = camera_worker.get_latest_frame()
            if frame_from_camera is not None:
                if current_frame_id == last_processed_frame_id:
                    # Chờ khung hình mới từ camera RTSP (tránh chạy suy luận lặp lại trên cùng 1 frame)
                    await asyncio.sleep(0.008)
                    continue
                last_processed_frame_id = current_frame_id
                current_frame = frame_from_camera
            else:
                if fallback_frame is not None:
                    current_frame = fallback_frame.copy()
                    # Khống chế tốc độ 25 FPS khi chạy ảnh tĩnh fallback
                    await asyncio.sleep(0.038)
                else:
                    current_frame = None

            if current_frame is not None:
                frame_start_time = time.perf_counter()

                # Chạy Pipeline xử lý ma trận và suy luận AI trong ThreadPool riêng để không block Event Loop của FastAPI
                annotated_frame, report_dictionary, new_events_list = await asyncio.to_thread(
                    pipeline_manager.process_frame, 
                    current_frame
                )
                
                frame_end_time = time.perf_counter()
                frame_duration_ms = (frame_end_time - frame_start_time) * 1000.0
                hardware_logger.update_frame_time(frame_duration_ms)

                for event in new_events_list:
                    violation_events_history_list.append(event)
                
                if annotated_frame.shape[1] > 960:
                    stream_frame = cv2.resize(annotated_frame, (960, 540), interpolation=cv2.INTER_LINEAR)
                else:
                    stream_frame = annotated_frame

                encode_success, buffer = cv2.imencode(".jpg", stream_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if encode_success:
                    base64_image = base64.b64encode(buffer).decode("utf-8")
                else:
                    base64_image = ""

                payload = {
                    "timestamp": time.time(),
                    "image_base64": base64_image,
                    "report_dictionary": report_dictionary,
                    "new_events_list": new_events_list,
                    "hardware_metrics": hardware_logger.get_current_metrics()
                }
                await websocket.send_json(payload)
            else:
                empty_payload = {
                    "timestamp": time.time(),
                    "image_base64": "",
                    "report_dictionary": {"error": "Khung hình rỗng"},
                    "new_events_list": []
                }
                await websocket.send_json(empty_payload)

            await asyncio.sleep(0.005)
    except WebSocketDisconnect:
        print("WebSockets Client disconnected.")
    except Exception as exception_error:
        print(f"Lỗi WebSocket Stream: {str(exception_error)}")
