import sys
import os
import uvicorn
import requests
import time
import multiprocessing

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_fastapi_server():
    from services.api_backend import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

def test_web_service_endpoints():
    print("=== BẮT ĐẦU KIỂM THỬ DỊCH VỤ WEB BACKEND FASTAPI & FRONTEND ===")
    
    server_process = multiprocessing.Process(target=run_fastapi_server)
    server_process.start()
    
    print("Đang chờ máy chủ FastAPI khởi động...")
    server_ready = False
    for attempt in range(20):
        time.sleep(1)
        try:
            check_response = requests.get("http://127.0.0.1:8000/api/status", timeout=1)
            if check_response.status_code == 200:
                server_ready = True
                break
            else:
                continue
        except Exception:
            continue
    
    try:
        # Kiểm thử Trang chủ Frontend GET /
        index_response = requests.get("http://127.0.0.1:8000/")
        print(f"Kiểm thử GET / (Frontend Page): Status Code = {index_response.status_code}")
        if index_response.status_code == 200:
            print("KIỂM THỬ CHÍNH XÁC: Đã tải được trang Frontend index.html!")
        else:
            print(f"KIỂM THỬ LỖI: Status code = {index_response.status_code}")

        # Kiểm thử API Trạng thái GET /api/status
        status_response = requests.get("http://127.0.0.1:8000/api/status")
        print(f"Kiểm thử GET /api/status: Status Code = {status_response.status_code}")
        print(f"Dữ liệu phản hồi: {status_response.json()}")

        # Kiểm thử API Vùng ROI GET /api/roi/zones
        zones_response = requests.get("http://127.0.0.1:8000/api/roi/zones")
        print(f"Kiểm thử GET /api/roi/zones: Status Code = {zones_response.status_code}")
        print(f"Danh sách vùng ROI: {zones_response.json()}")

    finally:
        print("Đang dừng máy chủ FastAPI...")
        server_process.terminate()
        server_process.join()
        print("Đã kết thúc kiểm thử dịch vụ Web thành công!")

if __name__ == "__main__":
    test_web_service_endpoints()
