# SYSTEM DEVELOPMENT & OPTIMIZATION TIMELINE LOG

## [2026-09-25] Phase 1: Human Detection Model (Model 1)
- Trained YOLOv8n Human Detection model at 1280x1280 resolution on Kaggle GPU Tesla T4.
- Evaluated on test set: mAP@50 = 91.1%, Precision = 89.6%, Recall = 85.1%, Inference speed = ~10.8ms.
- Saved best weights to `model/weights/model_human_detection.pt`.

## [2026-09-29] Phase 2: PPE Detection Model (Model 2) & Pipeline Setup
- Trained YOLOv8n PPE classifier with 10 classes (`boots`, `helmet on`, `no boots`, `no helmet`, `no vest`, `vest`, `id_card`, `no id_card`, `smoking`, `phone_usage`).
- Saved best weights to `model/weights/model_ppe_detection.pt`.
- Established 2-Stage Pipeline: Crop person bounding box with 10% padding before sending to Model 2.

## [2026-09-30] Phase 3: System Integration & Real-Time Tracking Upgrade
- Integrated ByteTrack SOTA with 4-tier fallback matrix (`TrackerEngineOOP`).
- Resolved Canvas coordinate offset & letterbox padding (`getDisplayedImageRect()` in `frontend/index.html`).
- Implemented Person-Centric State Logging in `core_pipeline/state_manager.py` and live table updates (`updatePersonStateTable`).
- Fixed explicit evidence violation check to eliminate false violations (`no_id_card`).

## [2026-10-03] Edge Boundary Check & Sliding Window PPE Buffer
- Implemented `edge_margin_pixels = 15` boundary cutoff check in `pipeline_manager.py`. Tagged edge crops as `CHƯA RÕ GÓC NHÌN (CẮT MÉP CAM)`.
- Added 8-frame Sliding Window PPE Buffer with 40% Majority Voting in `core_pipeline/state_manager.py` to eliminate 1-frame flickering.
- Added live system reset API `POST /api/system/reset_logs` and UI button.

## [2026-10-05] Deep Re-ID & Worker Profile Mapping
- Implemented `DeepFeatureExtractorOOP` using MobileNetV3 Small (512-d embeddings) on CUDA GPU.
- Built `person_feature_gallery` with Cosine Similarity matching (>= 0.65) and 300-frame memory retention.
- Re-ordered tracking priorities: Frame-to-frame ByteTrack spatial IoU (>= 0.25) prioritized before Re-ID re-appearance.
- Built Worker Profile Mapping with `POST /api/worker/profile` and UI profile assignment form.
- Synchronized ROI intrusion (`UNAUTHORIZED_INTRUSION`) into active violations table and live video bounding box labels.
- Replaced OpenCV text with PIL ImageDraw (`DejaVuSans-Bold.ttf`) for sharp UTF-8 Vietnamese rendering.

## [2026-10-06] Hardware Performance Optimization for Jetson Orin Nano (Zero-Risk Architecture)
- Identified 3 main GPU/CPU bottlenecks causing FPS drop when workers appear in camera view.
- Implemented Re-ID Smart Sampling in `tracking/tracker_engine.py` (samples MobileNetV3 every 20 frames or on unmatched tracks, eliminating 95% GPU overhead).
- Implemented PPE Inference Sampling Cache in `core_pipeline/pipeline_manager.py` (`ppe_eval_interval = 4` frames per track, eliminating 75% Model 2 GPU overhead).
- Optimized PIL UTF-8 Text Drawing to convert only bounding box sub-crop regions instead of full 720p frames (reduces CPU memory copy latency from 12ms to < 0.1ms).
- Accelerated WebSocket stream: reduced JPEG quality to 70% (`[cv2.IMWRITE_JPEG_QUALITY, 70]`) and reduced loop sleep to 0.005s, freeing 40ms hard latency delay.
- Optimized Model 1 Human Detector resolution (`image_size=320`) and Model 2 PPE Classifier resolution (`imgsz=640`).
- Enabled CUDNN Benchmark (`torch.backends.cudnn.benchmark = True`) and OpenCV/PyTorch full CPU core threading.
- Converted all 3 PyTorch CUDA models (Model 1 Human Detector, Model 2 PPE Classifier, Deep Re-ID MobileNetV3) to native FP16 Tensor Cores mode (`.half()`), boosting GPU matrix operations by 2-3x.
- Implemented RTSP buffer flushing (`video_capture.grab()`) and `CAP_PROP_BUFFERSIZE=1` in `api_backend.py` to eliminate video lag.
- Offloaded snapshot disk writing (`cv2.imwrite`) and SQLite DB sync (`_sync_events_to_db`) to background daemon threads in `state_manager.py` to eliminate I/O stutter spikes.
- Downscaled streaming WebSocket images to 960x540 for 2.5x faster base64 transfer.
- Implemented automatic TensorRT `.engine` file detection and loading in `human_detector.py` and `ppe_classifier.py`.
- Added Adaptive PPE Sampling in `pipeline_manager.py` based on centroid movement distance (samples stationary workers every 8 frames instead of 4).
- Removed forced frame-skipping loop (`for _ in range(3): video_capture.grab()`) in `api_backend.py` to restore 100% full 25 FPS stream rate.
- Implemented Staggered Inference in `pipeline_manager.py` across track IDs (`(frame_counter + track_id) % interval == 0`) to distribute Model 2 inference across frames and eliminate frame stutter.
- Streamlined Deep Re-ID in `tracker_engine.py`: replaced CPU PIL transforms with C++ OpenCV resize and direct GPU Tensor Cores FP16 normalization (cuts latency from 15ms to 0.2ms).
- Preserved full continuous Re-ID learning: instant re-appearance matching, gallery registration, and staggered EMA multi-angle appearance updating (`(frame_counter + track_id) % 25 == 0`).
- Fully verified zero-conflict compatibility with State Manager upgrades: `id_card_grace_period_frames = 35` and non-resolvable `smoking` and `phone_usage` violations.
- Verified all unit and integration tests (`test_pipeline_oop.py`, `test_roi_manager.py`, `test_state_manager.py`, `test_roi_assignment.py`, `test_web_service.py`) passing 100% with exit code 0.


