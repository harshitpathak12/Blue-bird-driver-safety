# Driver Safety System

Real-time driver monitoring backend: face recognition, fatigue and distraction detection, alerts, and daily safety scoring.

## Features

- **Face authentication** – Login by face image; register with live photo (base64); unique 9-character `driver_id`
- **Real-time video stream** – WebSocket accepts camera frames, runs ML models, returns annotated frames with HUD overlay
- **In-stream face recognition** – Identifies driver from video when `driver_id` is not provided
- **Multi-model distraction pipeline** – Calibration-based attention model, head pose (solvePnP), eye gaze (iris tracking), LSTM temporal model, online CNN, rule-based voting
- **Fatigue detection** – EAR/MAR with 2-second confirmation, PERCLOS, blink rate
- **Fusion engine** – Combines fatigue + distraction + drowsiness into unified state: `normal` | `fatigue` | `distraction` | `sleep`
- **MongoDB persistence** – Alerts, sessions, drivers, daily safety scores
- **REST API** – Login, register, sessions, monitor frame, alerts, safety score

## Tech Stack

- **Python 3.12+**
- **FastAPI** – REST + WebSocket
- **MongoDB** – Alerts, sessions, drivers, daily scores
- **OpenCV, MediaPipe** – Face detection and 478 landmarks
- **PyTorch** – LSTM temporal attention + online CNN (MobileNet-V3)
- **ONNX Runtime** – ArcFace face recognition
- **RetinaFace** – Face detection for recognition pipeline

## Prerequisites

- Python 3.12 or higher
- MongoDB (local or [MongoDB Atlas](https://www.mongodb.com/atlas))
- Model files in `models/` (see [Configuration](#configuration))

## Installation

```bash
cd driver_safety_system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Edit **`configs/config.yaml`**:

```yaml
mongodb:
  url: "mongodb+srv://USER:PASSWORD@CLUSTER.mongodb.net/"
  database: "driver_monitoring"
```

Optional: set **`MODEL_BASE_DIR`** environment variable to override the default `models/` directory. Expected files:

- `face_landmarker.task` (MediaPipe)
- `temporal_attention_lstm.pth` (LSTM distraction model)
- `eye_gaze.pkl`, `eye_scaler.pkl` (eye gaze classifier)
- `arcface.onnx`, `arcface_db.npz` (face recognition)
- `headpose_classifier.pkl` (legacy head pose — optional)

The server starts even if some model files are missing; affected features are disabled.

## Running

**On the VM (server only — no camera, no login UI, no calibration):**

```bash
# From project root
uvicorn app.api.main:app --host 0.0.0.0 --port 5000 --reload
# or: python -m app.api.main
```

- **API:** `http://localhost:5000`
- **Docs:** `http://localhost:5000/docs`
- **WebSocket:** `ws://localhost:5000/api/stream`
- **Recalibrate:** `GET /api/recalibrate`

## Client (local machine → VM)

The client performs face login (or **registration** if not in DB), frame-only confirmation, terminal calibration prompt, then the stream.

```bash
python -m data_pipeline.client --base-url http://<VM_HOST>:5000 --source 0
```

1. `POST /api/login/` — match face to stored embeddings (DB / cosine similarity).  
2. If not registered: terminal **name** and **age** → live webcam → `POST /api/login/register` (server: MediaPipe landmarks → **3D embedding** + MongoDB; same logic as `login.py`).  
3. **Login Completed** or **Registration Completed** on frame only (no dashboard).  
4. Terminal: **Do you want to Start Callibration?** — only **y** / **Y** continues.  
5. `GET /api/recalibrate`, then WebSocket stream (frame + dashboard from server).

Skip login/register: `python -m data_pipeline.client --stream-only --url ws://<VM_HOST>:5000/api/stream --source 0`

Press `x` to quit during stream.

## Project Structure

```
driver_safety_system/
├── app/
│   └── api/                # REST and WebSocket routes
│       ├── main.py         # FastAPI app factory + entry point
│       ├── login.py        # Face login & register
│       ├── sessions.py     # Session start/end
│       ├── monitor.py      # Single-frame ingest
│       ├── alerts.py       # Alerts CRUD
│       └── safety_score.py # Daily safety scores
├── data_pipeline/
│   └── client.py           # Local: login → calibration prompt → WebSocket (VM has no UI)
├── src/                    # Core services
│   └── pipelines/         # Real-time streaming pipelines
│       └── realtime_frame_pipeline.py
│   ├── driver_identity.py  # DriverIdentityService
│   ├── face_embedding_open3d.py # Open3D PointCloud → 3D embedding
│   ├── fusion/             # FusionEngine (fatigue/distraction/sleep)
│   ├── scoring/            # SafetyScoring
│   └── schemas/            # Pydantic request/response models
├── training/
│   └── scripts/            # Model loading + inference
│       ├── face_detection/         # FaceDetector (MediaPipe)
│       ├── fatigue_detection/      # ModelFatigue (EAR/MAR)
│       ├── blink_perclos/          # ModelDrowsiness + ModelEyeGaze
│       ├── distraction_detection/  # DistractionDetector (voting system)
│       │   ├── distraction_detector.py   # Main wrapper
│       │   ├── attention_model.py        # Calibration-based attention
│       │   ├── head_pose_estimator.py    # solvePnP head pose
│       │   ├── eye_gaze_estimator.py     # Iris gaze estimation
│       │   ├── geometric_temporal_classifier.py # Rule-based classifier
│       │   ├── temporal_attention_model.py     # LSTM temporal model
│       │   ├── online_cnn.py             # Online MobileNet-V3 CNN
│       │   └── face_3d_features.py       # 3D face features
│       └── face_recongnition/      # ArcFaceModel
├── database/               # MongoDB repositories (class-based)
├── models/                 # Trained model weight files
├── configs/                # config.yaml + ConfigLoader
├── utils/                  # OverlayRenderer (HUD drawing)
├── requirements.txt
└── README.md
```

## License

See the repository or organization for license details.
