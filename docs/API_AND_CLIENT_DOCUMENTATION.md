# Driver Safety System – API & Client Documentation

**Version:** 1.0.0  
**Last Updated:** March 26, 2026  
**Audience:** Frontend Team, Client Stakeholders

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Base URL & Environment](#2-base-url--environment)
3. [Authentication & Driver Identity](#3-authentication--driver-identity)
4. [REST API Reference](#4-rest-api-reference)
5. [WebSocket Real-Time Stream](#5-websocket-real-time-stream)
   - [WebSocket Driving Licence Verification](#51-websocket-driving-licence-verification-registration-step-2)
6. [Data Models & Schemas](#6-data-models--schemas)
7. [Integration Guidelines for Frontend](#7-integration-guidelines-for-frontend)
8. [Error Handling](#8-error-handling)
9. [CORS & Security](#9-cors--security)

---

## 1. Project Overview

The **Driver Safety System** is a real-time monitoring backend that:

- Detects driver fatigue, distraction, and drowsiness using computer vision
- Recognizes drivers via face authentication (ArcFace)
- Stores alerts and sessions in MongoDB
- Computes daily safety scores for fleet management

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **Face Login** | Driver logs in by uploading a face image; system matches against registered drivers |
| **Face Registration** | New drivers register with name, optional age, and **live photo from camera** (base64). Registration is **two-step**: first returns a `pending_id`, then driving-licence verification finalizes and creates the driver record. |
| **Real-Time Stream** | WebSocket accepts video frames, runs ML models (fatigue, head pose, eye gaze), returns annotated frame with HUD |
| **Face Recognition in Stream** | When `driver_id` is not passed in the WebSocket URL, system auto-identifies the driver from the video |
| **Sessions** | Start/end driving sessions; alerts are linked to sessions |
| **Alerts** | Fatigue, distraction, sleep events stored with `driver_id`, `session_id`, timestamp, GPS (optional) |
| **Safety Score** | Daily score (0–100) derived from fatigue, distraction, sleep counts; risk levels: Safe / Moderate Risk / High Risk |

---

## 2. Base URL & Environment

| Environment | Base URL |
|-------------|----------|
| Local development | `http://localhost:5000` |
| Production | Configure per deployment (e.g. `https://api.yourdomain.com`) |

### Run the Server

```bash
cd driver_safety_system
python -m app.api.main
# (equivalent) uvicorn app.api.main:app --host 0.0.0.0 --port 5000
```

- **Health check:** `GET /health` → `{"status": "ok"}`
- **API docs (Swagger):** `GET /docs`
- **ReDoc:** `GET /redoc` (if enabled)

---

## 3. Authentication & Driver Identity

- There is **no JWT/OAuth**; driver identity is established via **face recognition**.
- **`driver_id`** is a unique string ID assigned when a driver is **finalized** (after driving-licence verification).
- **Two-step onboarding** (Frontend should follow this flow):
  - **Step 1:** `POST /api/login/register` → returns **`pending_id`** (no DB driver row yet)
  - **Step 2:** `WS /api/login/ws/dl-verify?pending_id=...&qwen=1` → server validates licence + verifies name/age → creates driver in DB → returns `registration.status=completed` with `driver_id`
- Face embeddings are stored in MongoDB; login/stream use them to match drivers (3D MediaPipe embedding; optional 2D ArcFace when available).

---

## 4. REST API Reference

### 4.1 Login – Face Recognition

**`POST /api/login/`**

Log in using a face image. If `driver_id` is provided, verifies against that driver; otherwise matches against all registered drivers.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | File (multipart) | Yes | Face image (JPEG/PNG) |
| `driver_id` | Form | No | If provided, verify face against this driver only |

**Request:** `multipart/form-data` with `image` (and optionally `driver_id`)

**Response (200):**

```json
{
  "driver_id": "12345",
  "driver_name": "John Doe",
  "age": 35,
  "message": "Login successful"
}
```

**Errors:**

- `400` – Empty image or no face detected
- `401` – Face not recognized or invalid driver_id
- `500` – Unexpected server error (check server logs; should be rare)

---

### 4.2 Register – New Driver (live photo only)

**`POST /api/login/register`**

Register a new driver using a **live photo from the camera** (no file upload). This is **strict registration**:

- The server stores the face embedding + name/age as an **in-memory pending record**
- The response returns a **`pending_id`**
- The driver record is created **only after** driving-licence verification succeeds (see DL WebSocket below)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Driver name |
| `age` | number | No | Driver age (1–120) |
| `image_base64` | string | Yes | Base64-encoded JPEG/PNG from live camera (e.g. canvas.toDataURL('image/jpeg').split(',')[1]) |

**Request:** `application/json` body:

```json
{
  "name": "Jane Smith",
  "age": 28,
  "image_base64": "<base64 string from camera capture>"
}
```

**Response (200):**

```json
{
  "pending_id": "A1B2C3D4E5",
  "driver_id": null,
  "driver_name": "Jane Smith",
  "age": 28,
  "message": "Registration successful (waiting for driving licence verification)"
}
```

**Errors:**

- `400` – Empty image or no face detected

---

### 4.3 Driving Licence Finalize (REST, optional)

**`POST /api/login/finalize-dl`**

This is used only when you run an **external** licence detector service (legacy setup).
In the default setup, DL verification and finalize happen inside the DL WebSocket (`/api/login/ws/dl-verify`).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pending_id` | string | Yes | From `POST /api/login/register` |
| `verdict` | string | Yes | Must be `"valid"` |
| `dl_number` | string | Yes | Extracted DL number |
| `ocr_text` | string | Yes | Full OCR text used for name/age matching |
| `validity_end` | string | No | Optional ISO date string |

**Response (200):**

```json
{
  "driver_id": "DRIVER123",
  "dl_number": "MH12 20190001234",
  "validity_end": "2035-04-01",
  "message": "Driving licence verified"
}
```

---

### 4.3 Sessions

**`POST /api/sessions/start`**

Start a driving session.

**Body:**

```json
{
  "driver_id": "12345"
}
```

**Response (200):**

```json
{
  "session_id": "abc123...",
  "driver_id": "12345",
  "start_time": "2025-03-05T10:00:00Z",
  "status": "active",
  "message": "Session started"
}
```

---

**`POST /api/sessions/end`**

End a driving session.

**Body:**

```json
{
  "session_id": "abc123..."
}
```

**Response (200):**

```json
{
  "session_id": "abc123...",
  "end_time": "2025-03-05T12:00:00Z",
  "status": "ended",
  "message": "Session ended"
}
```

**Errors:**

- `404` – Session not found

---

### 4.4 Monitor – Single Frame (REST)

**`POST /api/monitor/frame`**

Ingest a single frame with metadata. Useful for testing; for live streaming, prefer WebSocket.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `driver_id` | Form | Yes | Driver ID |
| `session_id` | Form | No | Session ID |
| `gps_latitude` | Form | No | Latitude |
| `gps_longitude` | Form | No | Longitude |
| `frame` | File (multipart) | Yes | Image frame (JPEG/PNG) |

**Response (200):**

```json
{
  "processed": true,
  "alert": null,
  "driver_state": "normal"
}
```

If an alert is raised:

```json
{
  "processed": true,
  "alert": {
    "driver_id": "12345",
    "session_id": "abc123",
    "alert_type": "fatigue",
    "confidence_score": 0.85,
    "timestamp": "2025-03-05T10:30:00Z",
    "gps": { "latitude": 40.7, "longitude": -74.0 }
  },
  "driver_state": "fatigue"
}
```

---

### 4.5 Alerts

**`POST /api/alerts/`**

Manually create an alert (e.g. from external systems).

**Body:**

```json
{
  "driver_id": "12345",
  "session_id": "abc123",
  "alert_type": "fatigue",
  "confidence_score": 0.8,
  "gps": { "latitude": 40.7, "longitude": -74.0 }
}
```

**`alert_type`:** `"fatigue"` | `"distraction"` | `"sleep"`  
**`confidence_score`:** 0.0–1.0

---

**`GET /api/alerts/`**

List alerts with optional filters.

| Query Param | Type | Default | Description |
|-------------|------|---------|-------------|
| `driver_id` | string | - | Filter by driver |
| `session_id` | string | - | Filter by session |
| `limit` | int | 100 | Max alerts to return (1–500) |

**Response (200):**

```json
{
  "alerts": [
    {
      "alert_id": "...",
      "driver_id": "12345",
      "session_id": "abc123",
      "alert_type": "fatigue",
      "confidence_score": 0.85,
      "timestamp": "2025-03-05T10:30:00Z",
      "gps": { "latitude": 40.7, "longitude": -74.0 }
    }
  ],
  "total": 1
}
```

---

### 4.6 Safety Score

**`GET /api/safety-score/`**

Get daily safety scores for a driver.

| Query Param | Type | Required | Description |
|-------------|------|----------|-------------|
| `driver_id` | string | Yes | Driver ID |
| `date_from` | string | No | Start date (YYYY-MM-DD) |
| `date_to` | string | No | End date (YYYY-MM-DD) |

**Response (200):**

```json
{
  "driver_id": "12345",
  "daily_scores": [
    {
      "driver_id": "12345",
      "date": "2025-03-05",
      "fatigue_count": 2,
      "distraction_count": 1,
      "sleep_count": 0,
      "safety_score": 87.0,
      "risk_level": "Moderate Risk"
    }
  ]
}
```

**Risk levels:**

- **90–100** → Safe  
- **70–89** → Moderate Risk  
- **&lt;70** → High Risk  

**Formula:** `score = 100 - (fatigue × 5) - (distraction × 3) - (sleep × 10)`

---

**`POST /api/safety-score/compute`**

Compute and store daily score for a specific date.

| Query Param | Type | Description |
|-------------|------|-------------|
| `driver_id` | string | Driver ID |
| `score_date` | string | Date (YYYY-MM-DD) |

**Response (200):**

```json
{
  "driver_id": "12345",
  "date": "2025-03-05",
  "fatigue_count": 2,
  "distraction_count": 1,
  "sleep_count": 0,
  "safety_score": 87.0,
  "risk_level": "Moderate Risk"
}
```

---

## 5. WebSocket Real-Time Stream

**Endpoint:** `WS /api/stream`

**URL:** `ws://<host>:5000/api/stream` or `wss://<host>/api/stream`

### Query Parameters

| Param | Required | Description |
|-------|----------|-------------|
| `driver_id` | No | If provided, use this driver for events/sessions/alerts. If omitted, face recognition identifies the driver from the video. |

**Examples:**

- `ws://localhost:5000/api/stream` – Auto-identify driver from video
- `ws://localhost:5000/api/stream?driver_id=12345` – Use known driver

### Protocol

1. **Client** → **Server:** Send raw **JPEG bytes** (single frame) per message.
2. **Server** runs:
   - Face detection (landmarks)
   - Fatigue (EAR/MAR)
   - Drowsiness (PERCLOS, blinks)
   - Head pose (distraction)
   - Eye gaze
   - Face recognition (every 10 frames when `driver_id` not in URL)
3. **Server** → **Client:** Send back **JPEG bytes** of the annotated frame (with HUD overlay).

### Frame Overlay (HUD)

Each returned frame includes a left-side panel with:

- **Driver** – Recognized driver (e.g. `"John (12345)"`) or `"Unknown"`
- **EAR** – Eye Aspect Ratio (fatigue)
- **MAR** – Mouth Aspect Ratio (fatigue)
- **PERCLOS** – % eye closure over time
- **Blinks** – Blink count
- **Head** – Head pose (e.g. Forward, Left, Right)
- **Gaze** – Eye gaze (e.g. CENTER)
- **State** – `normal` | `fatigue` | `distraction` | `sleep`

When an alert is raised, a banner appears at the bottom (e.g. `FATIGUE`, `DISTRACTION`, `SLEEP`).

### Frontend Integration Example

```javascript
const ws = new WebSocket('ws://localhost:5000/api/stream?driver_id=12345');

ws.binaryType = 'arraybuffer';

ws.onopen = () => {
  // Capture webcam frame as JPEG and send
  const blob = await captureFrameAsJpeg(); // your capture logic
  ws.send(blob);
};

ws.onmessage = (event) => {
  const blob = event.data; // JPEG bytes
  const img = document.getElementById('preview');
  img.src = URL.createObjectURL(new Blob([blob]));
  // Optionally capture next frame and send (e.g. requestAnimationFrame loop)
};
```

**Recommendation:** Capture at ~10–15 FPS to balance latency and server load.

---

## 5.1 WebSocket Driving Licence Verification (Registration Step 2)

**Endpoint:** `WS /api/login/ws/dl-verify`

**URL (default, server-side DL pipeline):**

- `ws://<host>:5000/api/login/ws/dl-verify?pending_id=<PENDING_ID>&qwen=1`

### Purpose

This WebSocket finalizes registration:

- Client streams camera frames of the driving licence (JPEG)
- Server runs **YOLO detection every frame** + **Qwen OCR in background (throttled)**
- Server validates Indian DL rules + matches **name/age** against the pending registration
- On success, server creates the driver in MongoDB and closes the socket

### Query Parameters

| Param | Required | Description |
|-------|----------|-------------|
| `pending_id` | Yes | From `POST /api/login/register` |
| `qwen` | Yes | Must be `1` (enables JSON payloads; kept for protocol compatibility) |

### Protocol (per frame)

1. **Client → Server:** Send **binary JPEG bytes** (one frame)
2. **Server → Client:** Send **binary JPEG bytes** (annotated)
3. **Server → Client:** Send **JSON text** with detections and (optionally) registration status

JSON shape:

```json
{
  "detections": [
    {
      "bbox": [0, 0, 0, 0],
      "confidence": 0.9,
      "class": "driving_license",
      "ocr_text": "....",
      "ocr_lines": [],
      "validation_label": "processing",
      "validation_confidence": 0.0,
      "validation_reason": "OCR in progress…",
      "dl_numbers": [],
      "validity_end": null
    }
  ],
  "registration": {
    "status": "completed",
    "driver_id": "DRIVER123",
    "dl_number": "MH12 20190001234"
  }
}
```

### `registration.status` values

- **`completed`**: verification OK; server saved driver; socket closes
- **`rejected`**: DL invalid; socket closes
- **`error`**: name/age mismatch or finalize failure; socket closes
- **`waiting`**: only if `DL_VERIFY_MIN_SESSION_SEC > 0` on server; keep streaming until timer elapses
- **`ignored`**: first `invalid` verdict is skipped (anti-flicker); keep streaming

### Frontend guidance

- Keep sending frames until you see `registration.status === "completed"` (or `rejected`/`error`).
- Treat `validation_label` as UI feedback; the actual completion signal is `registration.status`.
- Suggested send rate: **~2 FPS** (licence OCR is expensive; higher FPS does not help accuracy).

## 6. Data Models & Schemas

### Driver

| Field | Type | Description |
|-------|------|-------------|
| driver_id | string | Unique 5-digit ID |
| name | string | Driver name |
| age | int? | Optional age |
| face_embedding | float[] | Stored for recognition |
| face_image_path | string? | Optional path to stored image |
| created_at | datetime | Registration time |
| last_seen | datetime? | Last login/activity |

### Session

| Field | Type | Description |
|-------|------|-------------|
| session_id | string | Unique ID |
| driver_id | string | Driver ID |
| start_time | datetime | Session start |
| end_time | datetime? | Session end (null if active) |

### Alert

| Field | Type | Description |
|-------|------|-------------|
| alert_id | string | Mongo ObjectId |
| driver_id | string | Driver ID |
| session_id | string? | Session ID |
| alert_type | string | fatigue \| distraction \| sleep |
| confidence_score | float | 0.0–1.0 |
| timestamp | datetime | When alert occurred |
| gps | object? | { latitude, longitude } |

### Daily Score

| Field | Type | Description |
|-------|------|-------------|
| driver_id | string | Driver ID |
| date | string | YYYY-MM-DD |
| fatigue_count | int | Fatigue alerts that day |
| distraction_count | int | Distraction alerts |
| sleep_count | int | Sleep alerts |
| safety_score | float | 0–100 |
| risk_level | string | Safe \| Moderate Risk \| High Risk |

---

## 7. Integration Guidelines for Frontend

### Recommended Flow

1. **Login:** `POST /api/login/` with face image → get `driver_id`, `driver_name`, `age`.
2. **Start session:** `POST /api/sessions/start` with `driver_id`.
3. **Open WebSocket:** `ws://host/api/stream?driver_id=<driver_id>`.
4. **Send frames:** Capture camera, encode as JPEG, send bytes.
5. **Display annotated frame:** Receive JPEG, render in `<img>` or `<video>` element.
6. **End session:** `POST /api/sessions/end` with `session_id` when trip ends.
7. **Dashboard:** `GET /api/alerts/?driver_id=...` and `GET /api/safety-score/?driver_id=...`.

### Frame Capture (Browser)

```javascript
// Example: capture from <video> (webcam stream)
async function captureFrameAsJpeg(videoElement, quality = 0.8) {
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoElement, 0, 0);
  return await new Promise(resolve => {
    canvas.toBlob(resolve, 'image/jpeg', quality);
  });
}
```

### CORS

The backend allows all origins (`allow_origins=["*"]`). For production, restrict to your frontend domain.

---

## 8. Error Handling

- REST APIs return standard HTTP status codes (400, 401, 404, 500).
- JSON error body: `{"detail": "Message"}` (FastAPI default).
- WebSocket: On error, connection may close; check `onerror` and `onclose`.

---

## 9. CORS & Security

- **CORS:** Enabled for all origins in development.
- **MongoDB:** Connection string in `configs/config.yaml`; database: `driver_monitoring`.
- **Secrets:** Do not commit MongoDB credentials; use environment variables in production.

---

## Quick Reference – All Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Service info + API list |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |
| POST | `/api/login/` | Face login |
| POST | `/api/login/register` | Face registration |
| POST | `/api/sessions/start` | Start session |
| POST | `/api/sessions/end` | End session |
| POST | `/api/monitor/frame` | Single frame (REST) |
| POST | `/api/alerts/` | Create alert |
| GET | `/api/alerts/` | List alerts |
| GET | `/api/safety-score/` | Get daily scores |
| POST | `/api/safety-score/compute` | Compute & store daily score |
| WS | `/api/stream` | Real-time video stream |

---

**Contact:** For technical questions, reach out to the backend/ML team.
