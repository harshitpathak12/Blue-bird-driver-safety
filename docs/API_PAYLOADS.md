# Driver Safety System – API Payloads (per PDF)

This document lists the **request and response payloads** for each API, aligned with the DRIVER SAFETY SYSTEM – COMPLETE PIPELINE DESIGN.

---

## 1. Data ingestion / metadata (PDF §1.1)

- **Metadata tagging:** `timestamp`, `driver_id`; optional GPS (Latitude + Longitude).
- Used in **POST /api/monitor/frame** (form: `driver_id`, `session_id`, `gps_latitude`, `gps_longitude`, `frame` file).

---

## 2. POST /api/login/

**Purpose:** Login with face image → match → return driver_id, driver_name, age.

**Request:** Multipart form  
- `image`: file (required)  
- `driver_id`: string (optional; if provided, verify face against this driver)

**Response:** `LoginResponse`
```json
{
  "driver_id": "string",
  "driver_name": "string",
  "age": 0,
  "message": "Login successful"
}
```

---

## 3. POST /api/login/register

**Purpose:** First-time registration: face image + name, age → store face_embedding, face_image_path.

**Request:** Multipart form  
- `name`: string (required)  
- `age`: int (optional)  
- `image`: file (required)

**Response:** `RegisterResponse`
```json
{
  "driver_id": "string",
  "driver_name": "string",
  "age": 0,
  "message": "Registration successful"
}
```

---

## 4. POST /api/sessions/start

**Request body:** `SessionStartRequest`
```json
{ "driver_id": "string" }
```

**Response:** `SessionStartResponse`
```json
{
  "session_id": "string",
  "driver_id": "string",
  "start_time": "datetime",
  "status": "active",
  "message": "Session started"
}
```

---

## 5. POST /api/sessions/end

**Request body:** `SessionEndRequest`
```json
{ "session_id": "string" }
```

**Response:** `SessionEndResponse`
```json
{
  "session_id": "string",
  "end_time": "datetime",
  "status": "ended",
  "message": "Session ended"
}
```

---

## 6. POST /api/monitor/frame

**Purpose:** Ingest one frame with metadata (driver_id, session_id, optional GPS). Returns processed result and optional alert.

**Request:** Multipart form  
- `driver_id`: string (required)  
- `session_id`: string (optional)  
- `gps_latitude`: string (optional)  
- `gps_longitude`: string (optional)  
- `frame`: file (required)

**Response:** `MonitorFrameResponse`
```json
{
  "processed": true,
  "alert": null | AlertResponse,
  "driver_state": "normal" | "fatigue" | "distraction" | "sleep"
}
```

`AlertResponse` (when present):
```json
{
  "alert_id": "string",
  "driver_id": "string",
  "session_id": "string",
  "alert_type": "fatigue" | "distraction" | "sleep",
  "confidence_score": 0.0,
  "timestamp": "datetime",
  "gps": { "latitude": 0.0, "longitude": 0.0 } | null
}
```

---

## 7. POST /api/alerts/

**Purpose:** Create alert and store in DB (PDF: driver_id, session_id, alert_type, confidence_score, timestamp, GPS).

**Request body:** `AlertCreateBody`
```json
{
  "driver_id": "string",
  "session_id": "string",
  "alert_type": "fatigue" | "distraction" | "sleep",
  "confidence_score": 0.0,
  "gps": { "latitude": 0.0, "longitude": 0.0 } | null
}
```

**Response:** `AlertResponse` (same shape as above, with `alert_id` and `timestamp` set by server).

---

## 8. GET /api/alerts/

**Query params:**  
- `driver_id`: string (optional)  
- `session_id`: string (optional)  
- `limit`: int (default 100, max 500)

**Response:** `AlertListResponse`
```json
{
  "alerts": [ AlertResponse, ... ],
  "total": 0
}
```

---

## 9. GET /api/safety-score/

**Purpose:** Daily safety scores (PDF: driver_id, date, fatigue_count, distraction_count, sleep_count, safety_score).  
Score formula: `100 - (fatigue_count×5) - (distraction_count×3) - (sleep_count×10)`.  
90–100 → Safe, 70–89 → Moderate Risk, &lt;70 → High Risk.

**Query params:**  
- `driver_id`: string (required)  
- `date_from`: string YYYY-MM-DD (optional)  
- `date_to`: string YYYY-MM-DD (optional)

**Response:** `SafetyScoreResponse`
```json
{
  "driver_id": "string",
  "daily_scores": [
    {
      "driver_id": "string",
      "date": "YYYY-MM-DD",
      "fatigue_count": 0,
      "distraction_count": 0,
      "sleep_count": 0,
      "safety_score": 0.0,
      "risk_level": "Safe" | "Moderate Risk" | "High Risk"
    }
  ]
}
```

---

## 10. POST /api/safety-score/compute

**Query params:**  
- `driver_id`: string (required)  
- `score_date`: string YYYY-MM-DD (required)

**Purpose:** Compute daily score from alerts for the given date and upsert into `daily_scores` collection.

**Response:** Object with `driver_id`, `date`, `fatigue_count`, `distraction_count`, `sleep_count`, `safety_score`, `risk_level`.
