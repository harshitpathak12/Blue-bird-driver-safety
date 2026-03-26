"""API payload schemas."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class GPSLocation(BaseModel):
    latitude: float = Field(..., description="Latitude")
    longitude: float = Field(..., description="Longitude")


class LoginRequest(BaseModel):
    driver_id: Optional[str] = Field(None, description="If provided, verify face against this driver")


class LoginResponse(BaseModel):
    driver_id: str
    driver_name: str
    age: Optional[int] = None
    message: str = "Login successful"


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1)
    age: Optional[int] = Field(None, ge=1, le=120)


class RegisterLiveBody(BaseModel):
    name: str = Field(..., min_length=1)
    age: Optional[int] = Field(None, ge=1, le=120)
    image_base64: str = Field(..., description="Base64-encoded image from live camera capture")


class RegisterResponse(BaseModel):
    # Strict mode: registration is stored as a pending record until DL verification succeeds.
    # `driver_id` stays empty until DL verification is complete.
    pending_id: str
    driver_id: Optional[str] = None
    driver_name: str
    age: Optional[int] = None
    message: str = "Registration successful"


class SessionStartResponse(BaseModel):
    session_id: str
    driver_id: str
    start_time: datetime
    status: str = "active"
    message: str = "Session started"


class SessionEndResponse(BaseModel):
    session_id: str
    end_time: datetime
    status: str = "ended"
    message: str = "Session ended"


class FrameMetadata(BaseModel):
    timestamp: Optional[datetime] = None
    driver_id: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    gps: Optional[GPSLocation] = None


class MonitorFrameRequest(BaseModel):
    driver_id: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    gps: Optional[GPSLocation] = None


class MonitorFrameResponse(BaseModel):
    processed: bool = True
    alert: Optional["AlertResponse"] = None
    driver_state: Optional[str] = None


class AlertCreateBody(BaseModel):
    driver_id: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    alert_type: str = Field(..., description="e.g. fatigue, distraction, sleep")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    gps: Optional[GPSLocation] = None


class AlertResponse(BaseModel):
    alert_id: Optional[str] = None
    driver_id: str
    session_id: Optional[str] = None
    alert_type: str
    confidence_score: float
    timestamp: datetime
    gps: Optional[GPSLocation] = None


class AlertListResponse(BaseModel):
    alerts: list[AlertResponse]
    total: int


class DailyScoreResponse(BaseModel):
    driver_id: str
    date: str = Field(..., description="Date in YYYY-MM-DD")
    fatigue_count: int = 0
    distraction_count: int = 0
    sleep_count: int = 0
    safety_score: float = Field(..., ge=0, le=100)
    risk_level: str = Field(..., description="Safe | Moderate Risk | High Risk")


class SafetyScoreResponse(BaseModel):
    driver_id: str
    daily_scores: list[DailyScoreResponse]


class SafetyScoreQueryParams(BaseModel):
    driver_id: str = Field(..., min_length=1)
    date_from: Optional[str] = None
    date_to: Optional[str] = None


MonitorFrameResponse.model_rebuild()

