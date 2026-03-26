"""
/api/monitor – Data ingestion + real-time monitoring (PDF: frame + metadata timestamp, driver_id, GPS).
Accepts frame + metadata; returns processed result and optional alert from fusion.
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from src.fusion import FusionEngine, ModelOutputs
from src.schemas.payloads import GPSLocation, MonitorFrameResponse
from database import alert_repository
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/monitor", tags=["monitor"])

# Fusion engine per PDF
fusion_engine = FusionEngine()


def _parse_float(s: str | None) -> Optional[float]:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


@router.post("/frame", response_model=MonitorFrameResponse)
async def process_frame(
    driver_id: str = Form(...),
    session_id: Optional[str] = Form(None),
    gps_latitude: Optional[str] = Form(None),
    gps_longitude: Optional[str] = Form(None),
    frame: UploadFile = File(...),
):
    """
    Ingest one frame with metadata (driver_id, session_id, optional GPS). Per PDF: metadata tagging.
    In production, run fatigue/distraction/blink models on frame and pass outputs to fusion.
    """
    raw = await frame.read()
    if not raw:
        logger.warning("Empty frame received from driver=%s", driver_id)
        return MonitorFrameResponse(processed=False, driver_state="normal")
    logger.debug("POST /api/monitor/frame driver=%s session=%s (%d bytes)", driver_id, session_id, len(raw))

    # TODO: decode frame, run fatigue + distraction + blink/PERCLOS models, build ModelOutputs
    # For now use stub outputs so fusion can still run
    outputs = ModelOutputs(
        perclos=0.0,
        blink_duration_sec=0.0,
        blink_rate_low=False,
        fatigue_score=0.0,
        head_turned_away_sec=0.0,
        distraction_score=0.0,
        eye_closure_duration_sec=0.0,
    )

    result = fusion_engine.fuse(outputs)
    lat = _parse_float(gps_latitude)
    lon = _parse_float(gps_longitude)

    alert_response = None
    if result.alert_type and result.confidence_score > 0:
        alert_repository.insert_alert(
            driver_id=driver_id,
            session_id=session_id,
            alert_type=result.alert_type,
            confidence_score=result.confidence_score,
            gps_latitude=lat,
            gps_longitude=lon,
        )
        from src.schemas.payloads import AlertResponse
        alert_response = AlertResponse(
            driver_id=driver_id,
            session_id=session_id,
            alert_type=result.alert_type,
            confidence_score=result.confidence_score,
            timestamp=datetime.now(timezone.utc),
            gps=GPSLocation(latitude=lat, longitude=lon) if lat is not None and lon is not None else None,
        )

    return MonitorFrameResponse(
        processed=True,
        alert=alert_response,
        driver_state=result.driver_state,
    )
