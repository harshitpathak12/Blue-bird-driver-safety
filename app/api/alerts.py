"""
/api/alerts – Store and retrieve alerts (PDF: driver_id, session_id, alert_type, confidence_score, timestamp, GPS).
"""

from fastapi import APIRouter, Query
from bson import ObjectId

from src.schemas.payloads import AlertCreateBody, AlertListResponse, AlertResponse, GPSLocation
from database import alert_repository
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


def _doc_to_alert_response(doc: dict) -> AlertResponse:
    gps = None
    if doc.get("gps"):
        gps = GPSLocation(latitude=doc["gps"]["latitude"], longitude=doc["gps"]["longitude"])
    return AlertResponse(
        alert_id=str(doc["_id"]) if doc.get("_id") else None,
        driver_id=doc["driver_id"],
        session_id=doc.get("session_id"),
        alert_type=doc["alert_type"],
        confidence_score=doc.get("confidence_score", 0.0),
        timestamp=doc["timestamp"],
        gps=gps,
    )


@router.post("/", response_model=AlertResponse)
async def create_alert(body: AlertCreateBody):
    """Create alert and store in DB (PDF structure with confidence_score and GPS)."""
    logger.info("POST /api/alerts/ driver=%s type=%s", body.driver_id, body.alert_type)
    row = alert_repository.insert_alert(
        driver_id=body.driver_id,
        session_id=body.session_id,
        alert_type=body.alert_type,
        confidence_score=body.confidence_score,
        gps_latitude=body.gps.latitude if body.gps else None,
        gps_longitude=body.gps.longitude if body.gps else None,
    )
    gps = None
    if row.get("gps"):
        gps = GPSLocation(latitude=row["gps"]["latitude"], longitude=row["gps"]["longitude"])
    return AlertResponse(
        alert_id=str(row["_id"]),
        driver_id=row["driver_id"],
        session_id=row.get("session_id"),
        alert_type=row["alert_type"],
        confidence_score=row["confidence_score"],
        timestamp=row["timestamp"],
        gps=gps,
    )


@router.get("/", response_model=AlertListResponse)
async def list_alerts(
    driver_id: str | None = Query(None),
    session_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    """List alerts with optional filter by driver_id and/or session_id."""
    docs = alert_repository.get_alerts(driver_id=driver_id, session_id=session_id, limit=limit)
    alerts = [_doc_to_alert_response(d) for d in docs]
    logger.debug("GET /api/alerts/ returned %d alerts (driver=%s)", len(alerts), driver_id)
    return AlertListResponse(alerts=alerts, total=len(alerts))
