"""
/api/safety-score – Daily safety scores (PDF: driver_id, date, fatigue_count, distraction_count, sleep_count, safety_score).
"""

from datetime import date, datetime, timedelta
from fastapi import APIRouter, Query

from src.schemas.payloads import DailyScoreResponse, SafetyScoreResponse
from src.scoring import SafetyScoring, risk_level_from_score
from database import alert_repository, daily_scores_repository
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/safety-score", tags=["safety-score"])


def _count_alerts_by_type(driver_id: str, date_str: str) -> tuple[int, int, int]:
    """Return (fatigue_count, distraction_count, sleep_count) for the given driver and date."""
    docs = alert_repository.get_alerts(driver_id=driver_id, limit=10000)
    fatigue = distraction = sleep = 0
    for d in docs:
        ts = d.get("timestamp")
        if ts and getattr(ts, "date", None):
            d_date = ts.date().isoformat()
        elif hasattr(ts, "strftime"):
            d_date = ts.strftime("%Y-%m-%d")
        else:
            continue
        if d_date != date_str:
            continue
        t = (d.get("alert_type") or "").lower()
        if t == "fatigue":
            fatigue += 1
        elif t == "distraction":
            distraction += 1
        elif t == "sleep":
            sleep += 1
    return fatigue, distraction, sleep


@router.get("/", response_model=SafetyScoreResponse)
async def get_safety_score(
    driver_id: str = Query(..., description="Driver ID"),
    date_from: str | None = Query(None, description="YYYY-MM-DD"),
    date_to: str | None = Query(None, description="YYYY-MM-DD"),
):
    """
    Get daily safety scores for driver. Per PDF: 90–100 Safe, 70–89 Moderate Risk, <70 High Risk.
    If no stored daily_scores, computes from alerts for requested date range.
    """
    scores = daily_scores_repository.get_daily_scores(driver_id, date_from=date_from, date_to=date_to)
    if scores:
        daily = [
            DailyScoreResponse(
                driver_id=s["driver_id"],
                date=s["date"],
                fatigue_count=s.get("fatigue_count", 0),
                distraction_count=s.get("distraction_count", 0),
                sleep_count=s.get("sleep_count", 0),
                safety_score=s["safety_score"],
                risk_level=risk_level_from_score(s["safety_score"]),
            )
            for s in scores
        ]
        return SafetyScoreResponse(driver_id=driver_id, daily_scores=daily)

    # Compute from alerts if no stored scores
    if date_from and date_to:
        start = datetime.strptime(date_from, "%Y-%m-%d").date()
        end = datetime.strptime(date_to, "%Y-%m-%d").date()
    else:
        start = end = date.today()
    daily = []
    d = start
    while d <= end:
        date_str = d.isoformat()
        fc, dc, sc = _count_alerts_by_type(driver_id, date_str)
        score = SafetyScoring.compute(fc, dc, sc)
        daily.append(
            DailyScoreResponse(
                driver_id=driver_id,
                date=date_str,
                fatigue_count=fc,
                distraction_count=dc,
                sleep_count=sc,
                safety_score=score,
                risk_level=risk_level_from_score(score),
            )
        )
        d = d + timedelta(days=1)
    return SafetyScoreResponse(driver_id=driver_id, daily_scores=daily)


@router.post("/compute")
async def compute_and_store_daily_score(
    driver_id: str = Query(...),
    score_date: str = Query(..., description="YYYY-MM-DD"),
):
    """Compute daily score from alerts for the given date and upsert into daily_scores collection."""
    fc, dc, sc = _count_alerts_by_type(driver_id, score_date)
    safety_score = SafetyScoring.compute(fc, dc, sc)
    logger.info("Computed daily score for driver=%s date=%s: %.1f (fatigue=%d distraction=%d sleep=%d)", driver_id, score_date, safety_score, fc, dc, sc)
    try:
        d = date.fromisoformat(score_date)
    except ValueError:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid score_date, use YYYY-MM-DD")
    daily_scores_repository.upsert_daily_score(
        driver_id=driver_id,
        score_date=d,
        fatigue_count=fc,
        distraction_count=dc,
        sleep_count=sc,
        safety_score=safety_score,
    )
    return {
        "driver_id": driver_id,
        "date": score_date,
        "fatigue_count": fc,
        "distraction_count": dc,
        "sleep_count": sc,
        "safety_score": safety_score,
        "risk_level": risk_level_from_score(safety_score),
    }
