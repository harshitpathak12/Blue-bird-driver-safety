from __future__ import annotations

from fastapi import APIRouter, WebSocket

from src.pipelines import pipeline

# Keep monitoring routes under the same `/api` prefix used by other server endpoints.
router = APIRouter(prefix="/api", tags=["realtime-monitoring"])


@router.websocket("/stream")
async def stream(websocket: WebSocket):
    """
    Real-time driver activity monitoring.

    Delegates to `src.pipelines.pipeline` (``RealtimeFramePipeline`` singleton).
    """
    await pipeline.stream(websocket)


@router.get("/recalibrate")
async def recalibrate():
    """
    Trigger distraction model recalibration on the server.
    """
    pipeline.distraction_detector.recalibrate()
    return {"status": "ok", "message": "Recalibration started."}

