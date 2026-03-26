"""
Driver Safety System – Complete Pipeline (per PDF).
APIs: /api/login, /api/monitor, /api/alerts, /api/safety-score + sessions.

When run as __main__, this module only starts the FastAPI/uvicorn server on the VM.
Face matching, calibration prompts, and monitoring UI are started from the local
machine via `python -m data_pipeline.client` (not from this file).
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import alerts, login, monitor, realtime_monitoring, safety_score, sessions
from app.server_warmup import preload_heavy_models
from utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def _app_lifespan(app: FastAPI):
    # Load DL YOLO + Qwen in a worker thread so startup doesn't block the event loop,
    # but still completes before traffic (lifespan runs before serving).
    logger.info("Preloading heavy models (DL verification — may take several minutes on CPU)...")
    await asyncio.get_running_loop().run_in_executor(None, preload_heavy_models)
    logger.info("Heavy model preload step finished")
    yield
    try:
        login.shutdown_dl_verification_executor()
    except Exception as e:
        logger.warning("DL executor shutdown: %s", e)


class AppFactory:
    """Application factory for production deployment."""

    @staticmethod
    def create() -> FastAPI:
        application = FastAPI(
            title="Driver Safety System",
            description="Complete pipeline: data ingestion, fusion, alerts, safety scoring",
            version="1.0.0",
            lifespan=_app_lifespan,
        )
        application.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        application.include_router(login.router)
        application.include_router(sessions.router)
        application.include_router(monitor.router)
        application.include_router(alerts.router)
        application.include_router(safety_score.router)
        application.include_router(realtime_monitoring.router)
        logger.info("FastAPI application created — all routers registered")
        return application


app = AppFactory.create()


@app.get("/")
def root():
    return {
        "service": "Driver Safety System",
        "apis": [
            "POST /api/login/",
            "POST /api/login/register",
            "POST /api/login/verify-dl",
            "POST /api/login/finalize-dl",
            "POST /api/sessions/start",
            "POST /api/sessions/end",
            "POST /api/monitor/frame",
            "POST /api/alerts/",
            "GET /api/alerts/",
            "GET /api/safety-score/",
            "POST /api/safety-score/compute",
            "GET /api/recalibrate",
            "WS /api/stream",
            "WS /api/login/ws/dl-verify",
        ],
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting uvicorn server on 0.0.0.0:5000")
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        reload_dirs=["app", "data_pipeline", "database", "src", "training", "utils", "configs"],
    )
