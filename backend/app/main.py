import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings          # must come first: loads .env into os.environ
from app.db import init_db
from app.routers import auth, scans, stats, report


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.model_status = {"preload": "skipped (PRELOAD_MODELS=false)"}
    if settings.preload_models:
        # Imported here so `uvicorn` still starts (and reports the error in /health)
        # even if an ML dependency is missing.
        try:
            from app.pipeline.compliance_engine import warm_up
            # Loads PaddleOCR, the YOLO-seg checkpoint, probes Ollama. Runs in a
            # worker thread so the event loop isn't blocked; weights then stay
            # resident for every request.
            app.state.model_status = await asyncio.to_thread(warm_up)
        except Exception as exc:
            app.state.model_status = {"preload": f"FAILED: {exc}"}
    yield


app = FastAPI(
    title="PackSure API",
    description=(
        "Automated compliance verification for packaged commodities under the "
        "Legal Metrology (Packaged Commodities) Rules, 2011. SIH26034."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"service": "PackSure API", "status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "models": getattr(app.state, "model_status", {})}


app.include_router(auth.router)
app.include_router(scans.router)
app.include_router(stats.router)
app.include_router(report.router)
