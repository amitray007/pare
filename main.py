from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from exceptions import PareError
from middleware import SecurityMiddleware
from routers import estimate, health, optimize
from utils.logging import get_logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: configure logging, verify tools. Shutdown: close connections."""
    # --- Startup ---
    setup_logging()
    logger = get_logger("main")

    tools = health.check_tools()
    missing = [name for name, available in tools.items() if not available]
    if missing:
        logger.warning(
            f"Missing tools: {missing}",
            extra={"context": {"missing_tools": missing}},
        )

    yield

    # --- Shutdown ---
    # Uvicorn's --timeout-graceful-shutdown handles connection draining.
    # This hook is for application-level cleanup.
    from security.rate_limiter import _redis

    if _redis:
        await _redis.close()

    logger.info("Pare shutting down")


app = FastAPI(
    title="Pare",
    description="Image Optimizer Service",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
origins = [o.strip() for o in settings.allowed_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=[
        "X-Original-Size",
        "X-Optimized-Size",
        "X-Reduction-Percent",
        "X-Original-Format",
        "X-Optimization-Method",
        "X-Request-ID",
    ],
)


# SecurityMiddleware handles: request ID, auth, rate limiting, PareError responses
app.add_middleware(SecurityMiddleware)


@app.exception_handler(PareError)
async def pare_error_handler(request: Request, exc: PareError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.error_code,
            "message": exc.message,
            **exc.details,
        },
    )


# Routers
app.include_router(health.router)
app.include_router(optimize.router)
app.include_router(estimate.router)
