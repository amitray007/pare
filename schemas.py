from typing import Optional

from pydantic import BaseModel, Field


class OptimizationConfig(BaseModel):
    """Optimization parameters (all optional with defaults)."""

    quality: int = Field(default=80, ge=1, le=100)
    strip_metadata: bool = True
    progressive_jpeg: bool = False
    png_lossy: bool = True
    max_reduction: Optional[float] = Field(
        default=None, ge=0, le=100,
        description="Cap size reduction at this percentage. The optimizer will "
        "search for the highest quality that stays within this limit.",
    )


class StorageConfig(BaseModel):
    """Storage upload configuration."""

    provider: str = Field(..., pattern=r"^(gcs)$")
    bucket: str
    path: str
    project: Optional[str] = None
    public: bool = False


class OptimizeRequest(BaseModel):
    """JSON body for URL-based optimization."""

    url: str
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    storage: Optional[StorageConfig] = None


class OptimizeResult(BaseModel):
    """Internal result passed between optimizer and response formatter."""

    success: bool
    original_size: int
    optimized_size: int
    reduction_percent: float
    format: str
    method: str
    optimized_bytes: bytes = b""
    message: Optional[str] = None


class StorageResult(BaseModel):
    """Storage upload result included in JSON responses."""

    provider: str
    url: str
    public_url: Optional[str] = None


class OptimizeResponse(BaseModel):
    """JSON response when storage is configured."""

    success: bool
    original_size: int
    optimized_size: int
    reduction_percent: float
    format: str
    method: str
    storage: Optional[StorageResult] = None
    message: Optional[str] = None


class EstimateResponse(BaseModel):
    """Response from the /estimate endpoint."""

    original_size: int
    original_format: str
    dimensions: dict
    color_type: Optional[str] = None
    bit_depth: Optional[int] = None
    estimated_optimized_size: int
    estimated_reduction_percent: float
    optimization_potential: str
    method: str
    already_optimized: bool
    confidence: str


class ErrorResponse(BaseModel):
    """Standard error response."""

    success: bool = False
    error: str
    message: str
    original_size: Optional[int] = None
    format: Optional[str] = None


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str = "ok"
    tools: dict
    version: str
