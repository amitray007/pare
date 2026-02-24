# ---- Stage 0: Build libvips with jpegli and all codecs ----
FROM debian:bookworm-slim AS libvips-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential ca-certificates pkg-config nasm curl \
    meson ninja-build gobject-introspection \
    # Core libvips deps
    libglib2.0-dev libexpat1-dev \
    # PNG
    libpng-dev zlib1g-dev \
    # WebP
    libwebp-dev \
    # HEIF (AVIF + HEIC)
    libheif-dev libaom-dev libde265-dev libx265-dev \
    # TIFF
    libtiff-dev \
    # GIF
    libcgif-dev \
    # libimagequant (for PNG palette quantization)
    libimagequant-dev \
    # Brotli (required by libjxl)
    libbrotli-dev \
    && rm -rf /var/lib/apt/lists/*

# Install libjxl + jpegli from pre-built Bookworm debs (includes libjpeg.so.62)
ARG LIBJXL_VERSION=0.11.1
RUN curl -L https://github.com/libjxl/libjxl/releases/download/v${LIBJXL_VERSION}/jxl-debs-amd64-debian-bookworm-v${LIBJXL_VERSION}.tar.gz \
    | tar xz \
    && dpkg -i --force-depends libjxl_*.deb libjxl-dev_*.deb libhwy_*.deb libhwy-dev_*.deb \
    && rm -f *.deb \
    && ldconfig

# Build libvips from source (linked against jpegli + all codecs above)
ARG VIPS_VERSION=8.16.0
RUN curl -L https://github.com/libvips/libvips/releases/download/v${VIPS_VERSION}/vips-${VIPS_VERSION}.tar.xz \
    | tar xJ \
    && cd vips-${VIPS_VERSION} \
    && meson setup build --prefix=/usr/local --buildtype=release \
         -Dintrospection=disabled \
    && cd build \
    && ninja \
    && ninja install \
    && ldconfig

# Collect libjxl/jpegli runtime .so files (not available via apt in production)
RUN mkdir -p /jxl-libs \
    && for pattern in libjxl.so* libjxl_cms.so* libjxl_threads.so* libjpeg.so* libhwy.so*; do \
         find /usr/lib /usr/local/lib -name "$pattern" -exec cp -aL {} /jxl-libs/ \; 2>/dev/null || true; \
       done \
    && ls -la /jxl-libs/

# ---- Stage 1: Production image ----
FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/amitray007/pare"
LABEL org.opencontainers.image.description="Serverless image compression API"
LABEL org.opencontainers.image.licenses="MIT"

# Install runtime dependencies via apt (reliable, handles transitive deps)
# Both builder and production are Bookworm so library versions match exactly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gifsicle \
    # GLib (core libvips dependency)
    libglib2.0-0 \
    # Image format codecs
    libexpat1 libpng16-16 zlib1g \
    libwebp7 libwebpmux3 libwebpdemux2 \
    libheif1 libaom3 libde265-0 libx265-199 \
    libtiff6 libcgif0 libimagequant0 \
    # Compression libraries
    libbrotli1 \
    # OpenMP (parallel processing in libvips)
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy libvips built from source
COPY --from=libvips-builder /usr/local/lib/ /usr/local/lib/
COPY --from=libvips-builder /usr/local/include/ /usr/local/include/

# Copy libjxl + jpegli runtime libs (from pre-built debs, not in Bookworm apt)
COPY --from=libvips-builder /jxl-libs/ /usr/lib/x86_64-linux-gnu/
RUN ldconfig

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build-time verification: ensure pyvips can load libvips with all codecs
RUN python -c "import pyvips; print(f'libvips {pyvips.version(0)}.{pyvips.version(1)}.{pyvips.version(2)} loaded OK')"

# Copy application
COPY . /app
WORKDIR /app

CMD ["sh", "-c", \
     "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers ${WORKERS:-4} --timeout-graceful-shutdown ${GRACEFUL_SHUTDOWN_TIMEOUT:-30}"]
