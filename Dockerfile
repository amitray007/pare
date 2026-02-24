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
    && dpkg -i --force-depends libjxl_*.deb libjxl-dev_*.deb \
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

# Auto-discover all runtime shared libraries that libvips needs via ldd
# (3 passes to resolve transitive dependencies)
RUN mkdir -p /runtime-libs \
    && for pass in 1 2 3; do \
         { ldd /usr/local/lib/libvips.so.42 2>/dev/null; \
           find /runtime-libs -type f -name '*.so*' -exec ldd {} + 2>/dev/null; } \
         | awk '/=>/ && $3 ~ /^\// {print $3}' | sort -u | while read lib; do \
           bn=$(basename "$lib"); \
           [ -f "/runtime-libs/$bn" ] || cp -aL "$lib" /runtime-libs/ 2>/dev/null || true; \
         done; \
       done

# ---- Stage 1: Production image ----
FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/amitray007/pare"
LABEL org.opencontainers.image.description="Serverless image compression API"
LABEL org.opencontainers.image.licenses="MIT"

# Copy libvips built from source
COPY --from=libvips-builder /usr/local/lib/ /usr/local/lib/
COPY --from=libvips-builder /usr/local/include/ /usr/local/include/

# Copy runtime shared libraries (codec deps from builder)
COPY --from=libvips-builder /runtime-libs/ /usr/lib/x86_64-linux-gnu/
RUN ldconfig

# gifsicle is kept for animated GIF inter-frame optimization
RUN apt-get update && apt-get install -y --no-install-recommends \
    gifsicle \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

CMD ["sh", "-c", \
     "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers ${WORKERS:-4} --timeout-graceful-shutdown ${GRACEFUL_SHUTDOWN_TIMEOUT:-30}"]
