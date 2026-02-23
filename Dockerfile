# ---- Stage 0: Build libvips with jpegli and all codecs ----
FROM debian:bookworm-slim AS libvips-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential git ca-certificates pkg-config nasm curl \
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
    # Highway (required by libjxl)
    libhwy-dev \
    # Brotli (required by libjxl)
    libbrotli-dev \
    && rm -rf /var/lib/apt/lists/*

# Build libjxl with jpegli
RUN git clone --depth 1 --branch v0.11.1 https://github.com/libjxl/libjxl.git /libjxl \
    && cd /libjxl \
    && git submodule update --init --depth 1 third_party/skcms third_party/libjpeg-turbo \
    && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
             -DCMAKE_BUILD_TYPE=Release \
             -DBUILD_TESTING=OFF \
             -DJPEGXL_ENABLE_TOOLS=OFF \
             -DJPEGXL_ENABLE_DOXYGEN=OFF \
             -DJPEGXL_ENABLE_MANPAGES=OFF \
             -DJPEGXL_ENABLE_BENCHMARK=OFF \
             -DJPEGXL_ENABLE_EXAMPLES=OFF \
             -DJPEGXL_ENABLE_FUZZERS=OFF \
             -DJPEGXL_ENABLE_JPEGLI=ON \
             -DJPEGXL_ENABLE_JPEGLI_LIBJPEG=ON \
             -DJPEGXL_ENABLE_SKCMS=ON \
             -DJPEGXL_ENABLE_SJPEG=OFF \
             -DJPEGXL_ENABLE_OPENEXR=OFF \
             .. \
    && make -j$(nproc) \
    && make install \
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

# ---- Stage 1: Production image ----
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/amitray007/pare"
LABEL org.opencontainers.image.description="Serverless image compression API"
LABEL org.opencontainers.image.licenses="MIT"

# Copy libvips and all codec libraries
COPY --from=libvips-builder /usr/local/lib/ /usr/local/lib/
COPY --from=libvips-builder /usr/local/include/ /usr/local/include/
RUN ldconfig

# gifsicle is kept for animated GIF inter-frame optimization
RUN apt-get update && apt-get install -y --no-install-recommends \
    gifsicle \
    libglib2.0-0 \
    libexpat1 \
    libpng16-16 \
    libwebp7 \
    libheif1 \
    libaom3 \
    libde265-0 \
    libx265-199 \
    libtiff6 \
    libcgif0 \
    libimagequant0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

CMD ["sh", "-c", \
     "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers ${WORKERS:-4} --timeout-graceful-shutdown ${GRACEFUL_SHUTDOWN_TIMEOUT:-30}"]
