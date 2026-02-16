# ---- Stage 0: Build jpegli (libjpeg.so.62 from libjxl) ----
FROM debian:bookworm-slim AS jpegli-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential git ca-certificates pkg-config \
    libbrotli-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch v0.11.1 https://github.com/libjxl/libjxl.git /libjxl \
    && cd /libjxl \
    && git submodule update --init --depth 1 third_party/highway third_party/skcms third_party/libjpeg-turbo \
    && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/jpegli \
             -DBUILD_TESTING=OFF \
             -DJPEGXL_ENABLE_TOOLS=ON \
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
    && make -j$(nproc) jpegli-static jpegli-libjpeg-shared cjxl djxl \
    && make install

# ---- Stage 1: Build MozJPEG from source ----
FROM debian:bookworm-slim AS mozjpeg-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake nasm build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG MOZJPEG_VERSION=4.1.5
RUN curl -L https://github.com/mozilla/mozjpeg/archive/refs/tags/v${MOZJPEG_VERSION}.tar.gz \
    | tar xz \
    && cd mozjpeg-${MOZJPEG_VERSION} \
    && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/mozjpeg \
             -DENABLE_SHARED=OFF \
             -DENABLE_STATIC=ON \
             -DPNG_SUPPORTED=OFF \
             .. \
    && make -j$(nproc) \
    && make install

# ---- Stage 2: Production image ----
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/amitray007/pare"
LABEL org.opencontainers.image.description="Serverless image compression API"
LABEL org.opencontainers.image.licenses="MIT"

# Copy jpegli libjpeg.so.62 (Pillow picks this up via ldconfig)
COPY --from=jpegli-builder /opt/jpegli/lib/libjpeg.so.62* /usr/local/lib/
# Copy JPEG XL CLI tools
COPY --from=jpegli-builder /opt/jpegli/bin/cjxl /usr/local/bin/cjxl
COPY --from=jpegli-builder /opt/jpegli/bin/djxl /usr/local/bin/djxl
RUN ldconfig

# Copy MozJPEG binaries (jpegtran always needed; cjpeg for JPEG_ENCODER=cjpeg fallback)
COPY --from=mozjpeg-builder /opt/mozjpeg/bin/cjpeg /usr/local/bin/cjpeg
COPY --from=mozjpeg-builder /opt/mozjpeg/bin/jpegtran /usr/local/bin/jpegtran

# Install system compression tools + codec libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    pngquant \
    gifsicle \
    webp \
    libheif-dev \
    libde265-dev \
    libaom-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Cloud Run sets $PORT; Uvicorn workers configurable
CMD ["sh", "-c", \
     "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers ${WORKERS:-4} --timeout-graceful-shutdown ${GRACEFUL_SHUTDOWN_TIMEOUT:-30}"]
