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

# Copy MozJPEG binaries
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
