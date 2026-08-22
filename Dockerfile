# Reproducible CPU-only batch-inference image for ML-NIDS.
# Model artifacts and input data are mounted at runtime; they are not embedded
# in the image.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements-runtime.txt ./
# Resolve the current Debian security release at build time; pinning this
# package would prevent timely security updates to the inherited base image.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --no-install-recommends -y util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-runtime.txt \
    # Package installation is a build-time concern. Removing pip from the
    # deployed image eliminates its unused vendored dependencies and reduces
    # the runtime attack surface.
    && rm -rf /usr/local/lib/python3.14/site-packages/pip \
        /usr/local/lib/python3.14/site-packages/pip-*.dist-info \
        /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.14

COPY src ./src

RUN useradd --create-home --uid 10001 nids \
    && mkdir -p /app/input /app/models /app/output /app/logs \
    && chown -R nids:nids /app

USER nids

# Supply --model, --data, and --output after the image name.
ENTRYPOINT ["python", "-m", "src.predict"]
