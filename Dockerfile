FROM python:3.14-slim

LABEL version="0.6"
LABEL description="Produce charts and data files of Agile metrics extracted \
from JIRA."

# Install requirements - do everything in one RUN to avoid persisting build dependencies
COPY ./requirements-prod.txt /requirements-prod.txt
RUN apt-get update && \
    # Install build dependencies
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        libffi-dev && \
    # Install Python packages
    pip install --no-cache-dir -r /requirements-prod.txt && \
    # Install runtime libraries (needed for scipy/numpy to work)
    apt-get install -y --no-install-recommends \
        libopenblas0 \
        liblapack3 && \
    # Remove build dependencies (in same layer so they're not persisted)
    apt-get purge -y --auto-remove \
        build-essential \
        python3-dev \
        gfortran \
        libffi-dev \
        libopenblas-dev \
        liblapack-dev && \
    # Clean up apt cache and temporary files
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt/archives/* && \
    rm -rf /var/cache/apt/archives/partial/* && \
    rm /requirements-prod.txt && \
    find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete


# Outputs will be written to the /data volume 
WORKDIR /data
VOLUME /data

# Install app and binary
COPY . /app
RUN pip install --no-cache-dir /app && \
    # Comprehensive cleanup: remove app source, pip cache, and Python bytecode
    rm -rf /app && \
    rm -rf /root/.cache/pip && \
    find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

# Run with a headless matplotlib backend
ENV MPLBACKEND="agg"
# Enable colored output in Docker
ENV FORCE_COLOR="1"

ENTRYPOINT ["jira-agile-metrics"]
