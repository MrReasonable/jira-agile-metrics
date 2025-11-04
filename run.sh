#!/bin/bash

# Load environment variables from .env if it exists
if [ -f "$PWD/.env" ]; then
  ENV_FILE="--env-file $PWD/.env"
else
  ENV_FILE=""
fi

# Always mount config.yaml if it exists
if [ -f "$PWD/config.yml" ]; then
  CONFIG_MOUNT="-v $PWD/config.yml:/data/config.yml"
else
  CONFIG_MOUNT=""
fi

# Ensure output directory exists on host
if [ ! -d "$PWD/output" ]; then
  mkdir "$PWD/output"
fi

# Always mount output directory
OUTPUT_MOUNT="-v $PWD/output:/data/output"

# Check if Docker image exists, build if it doesn't
IMAGE_NAME="jira-agile-metrics-dev"
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "\n==================== Building Docker Image ===================="
  echo "Docker image '$IMAGE_NAME' not found. Building..."
  docker build -t "$IMAGE_NAME" -f Dockerfile.develop .
  if [ $? -ne 0 ]; then
    echo "Failed to build Docker image. Exiting."
    exit 1
  fi
  echo "Docker image built successfully!"
else
  echo "\n==================== Docker Image Found ===================="
  echo "Using existing Docker image: $IMAGE_NAME"
fi

# NOTE: Set output paths in config.yml to /app/output or a relative path that resolves there

echo "\n==================== Running jira-agile-metrics CLI ===================="
docker run --rm \
  -v "$PWD:/app" \
  -w /app \
  $CONFIG_MOUNT \
  $OUTPUT_MOUNT \
  $ENV_FILE \
  -e JIRA_USERNAME="$JIRA_USERNAME" \
  -e JIRA_PASSWORD="$JIRA_PASSWORD" \
  -e JIRA_URL="$JIRA_URL" \
  -e PYTHONPATH=/app \
  -e MPLBACKEND=agg \
  jira-agile-metrics-dev \
  python -m jira_agile_metrics.cli "$@" -vv

EXIT_CODE_CLI=$?

echo "\n==================== Exit Codes ===================="
echo "jira-agile-metrics CLI exit code: $EXIT_CODE_CLI" 
