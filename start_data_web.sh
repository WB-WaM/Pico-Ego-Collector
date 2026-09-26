#!/usr/bin/env bash
# Start the Pico data-processing/annotation web page.
#
# One-time environment setup: ./install.sh
#
# Start:
#   ./start_data_web.sh

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HOST="${PICO_WEB_HOST:-127.0.0.1}"
PORT="${PICO_WEB_PORT:-8765}"
URL="http://${HOST}:${PORT}"

# Export concurrency. The web server itself remains a single service; these
# values are inherited by the lerobot export subprocess launched by the UI.
# Override them when needed, for example:
#   PICO_EXPORT_PREPROCESS_WORKERS=12 ./start_data_web.sh
CPU_THREADS="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
if (( CPU_THREADS >= 16 )); then
  DEFAULT_PREPROCESS_WORKERS=8
  DEFAULT_IMAGE_WRITER_THREADS=8
else
  DEFAULT_PREPROCESS_WORKERS=$(( CPU_THREADS / 2 ))
  (( DEFAULT_PREPROCESS_WORKERS < 1 )) && DEFAULT_PREPROCESS_WORKERS=1
  DEFAULT_IMAGE_WRITER_THREADS=$(( CPU_THREADS / 2 ))
  (( DEFAULT_IMAGE_WRITER_THREADS < 2 )) && DEFAULT_IMAGE_WRITER_THREADS=2
fi
export PICO_EXPORT_PREPROCESS_WORKERS="${PICO_EXPORT_PREPROCESS_WORKERS:-${DEFAULT_PREPROCESS_WORKERS}}"
export PICO_EXPORT_IMAGE_WRITER_THREADS="${PICO_EXPORT_IMAGE_WRITER_THREADS:-${DEFAULT_IMAGE_WRITER_THREADS}}"
export PICO_EXPORT_IMAGE_WRITER_PROCESSES="${PICO_EXPORT_IMAGE_WRITER_PROCESSES:-0}"
export PICO_EXPORT_ENCODER_THREADS="${PICO_EXPORT_ENCODER_THREADS:-8}"
export PICO_EXPORT_VIDEO_PRESET="${PICO_EXPORT_VIDEO_PRESET:-veryfast}"
export PICO_EXPORT_VIDEO_GOP="${PICO_EXPORT_VIDEO_GOP:-2}"
export PICO_EXPORT_VIDEO_CRF="${PICO_EXPORT_VIDEO_CRF:-30}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found. Initialize Conda first." >&2
  exit 1
fi

# Make `conda activate` available in a non-interactive shell.
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate pico
cd "${PROJECT_ROOT}"

echo "Active environment: ${CONDA_DEFAULT_ENV}"
echo "Web address: ${URL}"
echo "Export environment: lerobot"
echo "Available CPU threads: ${CPU_THREADS}"
echo "Export preprocessing workers: ${PICO_EXPORT_PREPROCESS_WORKERS}"
echo "Export image writer threads: ${PICO_EXPORT_IMAGE_WRITER_THREADS}"
echo "Image writer processes: ${PICO_EXPORT_IMAGE_WRITER_PROCESSES}"
echo "Video encoder threads: ${PICO_EXPORT_ENCODER_THREADS}"
echo "H.264 preset：${PICO_EXPORT_VIDEO_PRESET}"
echo "Video GOP/CRF: ${PICO_EXPORT_VIDEO_GOP}/${PICO_EXPORT_VIDEO_CRF}"

# Resume stopped old instances after TERM, wait for exit, and check the address
# before launching. Only this project's annotator is eligible for termination.
python scripts/stop_annotator.py --project-root "${PROJECT_ROOT}" --host "${HOST}" --port "${PORT}"

python scripts/serve_annotator.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --export-conda-env lerobot &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Set PICO_OPEN_BROWSER=0 on headless/remote machines.
if [[ "${PICO_OPEN_BROWSER:-1}" != "0" ]] && command -v xdg-open >/dev/null 2>&1; then
  xdg-open "${URL}" >/dev/null 2>&1 &
fi

wait "${SERVER_PID}"
