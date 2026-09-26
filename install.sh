#!/usr/bin/env bash
# Install the processing and export environments. Run from any directory.
set -Eeuo pipefail
trap 'echo "Installation failed at line ${LINENO}. Fix the error above and rerun ./install.sh." >&2' ERR

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0
SKIP_SYSTEM=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --skip-system) SKIP_SYSTEM=1 ;;
    -h|--help)
      echo "Usage: ./install.sh [--skip-system] [--dry-run]"
      echo "Requires Linux and Conda. Installs Ubuntu packages, submodules, and pico/lerobot environments."
      echo "--skip-system  Skip system packages already installed by an administrator"
      echo "--dry-run      Show commands without installing or modifying environments"
      exit 0 ;;
    *) echo "Unknown argument: $arg. See --help for usage." >&2; exit 2 ;;
  esac
done

run() {
  if (( DRY_RUN )); then
    printf '  '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [[ "$(uname -s)" != Linux ]]; then
  echo "This installer supports Linux; Ubuntu is recommended." >&2
  exit 1
fi
CONDA_BIN="${PICO_CONDA_BIN:-${CONDA_EXE:-$(command -v conda || true)}}"
if [[ -z "$CONDA_BIN" ]] || ! command -v "$CONDA_BIN" >/dev/null 2>&1; then
  echo "Conda not found. Install and initialize Miniconda/Miniforge, then rerun this script." >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "git not found. Install git first." >&2
  exit 1
fi
cd "$PROJECT_ROOT"
CONDA_BASE="$("$CONDA_BIN" info --base)"
CONDA_ENVS_JSON="$("$CONDA_BIN" env list --json)"

ensure_env() {
  local name="$1"
  local version exists
  exists="$("$CONDA_BASE/bin/python" -c \
    'import json, pathlib, sys; print(int(any(pathlib.Path(p).name == sys.argv[1] for p in json.load(sys.stdin)["envs"])))' \
    "$name" <<< "$CONDA_ENVS_JSON")"
  if [[ "$exists" == 1 ]]; then
    version="$("$CONDA_BIN" run -n "$name" python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    if [[ "$version" != "3.10" ]]; then
      echo "Existing $name environment uses Python $version; Python 3.10 is required. Update the environment first." >&2
      exit 1
    fi
    echo "Reusing existing $name environment (Python 3.10)."
  else
    run "$CONDA_BIN" create -n "$name" python=3.10 -y
  fi
}

echo "[1/5] Ubuntu system dependencies"
if (( ! SKIP_SYSTEM )); then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "apt-get not found. Install system dependencies manually, then use --skip-system." >&2
    exit 1
  fi
  system_packages=(build-essential ffmpeg libgl1 libegl1 libosmesa6 gvfs-backends)
  missing=()
  for package in "${system_packages[@]}"; do
    if [[ "$(dpkg-query -W -f='${db:Status-Status}' "$package" 2>/dev/null || true)" != installed ]]; then
      missing+=("$package")
    fi
  done
  if (( ${#missing[@]} )); then
    sudo_cmd=()
    if (( EUID != 0 )); then
      sudo_cmd=(sudo)
    fi
    run "${sudo_cmd[@]}" apt-get update
    run "${sudo_cmd[@]}" apt-get install -y "${missing[@]}"
  else
    echo "System dependencies are already installed."
  fi
else
  echo "Skipping system package installation."
fi

echo "[2/5] Initialize GMR / Wuji submodules"
run git submodule update --init --recursive

echo "[3/5] pico processing environment"
ensure_env pico
run "$CONDA_BIN" run --no-capture-output -n pico python -m pip install --upgrade pip setuptools wheel
run "$CONDA_BIN" run --no-capture-output -n pico python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
run "$CONDA_BIN" run --no-capture-output -n pico python -m pip install -r requirements.txt

echo "[4/5] lerobot export environment"
ensure_env lerobot
run "$CONDA_BIN" install -n lerobot -c conda-forge ffmpeg -y
run "$CONDA_BIN" run --no-capture-output -n lerobot python -m pip install --upgrade pip
run "$CONDA_BIN" run --no-capture-output -n lerobot python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
run "$CONDA_BIN" run --no-capture-output -n lerobot python -m pip install -r requirements_lerobot.txt

echo "[5/5] Check dependencies and core imports"
run "$CONDA_BIN" run --no-capture-output -n pico python -m pip check
run "$CONDA_BIN" run --no-capture-output -n lerobot python -m pip check
run "$CONDA_BIN" run --no-capture-output -n pico python -c \
  'import sys; sys.path.insert(0, "scripts"); import serve_annotator, retarget_pico_to_g1; from wuji_retargeting import Retargeter'
run "$CONDA_BIN" run --no-capture-output -n lerobot python -c \
  'import av; from lerobot.datasets.lerobot_dataset import LeRobotDataset'

if (( DRY_RUN )); then
  echo "Dry run complete. No environments were installed or modified."
else
  echo "Installation complete. Run ./start_data_web.sh and open http://127.0.0.1:8765"
fi
