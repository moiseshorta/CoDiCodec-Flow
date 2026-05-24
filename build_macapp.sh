#!/usr/bin/env bash
# build_macapp.sh — Build a self-contained macOS .app for okachihuali / live.
#
# Produces dist/mac-arm64/Okachihuali.app inside electron-gui/, which contains:
#   - The Electron frontend (renderer + main process)
#   - A relocatable Python 3.11 runtime with torch (MPS), numpy, sounddevice,
#     librosa, einops, soundfile, huggingface_hub, tqdm, codicodec
#   - The flow/ source package (runtime engine + server)
#   - The codicodec/ source package
#   - The model checkpoint (runs/v3_okachihuali/last.pt and ema.pt if present)
#
# Result: ~3 GB self-contained .app that can be copied to any Apple Silicon
# Mac and launched without any external dependencies.
#
# Prereqs (one-time):
#   - conda env `codicodec-flow` exists and works (`npm start` from repo)
#   - `conda install -n codicodec-flow conda-pack -c conda-forge -y`
#   - `npm --prefix electron-gui install`
#
# Usage: ./build_macapp.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
ELECTRON_DIR="$REPO_ROOT/electron-gui"
STAGING="$REPO_ROOT/build-staging"
ENV_NAME="codicodec-flow"
ENV_PATH="$HOME/miniconda3/envs/$ENV_NAME"
CONDA_PACK="$ENV_PATH/bin/conda-pack"
TARGET_RUN="runs/v3_okachihuali"

echo "==> repo root:  $REPO_ROOT"
echo "==> staging:    $STAGING"
echo "==> conda env:  $ENV_PATH"

# ---------------------------------------------------------------------------- #
# 0. Sanity checks                                                             #
# ---------------------------------------------------------------------------- #
if [[ ! -d "$ENV_PATH" ]]; then
  echo "ERROR: conda env not found at $ENV_PATH" >&2
  exit 1
fi
if [[ ! -x "$CONDA_PACK" ]]; then
  echo "ERROR: conda-pack not installed in env. Run:" >&2
  echo "       conda install -n $ENV_NAME conda-pack -c conda-forge -y" >&2
  exit 1
fi
if [[ ! -f "$REPO_ROOT/$TARGET_RUN/last.pt" ]]; then
  echo "ERROR: checkpoint missing at $REPO_ROOT/$TARGET_RUN/last.pt" >&2
  exit 1
fi
if [[ ! -d "$ELECTRON_DIR/node_modules/electron-builder" ]]; then
  echo "==> installing npm deps (electron-builder etc.)"
  ( cd "$ELECTRON_DIR" && npm install )
fi

# ---------------------------------------------------------------------------- #
# 1. Wipe + create staging                                                     #
# ---------------------------------------------------------------------------- #
echo "==> [1/5] preparing staging area"
rm -rf "$STAGING"
mkdir -p "$STAGING"

# ---------------------------------------------------------------------------- #
# 2. Pack the conda env into a relocatable tree                                #
# ---------------------------------------------------------------------------- #
echo "==> [2/5] packing conda env (this takes ~1-2 min)"
PACK_TGZ="$STAGING/python-runtime.tar.gz"
"$CONDA_PACK" \
    --name "$ENV_NAME" \
    --output "$PACK_TGZ" \
    --ignore-editable-packages \
    --ignore-missing-files \
    --compress-level 1 \
    --force

mkdir -p "$STAGING/python-runtime"
tar -xzf "$PACK_TGZ" -C "$STAGING/python-runtime"
rm "$PACK_TGZ"

# Run conda-unpack to fix absolute paths embedded in scripts / shebangs.
echo "==> running conda-unpack to make env relocatable"
"$STAGING/python-runtime/bin/conda-unpack"

# Strip large artifacts we don't need at runtime to reclaim disk.
echo "==> stripping unused parts of python runtime"
rm -rf "$STAGING/python-runtime/conda-meta"          || true
rm -rf "$STAGING/python-runtime/share/man"           || true
rm -rf "$STAGING/python-runtime/share/doc"           || true
find "$STAGING/python-runtime" -type d -name "__pycache__" -prune -exec rm -rf {} + || true
find "$STAGING/python-runtime" -type d -name "tests"       -prune -exec rm -rf {} + || true
find "$STAGING/python-runtime" -type f -name "*.pyc"       -delete || true

# ---------------------------------------------------------------------------- #
# 3. Stage source packages (flow + codicodec)                                  #
# ---------------------------------------------------------------------------- #
echo "==> [3/5] staging source packages"

# `flow` package: copy the python source only (skip caches, samples, runs).
rsync -a --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    "$REPO_ROOT/flow/" "$STAGING/flow/"

# `codicodec` package: bundle the local checkout in full (it's an editable
# install in dev; in the packaged app we put it on PYTHONPATH explicitly).
rsync -a --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude '.git' \
    --exclude 'tests' \
    --exclude 'docs' \
    "$REPO_ROOT/codicodec/" "$STAGING/codicodec/"

# ---------------------------------------------------------------------------- #
# 4. Stage the checkpoints (every runs/<name>/last.pt becomes a selectable     #
#    model in the GUI's MODEL dropdown).                                       #
# ---------------------------------------------------------------------------- #
echo "==> [4/5] staging model checkpoints"
# Always include the primary target run.
RUN_DIRS=("$TARGET_RUN")
# Also include other known runs if their last.pt exists. Add more here as you
# train new models you want bundled into the .app.
for extra in "runs/v2"; do
  if [[ -f "$REPO_ROOT/$extra/last.pt" ]]; then
    RUN_DIRS+=("$extra")
  fi
done

for run in "${RUN_DIRS[@]}"; do
  echo "    + staging $run"
  mkdir -p "$STAGING/$run"
  cp "$REPO_ROOT/$run/last.pt" "$STAGING/$run/last.pt"
  if [[ -f "$REPO_ROOT/$run/ema.pt" ]]; then
    cp "$REPO_ROOT/$run/ema.pt" "$STAGING/$run/ema.pt"
  fi
done

echo "==> staging size:"
du -sh "$STAGING"/* | sort -h

# ---------------------------------------------------------------------------- #
# 5. Run electron-builder                                                      #
# ---------------------------------------------------------------------------- #
echo "==> [5/5] running electron-builder"
cd "$ELECTRON_DIR"
npx electron-builder --mac --arm64 --dir

APP_PATH="$ELECTRON_DIR/dist/mac-arm64/Okachihuali.app"
if [[ -d "$APP_PATH" ]]; then
  echo
  echo "================================================================="
  echo "  Build complete."
  echo "  App: $APP_PATH"
  echo "  Size: $(du -sh "$APP_PATH" | cut -f1)"
  echo "================================================================="
  echo
  echo "Launch with:  open '$APP_PATH'"
else
  echo "ERROR: build did not produce expected .app at $APP_PATH" >&2
  exit 1
fi
