#!/bin/bash
# SessionStart hook for Claude Code on the web: rebuilds the RMG-Py development environment.
#
# Steps (each is skipped if already done, so the hook is safe to run repeatedly):
#   1. Install Miniforge (conda) into $CONDA_DIR
#   2. Create the `rmg_env` conda environment from environment.yml
#   3. Clone RMG-database next to this repository (../RMG-database), where RMG looks for it by default
#   4. Build and install RMG-Py (`make install`, which compiles the Cython extensions), or rebuild
#      incrementally (`make build`) if it was installed before
#   5. Install developer tools (pytest-xdist for parallel tests, py-spy for profiling)
#   6. Put rmg_env on the PATH for the session
#
# The first run takes a while (creating the environment downloads several GB); afterwards the
# container state is cached and the hook only does the incremental rebuild.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

REPO_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
CONDA_DIR="${CONDA_DIR:-/opt/miniforge}"
ENV_NAME="rmg_env"
DATABASE_DIR="$(dirname "$REPO_DIR")/RMG-database"
CA_BUNDLE="/root/.ccr/ca-bundle.crt"  # CA bundle of the web sandbox's HTTPS proxy, if present

log() { echo "[session-start] $*" >&2; }

# 1. Miniforge
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  log "Installing Miniforge into $CONDA_DIR"
  installer="$(mktemp -d)/miniforge.sh"
  curl -fsSL -o "$installer" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh"
  bash "$installer" -b -p "$CONDA_DIR" >&2
  rm -f "$installer"
fi
if [ -f "$CA_BUNDLE" ]; then
  "$CONDA_DIR/bin/conda" config --set ssl_verify "$CA_BUNDLE" >&2
fi

# 2. Conda environment
if [ ! -x "$CONDA_DIR/envs/$ENV_NAME/bin/python" ]; then
  log "Creating the $ENV_NAME environment from environment.yml (this takes a while)"
  "$CONDA_DIR/bin/conda" env create --file "$REPO_DIR/environment.yml" --name "$ENV_NAME" >&2
fi

# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 3. RMG-database
if [ ! -d "$DATABASE_DIR/input" ]; then
  log "Cloning RMG-database into $DATABASE_DIR"
  git clone --depth 1 https://github.com/ReactionMechanismGenerator/RMG-database "$DATABASE_DIR" >&2
fi

# 4. Build RMG-Py
cd "$REPO_DIR"
if [ ! -f .installed ]; then
  log "Building and installing RMG-Py (make install)"
  make install >&2
else
  log "Rebuilding changed Cython extensions (make build)"
  make build >&2
fi

# 5. Developer tools
python -m pip install --quiet --root-user-action=ignore pytest-xdist py-spy >&2

# 6. Use rmg_env for the rest of the session
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  {
    echo "export CONDA_PREFIX=\"$CONDA_DIR/envs/$ENV_NAME\""
    echo "export CONDA_DEFAULT_ENV=\"$ENV_NAME\""
    echo "export PATH=\"$CONDA_DIR/envs/$ENV_NAME/bin:$CONDA_DIR/bin:\$PATH\""
  } >> "$CLAUDE_ENV_FILE"
fi

log "RMG-Py environment is ready"
