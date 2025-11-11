#!/usr/bin/env bash

# Bootstrap a virtual environment, install dependencies, and start the Flask app.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating virtual environment at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
fi

echo "Activating virtual environment"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

PIP_BIN="${VENV_DIR}/bin/pip"

echo "Upgrading pip (once)"
"${PIP_BIN}" install --upgrade pip >/dev/null

echo "Installing requirements"
"${PIP_BIN}" install -r requirements_app.txt

echo "Starting Clay Millennium Proof Explorer"
exec "${VENV_DIR}/bin/python" proof_explorer.py
