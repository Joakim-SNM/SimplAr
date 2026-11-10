#!/bin/bash
# SimplAr launcher for macOS (and Linux): double-click to start.
# The first start installs the Python packages listed in requirements.txt.
cd "$(dirname "$0")" || exit 1

stop() {
    echo
    echo "$1"
    read -r -p "Press Return to close this window. " _
    exit 1
}

if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    stop "SimplAr needs Python 3.10 or later. Install it from https://www.python.org/downloads/ and double-click launch.command again."
fi

if ! python3 -c 'import numpy, scipy, pandas, openpyxl, matplotlib, PyQt5' 2>/dev/null; then
    echo "Installing the packages SimplAr needs. This happens once and takes a few minutes."
    echo
    python3 -m pip install -r requirements.txt || stop "The installation failed; the messages above say why."
fi

python3 launcher.py
