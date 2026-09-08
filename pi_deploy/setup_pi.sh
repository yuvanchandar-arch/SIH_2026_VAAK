#!/bin/bash
# setup_pi.sh — Raspberry Pi 4 Environment Setup Script for Dracarys KWS

set -e

echo "=== Dracarys KWS: Raspberry Pi 4 Deployment Setup ==="

# 1. OS Version & Config Path Detection (Bookworm vs Bullseye)
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "Detected OS: $NAME $VERSION"
fi

if [ -f /boot/firmware/config.txt ]; then
    CONFIG_FILE="/boot/firmware/config.txt"
elif [ -f /boot/config.txt ]; then
    CONFIG_FILE="/boot/config.txt"
else
    CONFIG_FILE="/boot/config.txt"
fi

echo "Target Raspberry Pi config file: $CONFIG_FILE"

# 2. Configure INMP441 I2S Microphone Overlay
echo "Checking I2S overlay configuration in $CONFIG_FILE..."
if grep -q "dtoverlay=googlevoicehat-soundcard" "$CONFIG_FILE"; then
    echo "  I2S overlay (googlevoicehat-soundcard) already enabled."
else
    echo "  Adding I2S microphone overlay to $CONFIG_FILE..."
    echo "" | sudo tee -a "$CONFIG_FILE"
    echo "# INMP441 I2S Microphone Overlay for Dracarys KWS" | sudo tee -a "$CONFIG_FILE"
    echo "dtoverlay=googlevoicehat-soundcard" | sudo tee -a "$CONFIG_FILE"
    echo "  I2S overlay added. Note: Reboot required if not previously enabled."
fi

# 3. Install System Dependencies & Setup Python Virtual Environment
echo ""
echo "Installing system dependencies..."
sudo apt-get update && sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    libasound2-dev \
    portaudio19-dev \
    ffmpeg \
    alsa-utils

echo ""
echo "Setting up Python virtual environment (venv)..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Created virtual environment in ~/pi_deploy/venv"
fi

echo "Installing Python edge dependencies into venv..."
./venv/bin/pip install --upgrade pip
./venv/bin/pip install tflite-runtime "numpy<2" sounddevice websocket-client psutil

# 4. Verify I2S Capture Devices
echo ""
echo "=== Audio Capture Device Verification ==="
if command -v arecord &> /dev/null; then
    arecord -l || echo "Warning: No capture devices listed yet. Ensure INMP441 wiring and reboot if required."
else
    echo "alsa-utils not found. Install via: sudo apt-get install -y alsa-utils"
fi

echo ""
echo "=== Deployment Setup Complete ==="
echo "To run live keyword spotting:"
echo "  source venv/bin/activate"
echo "  python3 live_kws.py"
