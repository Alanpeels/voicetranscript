#!/usr/bin/env bash
# exit on error
set -o errexit

# Update pip first
pip install --upgrade pip

# Install requirements from the specific file
pip install -r requirements.txt
