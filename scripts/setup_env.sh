#!/bin/bash

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p output
mkdir -p logs

echo "Environment setup completed!"