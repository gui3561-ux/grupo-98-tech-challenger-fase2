#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "====================================="
echo " Starting Zero-Touch Setup..."
echo "====================================="

# Check if the specific data file is missing
if [ ! -f "data/raw/ml-100k/u.data" ]; then
    echo "[Data Check] Dataset missing! Fetching data..."

    mkdir -p data/raw
    curl -L -o data/raw/ml-100k.zip https://files.grouplens.org/datasets/movielens/ml-100k.zip
    unzip -o data/raw/ml-100k.zip -d data/raw
    rm data/raw/ml-100k.zip  # Clean up the zip file to save space

    echo "[Data Check] Data successfully downloaded and extracted."
else
    echo "[Data Check] Dataset already exists. Skipping download."
fi

echo "====================================="
echo " Starting Application..."
echo "====================================="

# Execute the main command passed to the Docker container
exec "$@"
