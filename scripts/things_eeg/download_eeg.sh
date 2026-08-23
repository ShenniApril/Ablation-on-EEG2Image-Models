#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# --------------------------------
# Configuration
# --------------------------------
OUT_DIR="./data/things_eeg/raw_eeg"
BASE_URL="https://files.osf.io/v1/resources/crxs4/providers/googledrive"
SUBS=(02)

mkdir -p "$OUT_DIR"

# --------------------------------
# Step 1: Download all zip files (resume + retry)
# --------------------------------
echo "Starting dataset download..."

for sid in "${SUBS[@]}"; do
  zip_path="${OUT_DIR}/sub-${sid}.zip"
  url="${BASE_URL}/sub-${sid}.zip"

  echo "Downloading sub-${sid}.zip ..."
  aria2c \
  --continue=true \
  --max-connection-per-server=8 \
  --split=8 \
  --min-split-size=10M \
  --max-tries=10 \
  --retry-wait=5 \
  --file-allocation=none \
  --dir="$OUT_DIR" \
  --out="sub-${sid}.zip" \
  "$url"
done

echo "All files downloaded successfully."
echo

# --------------------------------
# Step 2: Unzip and remove only after success
# --------------------------------
echo "Extracting all zip files..."

for sid in "${SUBS[@]}"; do
  zip_path="${OUT_DIR}/sub-${sid}.zip"

  if [ -f "$zip_path" ]; then
    echo "Extracting sub-${sid}.zip ..."
    if unzip -n -q "$zip_path" -d "$OUT_DIR"; then
      echo "Removing sub-${sid}.zip"
      rm -f "$zip_path"
    else
      echo "Extraction failed for sub-${sid}.zip; keeping the file for inspection."
    fi
  else
    echo "sub-${sid}.zip already removed; skipping."
  fi
done

echo "All files extracted to: $OUT_DIR"