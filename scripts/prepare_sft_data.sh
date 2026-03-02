#!/bin/bash
# prepare_sft_data.sh — Download training data from GCS for SFT
#
# Downloads annotated videos and training JSONs to a local directory
# structured for the cosmos-rl SFT pipeline.
#
# Usage:
#   # Download to default /data/autolane-handoff
#   ./scripts/prepare_sft_data.sh
#
#   # Download to custom path
#   ./scripts/prepare_sft_data.sh /mnt/data/autolane-handoff

set -euo pipefail

DATA_DIR="${1:-/data/autolane-handoff}"
GCS_BUCKET="gs://autolane-handoff-datagen"

echo "=== Preparing SFT Training Data ==="
echo "  GCS bucket: ${GCS_BUCKET}"
echo "  Local dir:  ${DATA_DIR}"

# Create directory structure
mkdir -p "${DATA_DIR}/training"
mkdir -p "${DATA_DIR}/data/annotations"

# Download training JSONs
echo ""
echo "--- Downloading training JSONs ---"
gsutil -m cp \
    "${GCS_BUCKET}/data/training/llava_train.json" \
    "${GCS_BUCKET}/data/training/llava_val.json" \
    "${GCS_BUCKET}/data/training/llava_all.json" \
    "${DATA_DIR}/training/"

echo "  Train entries: $(python3 -c "import json; print(len(json.load(open('${DATA_DIR}/training/llava_train.json'))))")"
echo "  Val entries:   $(python3 -c "import json; print(len(json.load(open('${DATA_DIR}/training/llava_val.json'))))")"

# Download annotated videos (these are referenced by video paths in the training data)
echo ""
echo "--- Downloading annotated videos ---"
echo "  This may take a while (downloading all annotated.mp4 files)..."
gsutil -m rsync -r \
    "${GCS_BUCKET}/data/annotations/" \
    "${DATA_DIR}/data/annotations/"

# Count downloaded videos
VIDEO_COUNT=$(find "${DATA_DIR}/data/annotations" -name "annotated.mp4" | wc -l | tr -d ' ')
echo "  Downloaded ${VIDEO_COUNT} annotated videos"

# Verify paths match training data
echo ""
echo "--- Verifying data integrity ---"
python3 -c "
import json
from pathlib import Path

data_dir = '${DATA_DIR}'
train = json.load(open(f'{data_dir}/training/llava_train.json'))
missing = []
for entry in train:
    video = Path(data_dir) / entry['video']
    if not video.exists():
        missing.append(entry['video'])

if missing:
    print(f'  WARNING: {len(missing)} videos not found:')
    for m in missing[:5]:
        print(f'    {m}')
    if len(missing) > 5:
        print(f'    ... and {len(missing) - 5} more')
else:
    print(f'  All {len(train)} training videos verified.')
"

echo ""
echo "=== Data preparation complete ==="
echo ""
echo "Update sft_config.toml paths:"
echo "  annotation_path = \"${DATA_DIR}/training/llava_train.json\""
echo "  media_path = \"${DATA_DIR}\""
