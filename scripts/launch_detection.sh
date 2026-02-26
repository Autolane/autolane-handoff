#!/bin/bash
# launch_detection.sh — Build and launch Grounding DINO detection jobs on
# Vertex AI Custom Training.
#
# Usage:
#   # Build container image (Cloud Build)
#   ./scripts/launch_detection.sh --cloud-build
#
#   # Submit single detection job
#   ./scripts/launch_detection.sh --vertex
#
#   # Submit 4 parallel batch jobs on spot VMs
#   ./scripts/launch_detection.sh --vertex --parallel 4 --spot
#
#   # Download results
#   ./scripts/launch_detection.sh --download

set -euo pipefail

# --- Defaults ---
PROJECT_ID="${GCP_PROJECT_ID:-autolane-handoff-20260221}"
REGION="${GCP_REGION:-us-central1}"
GCS_BUCKET="${DATAGEN_GCS_BUCKET:-autolane-handoff-datagen}"
IMAGE="us-central1-docker.pkg.dev/autolane-handoff-20260221/datagen/grounding-dino:latest"
MODE="both"
ACTION=""
MACHINE_TYPE="n1-standard-8"
ACCELERATOR_TYPE="NVIDIA_TESLA_T4"
ACCELERATOR_COUNT=1
BOX_THRESHOLD="0.35"
TEXT_THRESHOLD="0.25"
POLL_INTERVAL=120
BATCH_INDEX=""
TOTAL_BATCHES=""
PARALLEL=""
SPOT="false"

# --- Parse CLI ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --cloud-build)  ACTION="cloud-build"; shift ;;
        --vertex)       ACTION="vertex"; shift ;;
        --download)     ACTION="download"; shift ;;
        --project)      PROJECT_ID="$2"; shift 2 ;;
        --region)       REGION="$2"; shift 2 ;;
        --gcs-bucket)   GCS_BUCKET="$2"; shift 2 ;;
        --image)        IMAGE="$2"; shift 2 ;;
        --mode)         MODE="$2"; shift 2 ;;
        --machine-type) MACHINE_TYPE="$2"; shift 2 ;;
        --gpu-type)     ACCELERATOR_TYPE="$2"; shift 2 ;;
        --box-threshold) BOX_THRESHOLD="$2"; shift 2 ;;
        --text-threshold) TEXT_THRESHOLD="$2"; shift 2 ;;
        --batch)        BATCH_INDEX="$2"; shift 2 ;;
        --total-batches) TOTAL_BATCHES="$2"; shift 2 ;;
        --parallel)     PARALLEL="$2"; shift 2 ;;
        --spot)         SPOT="true"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$ACTION" ]]; then
    echo "Usage: $0 --cloud-build|--vertex|--download [options]"
    echo ""
    echo "Actions:"
    echo "  --cloud-build  Build Docker image via Cloud Build"
    echo "  --vertex       Submit Vertex AI Custom Job"
    echo "  --download     Download detection results from GCS"
    echo ""
    echo "Options:"
    echo "  --mode MODE         image2world|text2world|both (default: both)"
    echo "  --parallel N        Submit N parallel batch jobs"
    echo "  --spot              Use spot/preemptible VMs"
    echo "  --box-threshold F   Box confidence threshold (default: 0.35)"
    echo "  --text-threshold F  Text confidence threshold (default: 0.25)"
    exit 1
fi

# --- Cloud Build ---
if [[ "$ACTION" == "cloud-build" ]]; then
    echo "=== Building Grounding DINO container via Cloud Build ==="
    echo "  Image: $IMAGE"
    echo "  Project: $PROJECT_ID"

    BUILD_CONFIG=$(mktemp /tmp/cloudbuild-XXXXXX)
    trap 'rm -f "$BUILD_CONFIG"' EXIT
    cat > "$BUILD_CONFIG" <<YAML
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-f', 'detection/Dockerfile', '-t', '$IMAGE', '.']
images:
  - '$IMAGE'
timeout: 3600s
YAML

    gcloud builds submit . \
        --project="$PROJECT_ID" \
        --config="$BUILD_CONFIG"

    rm -f "$BUILD_CONFIG"
    trap - EXIT

    echo "Done. Image pushed to $IMAGE"
    exit 0
fi

# --- Download ---
if [[ "$ACTION" == "download" ]]; then
    echo "=== Downloading detection results from gs://$GCS_BUCKET ==="

    mkdir -p data/detection/image2world data/detection/text2world

    echo "[1/2] Downloading image2world detections..."
    if ! stderr=$(gsutil -m rsync -r "gs://$GCS_BUCKET/data/detection/image2world/" data/detection/image2world/ 2>&1); then
        if echo "$stderr" | grep -q "No URLs matched"; then
            echo "  (none found)"
        else
            echo "  Error: $stderr" >&2
        fi
    fi

    echo "[2/2] Downloading text2world detections..."
    if ! stderr=$(gsutil -m rsync -r "gs://$GCS_BUCKET/data/detection/text2world/" data/detection/text2world/ 2>&1); then
        if echo "$stderr" | grep -q "No URLs matched"; then
            echo "  (none found)"
        else
            echo "  Error: $stderr" >&2
        fi
    fi

    echo "Done. Results downloaded to data/detection/"
    exit 0
fi

# --- Vertex AI Custom Job ---
if [[ "$ACTION" == "vertex" ]]; then

    submit_job() {
        local batch_idx="${1:-}"
        local batch_total="${2:-}"

        local batch_suffix=""
        if [[ -n "$batch_idx" ]]; then
            batch_suffix="-b$((batch_idx + 1))of${batch_total}"
        fi
        local job_name="detection-gdino${batch_suffix}-$(date +%Y%m%d-%H%M%S)"

        echo "=== Submitting Vertex AI Detection Job ===" >&2
        echo "  Job name:  $job_name" >&2
        echo "  Mode:      $MODE" >&2
        if [[ -n "$batch_idx" ]]; then
            echo "  Batch:     $((batch_idx + 1)) of $batch_total" >&2
        fi
        echo "  Machine:   $MACHINE_TYPE + $ACCELERATOR_TYPE" >&2

        local config_file
        config_file=$(mktemp /tmp/detection-config-XXXXXX)
        trap 'rm -f "$config_file"' RETURN

        cat > "$config_file" <<YAML
workerPoolSpecs:
  - machineSpec:
      machineType: $MACHINE_TYPE
      acceleratorType: $ACCELERATOR_TYPE
      acceleratorCount: $ACCELERATOR_COUNT
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 200
    containerSpec:
      imageUri: $IMAGE
      env:
        - name: DETECTION_GCS_BUCKET
          value: "$GCS_BUCKET"
        - name: DETECTION_MODE
          value: "$MODE"
        - name: DETECTION_BOX_THRESHOLD
          value: "$BOX_THRESHOLD"
        - name: DETECTION_TEXT_THRESHOLD
          value: "$TEXT_THRESHOLD"
YAML

        if [[ -n "$batch_idx" ]]; then
            cat >> "$config_file" <<YAML
        - name: DETECTION_BATCH_INDEX
          value: "$batch_idx"
        - name: DETECTION_TOTAL_BATCHES
          value: "$batch_total"
YAML
        fi

        if [[ "$SPOT" == "true" ]]; then
            cat >> "$config_file" <<YAML
scheduling:
  strategy: SPOT
YAML
        fi

        gcloud ai custom-jobs create \
            --project="$PROJECT_ID" \
            --region="$REGION" \
            --display-name="$job_name" \
            --config="$config_file" >&2

        rm -f "$config_file"
        echo "$job_name"
    }

    poll_job() {
        local job_name="$1"
        echo ""
        echo "Polling job: $job_name (every ${POLL_INTERVAL}s)..."

        while true; do
            local state
            state=$(gcloud ai custom-jobs list \
                --project="$PROJECT_ID" \
                --region="$REGION" \
                --filter="displayName=$job_name" \
                --format="value(state)" \
                --limit=1 2>/dev/null)

            echo "  [$(date +%H:%M:%S)] Job state: $state"

            case "$state" in
                JOB_STATE_SUCCEEDED)
                    echo "Job $job_name completed successfully."
                    return 0
                    ;;
                JOB_STATE_FAILED|JOB_STATE_CANCELLED)
                    echo "Job $job_name ended with state: $state"
                    return 1
                    ;;
            esac

            sleep "$POLL_INTERVAL"
        done
    }

    poll_all_jobs() {
        local -a job_names=("$@")
        local total=${#job_names[@]}
        local -a completed=()
        local -a failed=()

        echo ""
        echo "Polling $total parallel jobs (every ${POLL_INTERVAL}s)..."

        while true; do
            for job_name in "${job_names[@]}"; do
                local skip=false
                for c in "${completed[@]+"${completed[@]}"}"; do
                    [[ "$c" == "$job_name" ]] && skip=true && break
                done
                for f in "${failed[@]+"${failed[@]}"}"; do
                    [[ "$f" == "$job_name" ]] && skip=true && break
                done
                $skip && continue

                local state
                state=$(gcloud ai custom-jobs list \
                    --project="$PROJECT_ID" \
                    --region="$REGION" \
                    --filter="displayName=$job_name" \
                    --format="value(state)" \
                    --limit=1 2>/dev/null)

                case "$state" in
                    JOB_STATE_SUCCEEDED)
                        completed+=("$job_name")
                        echo "  [$(date +%H:%M:%S)] $job_name: SUCCEEDED (${#completed[@]}/$total done)"
                        ;;
                    JOB_STATE_FAILED|JOB_STATE_CANCELLED)
                        failed+=("$job_name")
                        echo "  [$(date +%H:%M:%S)] $job_name: $state"
                        ;;
                esac
            done

            local finished=$(( ${#completed[@]} + ${#failed[@]} ))
            if [[ $finished -eq $total ]]; then
                break
            fi

            echo "  [$(date +%H:%M:%S)] Progress: ${#completed[@]} succeeded, ${#failed[@]} failed, $((total - finished)) running"
            sleep "$POLL_INTERVAL"
        done

        echo ""
        echo "=== All $total jobs finished ==="
        echo "  Succeeded: ${#completed[@]}"
        if [[ ${#failed[@]} -gt 0 ]]; then
            echo "  Failed: ${#failed[@]}"
            for f in "${failed[@]}"; do
                echo "    - $f"
            done
            return 1
        fi
        return 0
    }

    # --- Parallel batch mode ---
    if [[ -n "$PARALLEL" ]]; then
        num_batches="$PARALLEL"
        echo "=== Submitting $num_batches parallel detection jobs ==="
        declare -a job_names

        for ((i=0; i<num_batches; i++)); do
            job=$(submit_job "$i" "$num_batches")
            job_names+=("$job")
            sleep 1
        done
        poll_all_jobs "${job_names[@]}"
        exit $?
    fi

    # --- Single job ---
    if [[ -n "$BATCH_INDEX" && -n "$TOTAL_BATCHES" ]]; then
        job=$(submit_job "$BATCH_INDEX" "$TOTAL_BATCHES")
    else
        job=$(submit_job)
    fi
    poll_job "$job"

    exit $?
fi
