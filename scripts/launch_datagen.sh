#!/bin/bash
# launch_datagen.sh — Build, launch, and manage Cosmos Predict 2.5 datagen
# jobs on Vertex AI Custom Training.
#
# Usage:
#   # Upload inputs to GCS
#   ./scripts/launch_datagen.sh --upload --gcs-bucket autolane-handoff-datagen
#
#   # Build container image (Cloud Build — works from ARM Macs)
#   ./scripts/launch_datagen.sh --cloud-build \
#       --project autolane-handoff-20260221 \
#       --image us-central1-docker.pkg.dev/autolane-handoff-20260221/datagen/cosmos-predict25:latest
#
#   # Build container image locally (linux/amd64)
#   ./scripts/launch_datagen.sh --build \
#       --image us-central1-docker.pkg.dev/autolane-handoff-20260221/datagen/cosmos-predict25:latest
#
#   # Submit Image2World job
#   ./scripts/launch_datagen.sh --vertex --mode image2world \
#       --project autolane-handoff-20260221 \
#       --image us-central1-docker.pkg.dev/autolane-handoff-20260221/datagen/cosmos-predict25:latest \
#       --gcs-bucket autolane-handoff-datagen
#
#   # Submit 8 parallel batch jobs for Image2World
#   ./scripts/launch_datagen.sh --vertex --mode image2world --parallel 8 \
#       --image us-central1-docker.pkg.dev/autolane-handoff-20260221/datagen/cosmos-predict25:latest
#
#   # Submit both modes sequentially (polls between jobs)
#   ./scripts/launch_datagen.sh --vertex --mode both \
#       --project autolane-handoff-20260221 \
#       --image us-central1-docker.pkg.dev/autolane-handoff-20260221/datagen/cosmos-predict25:latest \
#       --gcs-bucket autolane-handoff-datagen
#
#   # Download completed clips from GCS
#   ./scripts/launch_datagen.sh --download --gcs-bucket autolane-handoff-datagen

set -euo pipefail

# --- Defaults ---
PROJECT_ID="${GCP_PROJECT_ID:-autolane-handoff-20260221}"
REGION="${GCP_REGION:-us-central1}"
GCS_BUCKET="${DATAGEN_GCS_BUCKET:-autolane-handoff-datagen}"
IMAGE=""
MODE="image2world"
ACTION=""
MACHINE_TYPE="a2-highgpu-1g"
ACCELERATOR_TYPE="NVIDIA_TESLA_A100"
ACCELERATOR_COUNT=1
MODEL_KEY="2B/post-trained"
PROMPT_IDS=""
NUM_INFERENCE_STEPS=35
SEED=42
NUM_SEEDS=1
USE_MODEL_CACHE="true"
DISABLE_GUARDRAILS="true"
POLL_INTERVAL=120  # seconds between job status checks
BATCH_INDEX=""
TOTAL_BATCHES=""
PARALLEL=""
SPOT="false"

# --- Parse CLI ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --upload)       ACTION="upload"; shift ;;
        --cloud-build)  ACTION="cloud-build"; shift ;;
        --build)        ACTION="build"; shift ;;
        --vertex)       ACTION="vertex"; shift ;;
        --download)     ACTION="download"; shift ;;
        --project)      PROJECT_ID="$2"; shift 2 ;;
        --region)       REGION="$2"; shift 2 ;;
        --gcs-bucket)   GCS_BUCKET="$2"; shift 2 ;;
        --image)        IMAGE="$2"; shift 2 ;;
        --mode)         MODE="$2"; shift 2 ;;
        --model)        MODEL_KEY="$2"; shift 2 ;;
        --prompt-ids)   PROMPT_IDS="$2"; shift 2 ;;
        --steps)        NUM_INFERENCE_STEPS="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --num-seeds)    NUM_SEEDS="$2"; shift 2 ;;
        --machine-type) MACHINE_TYPE="$2"; shift 2 ;;
        --no-cache)     USE_MODEL_CACHE="false"; shift ;;
        --no-guardrails) DISABLE_GUARDRAILS="true"; shift ;;
        --guardrails)    DISABLE_GUARDRAILS="false"; shift ;;
        --batch)        BATCH_INDEX="$2"; shift 2 ;;
        --total-batches) TOTAL_BATCHES="$2"; shift 2 ;;
        --parallel)     PARALLEL="$2"; shift 2 ;;
        --spot)         SPOT="true"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$ACTION" ]]; then
    echo "Usage: $0 --upload|--cloud-build|--build|--vertex|--download [options]"
    echo ""
    echo "Actions:"
    echo "  --upload       Upload site photos + prompts to GCS"
    echo "  --cloud-build  Build Docker image via Cloud Build (for ARM Macs)"
    echo "  --build        Build Docker image locally (linux/amd64)"
    echo "  --vertex       Submit Vertex AI Custom Job"
    echo "  --download     Download completed clips from GCS"
    echo ""
    echo "Options:"
    echo "  --project ID        GCP project ID (default: $PROJECT_ID)"
    echo "  --region REGION     GCP region (default: $REGION)"
    echo "  --gcs-bucket NAME   GCS bucket name (default: $GCS_BUCKET)"
    echo "  --image URI         Container image URI (required for build/vertex)"
    echo "  --mode MODE         image2world|text2world|both (default: image2world)"
    echo "  --model KEY         NVIDIA model key (default: $MODEL_KEY)"
    echo "  --prompt-ids IDS    Comma-separated prompt IDs to filter"
    echo "  --steps N           Inference steps (default: $NUM_INFERENCE_STEPS)"
    echo "  --seed N            Random seed (default: $SEED)"
    echo "  --num-seeds N       Seeds per prompt for text2world (default: $NUM_SEEDS)"
    echo "  --machine-type TYPE Machine type (default: $MACHINE_TYPE)"
    echo "  --no-cache          Disable GCS model cache"
    echo "  --parallel N        Submit N parallel batch jobs (each gets 1/N of work)"
    echo "  --batch IDX         Batch index (0-based) for manual batch control"
    echo "  --total-batches N   Total number of batches for manual batch control"
    echo "  --spot              Use spot/preemptible VMs (cheaper, may be preempted)"
    exit 1
fi

# --- Upload ---
if [[ "$ACTION" == "upload" ]]; then
    echo "=== Uploading datagen inputs to gs://$GCS_BUCKET ==="

    echo "[1/2] Uploading site photos..."
    gsutil -m rsync -r datagen/site_photos/ "gs://$GCS_BUCKET/datagen/site_photos/"

    echo "[2/2] Uploading prompts..."
    gsutil -m rsync -r datagen/prompts/ "gs://$GCS_BUCKET/datagen/prompts/"

    echo "Done. Inputs uploaded to gs://$GCS_BUCKET/"
    exit 0
fi

# --- Cloud Build ---
if [[ "$ACTION" == "cloud-build" ]]; then
    if [[ -z "$IMAGE" ]]; then
        echo "Error: --image is required for --cloud-build"
        exit 1
    fi

    echo "=== Building container image via Cloud Build ==="
    echo "  Image: $IMAGE"
    echo "  Project: $PROJECT_ID"

    # Cloud Build requires a cloudbuild.yaml to use a non-default Dockerfile
    BUILD_CONFIG=$(mktemp /tmp/cloudbuild-XXXXXX)
    cat > "$BUILD_CONFIG" <<YAML
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-f', 'datagen/Dockerfile', '-t', '$IMAGE', '.']
images:
  - '$IMAGE'
timeout: 3600s
YAML

    gcloud builds submit . \
        --project="$PROJECT_ID" \
        --config="$BUILD_CONFIG" \
        --ignore-file=datagen/.dockerignore

    rm -f "$BUILD_CONFIG"

    echo "Done. Image pushed to $IMAGE"
    exit 0
fi

# --- Local Build ---
if [[ "$ACTION" == "build" ]]; then
    if [[ -z "$IMAGE" ]]; then
        echo "Error: --image is required for --build"
        exit 1
    fi

    echo "=== Building container image locally (linux/amd64) ==="
    echo "  Image: $IMAGE"

    docker build \
        --platform linux/amd64 \
        -f datagen/Dockerfile \
        -t "$IMAGE" \
        .

    echo "Pushing image..."
    docker push "$IMAGE"

    echo "Done. Image pushed to $IMAGE"
    exit 0
fi

# --- Download ---
if [[ "$ACTION" == "download" ]]; then
    echo "=== Downloading synthetic clips from gs://$GCS_BUCKET ==="

    mkdir -p data/synthetic_clips/image2world data/synthetic_clips/text2world

    echo "[1/2] Downloading image2world clips..."
    gsutil -m rsync -r "gs://$GCS_BUCKET/data/synthetic_clips/image2world/" data/synthetic_clips/image2world/ 2>/dev/null || echo "  (no image2world clips found)"

    echo "[2/2] Downloading text2world clips..."
    gsutil -m rsync -r "gs://$GCS_BUCKET/data/synthetic_clips/text2world/" data/synthetic_clips/text2world/ 2>/dev/null || echo "  (no text2world clips found)"

    echo "Done. Clips downloaded to data/synthetic_clips/"
    exit 0
fi

# --- Vertex AI Custom Job ---
if [[ "$ACTION" == "vertex" ]]; then
    if [[ -z "$IMAGE" ]]; then
        echo "Error: --image is required for --vertex"
        exit 1
    fi

    # Resolve HF_TOKEN from env or .env file
    if [[ -z "${HF_TOKEN:-}" ]] && [[ -f .env ]]; then
        HF_TOKEN=$(grep -E '^HF_TOKEN=' .env | cut -d= -f2- || true)
    fi
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "Error: HF_TOKEN not set. Export it or add to .env file."
        exit 1
    fi

    submit_job() {
        local job_mode="$1"
        local batch_idx="${2:-}"
        local batch_total="${3:-}"

        local batch_suffix=""
        if [[ -n "$batch_idx" ]]; then
            batch_suffix="-b$((batch_idx + 1))of${batch_total}"
        fi
        local job_name="datagen-${job_mode}${batch_suffix}-$(date +%Y%m%d-%H%M%S)"

        echo "=== Submitting Vertex AI Custom Job ===" >&2
        echo "  Job name:  $job_name" >&2
        echo "  Mode:      $job_mode" >&2
        if [[ -n "$batch_idx" ]]; then
            echo "  Batch:     $((batch_idx + 1)) of $batch_total" >&2
        fi
        echo "  Image:     $IMAGE" >&2
        echo "  Machine:   $MACHINE_TYPE" >&2
        echo "  GPU:       ${ACCELERATOR_COUNT}x $ACCELERATOR_TYPE" >&2
        echo "  Model:     $MODEL_KEY" >&2
        echo "  Bucket:    $GCS_BUCKET" >&2

        # Build config YAML
        local config_file
        config_file=$(mktemp /tmp/datagen-config-XXXXXX)

        cat > "$config_file" <<YAML
workerPoolSpecs:
  - machineSpec:
      machineType: $MACHINE_TYPE
      acceleratorType: $ACCELERATOR_TYPE
      acceleratorCount: $ACCELERATOR_COUNT
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 300
    containerSpec:
      imageUri: $IMAGE
      env:
        - name: DATAGEN_MODE
          value: "$job_mode"
        - name: DATAGEN_GCS_BUCKET
          value: "$GCS_BUCKET"
        - name: DATAGEN_MODEL_KEY
          value: "$MODEL_KEY"
        - name: DATAGEN_NUM_INFERENCE_STEPS
          value: "$NUM_INFERENCE_STEPS"
        - name: DATAGEN_SEED
          value: "$SEED"
        - name: DATAGEN_NUM_SEEDS
          value: "$NUM_SEEDS"
        - name: DATAGEN_USE_MODEL_CACHE
          value: "$USE_MODEL_CACHE"
        - name: DATAGEN_DISABLE_GUARDRAILS
          value: "$DISABLE_GUARDRAILS"
        - name: HF_TOKEN
          value: "$HF_TOKEN"
YAML

        # Add prompt IDs filter if specified
        if [[ -n "$PROMPT_IDS" ]]; then
            cat >> "$config_file" <<YAML
        - name: DATAGEN_PROMPT_IDS
          value: "$PROMPT_IDS"
YAML
        fi

        # Add batch config if specified
        if [[ -n "$batch_idx" ]]; then
            cat >> "$config_file" <<YAML
        - name: DATAGEN_BATCH_INDEX
          value: "$batch_idx"
        - name: DATAGEN_TOTAL_BATCHES
          value: "$batch_total"
YAML
        fi

        # Add spot/preemptible scheduling if requested
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
        echo "$job_name"  # only this goes to stdout for capture
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
            local all_done=true

            for job_name in "${job_names[@]}"; do
                # Skip already completed/failed jobs
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
                    *)
                        all_done=false
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
        echo "=== Submitting $num_batches parallel batch jobs for $MODE ==="
        echo ""

        declare -a job_names

        if [[ "$MODE" == "both" ]]; then
            # Submit I2W batches, poll, then T2W batches
            echo "--- Phase 1: Image2World ($num_batches batches) ---"
            i2w_jobs=()
            for ((i=0; i<num_batches; i++)); do
                job=$(submit_job "image2world" "$i" "$num_batches")
                i2w_jobs+=("$job")
                sleep 1  # stagger submissions slightly
            done
            poll_all_jobs "${i2w_jobs[@]}" || { echo "Some Image2World jobs failed."; exit 1; }

            echo ""
            echo "--- Phase 2: Text2World ($num_batches batches) ---"
            t2w_jobs=()
            for ((i=0; i<num_batches; i++)); do
                job=$(submit_job "text2world" "$i" "$num_batches")
                t2w_jobs+=("$job")
                sleep 1
            done
            poll_all_jobs "${t2w_jobs[@]}" || { echo "Some Text2World jobs failed."; exit 1; }

            echo ""
            echo "=== All batches complete for both modes ==="
        else
            for ((i=0; i<num_batches; i++)); do
                job=$(submit_job "$MODE" "$i" "$num_batches")
                job_names+=("$job")
                sleep 1  # stagger submissions slightly
            done
            poll_all_jobs "${job_names[@]}"
        fi

        exit $?
    fi

    # --- Single batch mode (manual --batch/--total-batches) ---
    if [[ -n "$BATCH_INDEX" && -n "$TOTAL_BATCHES" ]]; then
        if [[ "$MODE" == "both" ]]; then
            job1=$(submit_job "image2world" "$BATCH_INDEX" "$TOTAL_BATCHES")
            poll_job "$job1" || { echo "Image2World batch job failed, aborting."; exit 1; }
            job2=$(submit_job "text2world" "$BATCH_INDEX" "$TOTAL_BATCHES")
            poll_job "$job2" || { echo "Text2World batch job failed."; exit 1; }
        else
            job=$(submit_job "$MODE" "$BATCH_INDEX" "$TOTAL_BATCHES")
            poll_job "$job"
        fi
        exit $?
    fi

    # --- Standard single job mode ---
    if [[ "$MODE" == "both" ]]; then
        echo "=== Running both modes sequentially ==="
        echo ""

        # Image2World first
        job1=$(submit_job "image2world")
        poll_job "$job1" || { echo "Image2World job failed, aborting."; exit 1; }

        echo ""

        # Text2World second
        job2=$(submit_job "text2world")
        poll_job "$job2" || { echo "Text2World job failed."; exit 1; }

        echo ""
        echo "=== Both jobs complete ==="
    else
        job=$(submit_job "$MODE")
        poll_job "$job"
    fi

    exit 0
fi
