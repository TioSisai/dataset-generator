#!/bin/bash
#
# BaseAL Dataset Generator - Remote Server Runner
#
# Usage: ./run.sh [OPTIONS]
#
# Example:
#   ./run.sh --model birdnet --audio-path /data/audio --metadata-path /data/metadata.parquet --output-path /output/dataset
#

set -e  # Exit on error

# Default values
MODEL="birdnet"
AUDIO_PATH=""
METADATA_PATH=""
OUTPUT_PATH=""
VALIDATION_FRACTION=0.1
RANDOM_SEED=42
MIN_OVERLAP=0.0
NO_EVENT_LABEL="no_call"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --audio-path)
            AUDIO_PATH="$2"
            shift 2
            ;;
        --metadata-path)
            METADATA_PATH="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --validation-fraction)
            VALIDATION_FRACTION="$2"
            shift 2
            ;;
        --random-seed)
            RANDOM_SEED="$2"
            shift 2
            ;;
        --min-overlap)
            MIN_OVERLAP="$2"
            shift 2
            ;;
        --no-event-label)
            NO_EVENT_LABEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model              Embedding model (birdnet, perch_v2, biolingual) [default: birdnet]"
            echo "  --audio-path         Path to audio files directory (required)"
            echo "  --metadata-path      Path to metadata parquet file (required)"
            echo "  --output-path        Output directory for dataset (required)"
            echo "  --validation-fraction Fraction for validation split [default: 0.1]"
            echo "  --random-seed        Random seed for reproducibility [default: 42]"
            echo "  --min-overlap        Minimum overlap for segment labeling [default: 0.0]"
            echo "  --no-event-label     Label for segments without events [default: no_call]"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$AUDIO_PATH" ]]; then
    echo "Error: --audio-path is required"
    exit 1
fi

if [[ -z "$METADATA_PATH" ]]; then
    echo "Error: --metadata-path is required"
    exit 1
fi

if [[ -z "$OUTPUT_PATH" ]]; then
    echo "Error: --output-path is required"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment if it exists
if [[ -d "$SCRIPT_DIR/.venv" ]]; then
    echo "Activating virtual environment..."
    source "$SCRIPT_DIR/.venv/bin/activate"
elif [[ -d "$SCRIPT_DIR/venv" ]]; then
    echo "Activating virtual environment..."
    source "$SCRIPT_DIR/venv/bin/activate"
fi

echo "========================================"
echo "BaseAL Dataset Generator"
echo "========================================"
echo "Model:               $MODEL"
echo "Audio path:          $AUDIO_PATH"
echo "Metadata path:       $METADATA_PATH"
echo "Output path:         $OUTPUT_PATH"
echo "Validation fraction: $VALIDATION_FRACTION"
echo "Random seed:         $RANDOM_SEED"
echo "========================================"
echo ""

# Run the generator
python "$SCRIPT_DIR/generator.py" \
    --model "$MODEL" \
    --audio-path "$AUDIO_PATH" \
    --metadata-path "$METADATA_PATH" \
    --output-path "$OUTPUT_PATH" \
    --validation-fraction "$VALIDATION_FRACTION" \
    --random-seed "$RANDOM_SEED" \
    --min-overlap "$MIN_OVERLAP" \
    --no-event-label "$NO_EVENT_LABEL"

echo ""
echo "Done!"
