#!/usr/bin/env python3
"""
BaseAL Dataset Generator

Generates a dataset in a BaseAL friendly format.

Pipeline:
1. Split audio into fixed-length segments (length dependent on model selected)
2. Generate embeddings per segment using pretrained models (BirdNET, Perch, etc.) - using bacpipe
3. Convert onset/offset labels to per-segment labels
4. Package into BaseAL format

Required Format:
    dataset_name/
    ├── data/
    |   ├── birdnet/
    │   |   ├── file1_000_003.wav
    │   |   ├── file1_003_006.wav
    |   |   ├── ...
    │   └── perch_v2/
    │       └── ...
    ├── embeddings/
    │   ├── birdnet/
    │   │   ├── file1_000_003_birdnet.npy
    │   │   └── ...
    │   └── perch_v2/
    │       └── ...
    ├── labels.csv        # filename, label, validation
    └── metadata.csv      # All segment metadata
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from utils.helpers import convert_for_json
from utils.embeddings import initialise, generate_embeddings
from utils.segment_labels import split_metadata_to_segments, create_labels_csv, SegmentConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate BaseAL-formatted dataset with embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        default="birdnet",
        choices=["birdnet", "perch_v2"],
        help="Embedding model to use (perch_v2 only runs on Linux/WSL)"
    )

    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="Path to directory containing audio files"
    )

    parser.add_argument(
        "--metadata-path",
        type=str,
        required=True,
        help="Path to metadata parquet file"
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output directory for the BaseAL dataset"
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation"
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.0,
        help="Minimum overlap for segment labeling"
    )

    parser.add_argument(
        "--no-event-label",
        type=str,
        default="no_call",
        help="Label for segments without events"
    )

    return parser.parse_args()


def setup_directories(output_path: Path, model: str) -> tuple[Path, Path]:
    """Create output directory structure."""
    output_path.mkdir(exist_ok=True)

    seg_path = output_path / "data" / model
    emb_path = output_path / "embeddings" / model

    seg_path.mkdir(exist_ok=True, parents=True)
    emb_path.mkdir(exist_ok=True, parents=True)

    return seg_path, emb_path


def save_metadata(segment_df: pd.DataFrame, output_path: Path):
    """Save segment metadata to CSV, converting numpy arrays to JSON strings."""
    csv_df = segment_df.copy()

    for col in ['segment_events', 'segment_event_clusters', 'ebird_code_multilabel', 'ebird_code_secondary']:
        if col in csv_df.columns:
            csv_df[col] = csv_df[col].apply(lambda x: json.dumps(convert_for_json(x)))

    csv_df.to_csv(output_path / "metadata.csv", index=False, encoding='utf-8')


def main():
    args = parse_args()

    # Convert paths
    audio_path = Path(args.audio_path)
    metadata_path = Path(args.metadata_path)
    output_path = Path(args.output_path)

    # Validate inputs
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio path does not exist: {audio_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

    print(f"Model: {args.model}")
    print(f"Audio path: {audio_path}")
    print(f"Metadata path: {metadata_path}")
    print(f"Output path: {output_path}")
    print()

    # Setup directories
    seg_path, emb_path = setup_directories(output_path, args.model)

    # Initialize embedder
    print("Initializing embedder...")
    embedder = initialise(model_name=args.model)

    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = generate_embeddings(
        audio_dir=audio_path,
        embedder=embedder,
        segments_dir=seg_path,
        output_dir=emb_path
    )

    # Load and process metadata
    print("\nProcessing metadata...")
    df = pd.read_parquet(metadata_path)
    print(f"Original: {len(df)} files")

    # Calculate segment duration from model
    duration = embedder.model.segment_length / embedder.model.sr

    # Configure segmentation
    config = SegmentConfig(
        segment_duration=duration,
        min_overlap=args.min_overlap,
        no_event_label=args.no_event_label
    )

    # Split into segments
    segment_df = split_metadata_to_segments(df, config)
    print(f"Segments: {len(segment_df)} ({segment_df['has_event'].sum()} with events)")

    # Save metadata
    print("\nSaving metadata...")
    save_metadata(segment_df, output_path)

    # Create and save labels
    print("Creating labels...")
    labels_df = create_labels_csv(
        segment_df,
        validation_fraction=args.validation_fraction,
        random_seed=args.random_seed
    )
    labels_df.to_csv(output_path / "labels.csv", index=False, encoding='utf-8')

    print(f"\nDataset generation complete!")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
