"""
Segment-wise label splitter for audio metadata.

This module converts file-level audio annotations (with onset/offset times)
into segment-level labels suitable for training models on fixed-length audio chunks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class SegmentConfig:
    """Configuration for segment splitting."""
    segment_duration: float = 5.0  # seconds
    min_overlap: float = 0.0  # minimum overlap (seconds) for an event to be included
    overlap_ratio: Optional[float] = None  # if set, use ratio instead of absolute overlap
    no_event_label: str = "no_call"  # label for segments with no detected events
    label_separator: str = ";"  # separator for multilabel output


def get_combined_labels(row: pd.Series, separator: str = ";") -> str:
    """
    Combine primary and secondary labels into a single semicolon-separated string.

    Args:
        row: A row from the metadata DataFrame
        separator: Separator to use between labels

    Returns:
        Combined label string (e.g., "mallar3;yellow2")
    """
    labels = []

    # Get primary labels from ebird_code_multilabel
    multilabel = row.get("ebird_code_multilabel")
    if multilabel is not None:
        if isinstance(multilabel, np.ndarray):
            labels.extend(multilabel.tolist())
        elif isinstance(multilabel, list):
            labels.extend(multilabel)

    # Get secondary labels
    secondary = row.get("ebird_code_secondary")
    if secondary is not None:
        if isinstance(secondary, np.ndarray):
            labels.extend(secondary.tolist())
        elif isinstance(secondary, list):
            labels.extend(secondary)

    # Remove duplicates while preserving order
    seen = set()
    unique_labels = []
    for label in labels:
        if label and label not in seen:
            seen.add(label)
            unique_labels.append(label)

    return separator.join(unique_labels) if unique_labels else row.get("ebird_code", "")


def event_overlaps_segment(
    event_onset: float,
    event_offset: float,
    seg_start: float,
    seg_end: float,
    config: SegmentConfig
) -> bool:
    """
    Check if an event overlaps with a segment based on config thresholds.

    Args:
        event_onset: Start time of the event (seconds)
        event_offset: End time of the event (seconds)
        seg_start: Start time of the segment (seconds)
        seg_end: End time of the segment (seconds)
        config: Segmentation configuration

    Returns:
        True if the event overlaps sufficiently with the segment
    """
    # Calculate overlap
    overlap_start = max(event_onset, seg_start)
    overlap_end = min(event_offset, seg_end)
    overlap_duration = max(0, overlap_end - overlap_start)

    if overlap_duration == 0:
        return False

    if config.overlap_ratio is not None:
        # Use ratio-based threshold
        event_duration = event_offset - event_onset
        if event_duration > 0:
            return (overlap_duration / event_duration) >= config.overlap_ratio
        return False
    else:
        # Use absolute overlap threshold
        return overlap_duration >= config.min_overlap


def get_events_in_segment(
    events: np.ndarray,
    seg_start: float,
    seg_end: float,
    config: SegmentConfig
) -> list[int]:
    """
    Get indices of events that overlap with a segment.

    Args:
        events: Array of [onset, offset] pairs
        seg_start: Start time of the segment (seconds)
        seg_end: End time of the segment (seconds)
        config: Segmentation configuration

    Returns:
        List of event indices that overlap with the segment
    """
    overlapping_indices = []

    if events is None or len(events) == 0:
        return overlapping_indices

    for i, event in enumerate(events):
        if event_overlaps_segment(event[0], event[1], seg_start, seg_end, config):
            overlapping_indices.append(i)

    return overlapping_indices


def split_row_into_segments(
    row: pd.Series,
    config: SegmentConfig
) -> list[dict]:
    """
    Split a single metadata row into segment-level entries.

    Args:
        row: A row from the metadata DataFrame
        config: Segmentation configuration

    Returns:
        List of dictionaries, one per segment
    """
    segments = []
    file_length = row["length"]
    filepath = Path(row["filepath"])
    stem = filepath.stem

    # Calculate number of segments
    num_segments = int(np.ceil(file_length / config.segment_duration))

    # Get events and clusters
    events = row.get("detected_events", np.array([]))
    event_clusters = row.get("event_cluster", np.array([]))

    # Handle None or empty arrays
    if events is None or (isinstance(events, np.ndarray) and len(events) == 0):
        events = np.array([])
    if event_clusters is None or (isinstance(event_clusters, np.ndarray) and len(event_clusters) == 0):
        event_clusters = np.array([])

    for i in range(num_segments):
        seg_start = i * config.segment_duration
        seg_end = seg_start + config.segment_duration

        # Generate segment filename
        segment_filename = f"{stem}_{int(seg_start):03d}_{int(seg_end):03d}.wav"

        # Find overlapping events
        overlapping_event_indices = get_events_in_segment(events, seg_start, seg_end, config)

        # Determine labels for this segment
        if len(overlapping_event_indices) > 0:
            # Use combined multilabel (primary + secondary) for segments with events
            label = get_combined_labels(row, config.label_separator)
            has_event = True

            # Get the overlapping event times (relative to segment)
            segment_events = []
            segment_clusters = []
            for idx in overlapping_event_indices:
                event = events[idx]
                # Convert to segment-relative times
                rel_onset = max(0, event[0] - seg_start)
                rel_offset = min(config.segment_duration, event[1] - seg_start)
                segment_events.append([rel_onset, rel_offset])
                if len(event_clusters) > idx:
                    segment_clusters.append(event_clusters[idx])
        else:
            # No events in this segment
            label = config.no_event_label
            has_event = False
            segment_events = []
            segment_clusters = []

        # Build segment metadata
        segment_data = {
            "filename": segment_filename,
            "original_filepath": row["filepath"],
            "segment_start": seg_start,
            "segment_end": seg_end,
            "label": label,
            "has_event": has_event,
            "segment_events": segment_events,
            "segment_event_clusters": segment_clusters,
            # Propagate metadata
            "ebird_code_multilabel": row.get("ebird_code_multilabel"),
            "ebird_code_secondary": row.get("ebird_code_secondary"),
            "lat": row.get("lat"),
            "long": row.get("long"),
            "call_type": row.get("call_type"),
            "sex": row.get("sex"),
            "license": row.get("license"),
            "local_time": row.get("local_time"),
            "quality": row.get("quality"),
            "microphone": row.get("microphone"),
            "source": row.get("source"),
            "recordist": row.get("recordist"),
            "order": row.get("order"),
            "species_group": row.get("species_group"),
            "genus": row.get("genus"),
        }

        segments.append(segment_data)

    return segments


def split_metadata_to_segments(
    df: pd.DataFrame,
    config: Optional[SegmentConfig] = None
) -> pd.DataFrame:
    """
    Split entire metadata DataFrame into segment-level entries.

    Args:
        df: Input DataFrame with file-level metadata
        config: Segmentation configuration (uses defaults if None)

    Returns:
        DataFrame with one row per segment
    """
    if config is None:
        config = SegmentConfig()

    all_segments = []

    for idx, row in df.iterrows():
        segments = split_row_into_segments(row, config)
        all_segments.extend(segments)

    return pd.DataFrame(all_segments)


def filter_by_existing_audio(
    segment_df: pd.DataFrame,
    audio_dir: Path,
    audio_extensions: set[str] = {".wav", ".ogg", ".flac"}
) -> pd.DataFrame:
    """
    Filter segment metadata to only include rows where audio files exist.

    Args:
        segment_df: DataFrame with segment-level metadata
        audio_dir: Directory containing audio segment files
        audio_extensions: Set of valid audio file extensions

    Returns:
        Filtered DataFrame with only rows that have corresponding audio files
    """
    audio_dir = Path(audio_dir)
    existing_files = set()

    for ext in audio_extensions:
        for f in audio_dir.glob(f"*{ext}"):
            existing_files.add(f.name)

    # Also check without extension (match by stem)
    existing_stems = {Path(f).stem for f in existing_files}

    mask = segment_df["filename"].apply(
        lambda x: x in existing_files or Path(x).stem in existing_stems
    )

    filtered_df = segment_df[mask].copy()
    return filtered_df


def create_labels_csv(
    segment_df: pd.DataFrame,
    validation_fraction: float = 0.1,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Create a labels.csv file in the expected format.

    Format: filename, label, validation

    Args:
        segment_df: DataFrame with segment-level metadata
        validation_fraction: Fraction of samples to use for validation
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: filename, label, validation
    """
    np.random.seed(random_seed)

    labels_df = pd.DataFrame({
        "filename": segment_df["filename"],
        "label": segment_df["label"],
        "validation": np.random.random(len(segment_df)) < validation_fraction
    })

    return labels_df


def main():
    """Example usage of the segment splitter."""
    import argparse

    parser = argparse.ArgumentParser(description="Split audio metadata into segments")
    parser.add_argument("input", type=str, help="Input parquet file")
    parser.add_argument("--output-metadata", type=str, default="metadata_segments.parquet",
                        help="Output metadata parquet file")
    parser.add_argument("--output-labels", type=str, default="labels.csv",
                        help="Output labels CSV file")
    parser.add_argument("--segment-duration", type=float, default=5.0,
                        help="Segment duration in seconds")
    parser.add_argument("--min-overlap", type=float, default=0.0,
                        help="Minimum overlap for event inclusion")
    parser.add_argument("--validation-fraction", type=float, default=0.1,
                        help="Fraction of samples for validation")

    args = parser.parse_args()

    # Load input data
    print(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df)} rows")

    # Configure segmentation
    config = SegmentConfig(
        segment_duration=args.segment_duration,
        min_overlap=args.min_overlap
    )

    # Split into segments
    print(f"Splitting into {config.segment_duration}s segments...")
    segment_df = split_metadata_to_segments(df, config)
    print(f"  Created {len(segment_df)} segments")

    # Save metadata
    print(f"Saving metadata to {args.output_metadata}...")
    segment_df.to_parquet(args.output_metadata, index=False)

    # Create and save labels
    print(f"Creating labels with {args.validation_fraction*100}% validation split...")
    labels_df = create_labels_csv(segment_df, args.validation_fraction)
    labels_df.to_csv(args.output_labels, index=False)
    print(f"Saved labels to {args.output_labels}")

    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Total segments: {len(segment_df)}")
    print(f"Segments with events: {segment_df['has_event'].sum()}")
    print(f"Segments without events: {(~segment_df['has_event']).sum()}")
    print(f"\nLabel distribution:")
    print(segment_df["label"].value_counts().head(20))


if __name__ == "__main__":
    main()
