"""
Streamlined bacpipe embedding generation test.
Uses BirdNet model to generate embeddings from audio files.
Audio is automatically segmented into 3-second windows by the model.
"""
# Suppress warnings before any imports
import os
import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF C++ logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN messages

warnings.filterwarnings('ignore', category=UserWarning)  # Panel/bokeh warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*IProgress.*')  # tqdm jupyter warning

# Suppress TensorFlow Python warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('bacpipe').setLevel(logging.ERROR)

import bacpipe
from bacpipe.generate_embeddings import Embedder
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

logger = logging.getLogger(__name__)

# Re-export Embedder for type hints
__all__ = ['initialise', 'generate_embeddings', 'Embedder']


def initialise(
    model_name: str = 'birdnet',
    run_pretrained_classifier: bool = False,
) -> Embedder:
    # Download model checkpoint if needed
    bacpipe.ensure_models_exist(
        model_base_path=Path(bacpipe.settings.model_base_path),
        model_names=[model_name]
    )

    # Initialize the embedder with model
    # Pass all settings as kwargs (model_utils_base_path is set internally from package)
    tmp_dict = vars(bacpipe.settings) | {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if "perch_v2" in model_name:
        tmp_dict["run_pretrained_classifier"] = run_pretrained_classifier
    embedder = Embedder(
        model_name=model_name,
        **tmp_dict,
    )

    print(f"Model: {model_name}")
    print(f"Sample rate: {embedder.model.sr} Hz")
    print(f"Segment length: {embedder.model.segment_length} samples "
          f"({embedder.model.segment_length / embedder.model.sr:.1f}s)")
    return embedder


def generate_embeddings(
    audio_dir: str | Path,
    embedder: Embedder,
    model_name: str = "birdnet",
    output_dir: str | Path = "embeddings_output",
    segments_dir: str | Path = "test_segments",
    save_segments: bool = True,
) -> list[str]:
    """
    Generate segments and embeddings from multilength audio files.

    Audio is automatically segmented based on model requirements for example
    BirdNet: 3-second windows at 48kHz (144000 samples). Saved audio segments
    preserve each source file's original sampling rate.

    Parameters
    ----------
    audio_dir : str or Path
        Directory containing audio files (.wav, .mp3, .flac, .ogg etc.)
    model_name : str
        Model to use for embeddings (default: "birdnet")
    output_dir : str or Path
        Directory to save embeddings
    segments_dir : str or Path
        Directory to save audio segments
    save_segments : bool
        Whether to save segmented audio files to segments_dir

    Returns
    -------
    list[str]
        List of processed audio filenames.
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    segments_dir = Path(segments_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_segments:
        segments_dir.mkdir(parents=True, exist_ok=True)

    # Process each audio file
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3")) + \
        list(audio_dir.glob("*.flac")) + list(audio_dir.glob("*.ogg"))
    print(f"\nFound {len(audio_files)} audio files")

    total = len(audio_files)
    processed_files: list[str] = []

    if total == 0:
        return processed_files
    log_interval = max(1, total // 100)  # Log every 1% of files
    for audio_idx, audio_file in enumerate(audio_files):
        if (audio_idx + 1) % log_interval == 0 or audio_idx == total - 1:
            print(f"Processing file {audio_idx + 1}/{total}: {audio_file.name}")
        # Use bacpipe's official per-file pipeline.
        # This route applies model-specific preparation and memory handling.
        try:
            batched_embeddings = embedder.get_embeddings_from_model(audio_file)
        except Exception as exc:
            logger.warning("Skipping %s due to embedding error: %s", audio_file, exc)
            continue

        segment_duration = float(embedder.model.segment_length) / float(embedder.model.sr)
        waveform = None
        original_sr = None
        samples_per_segment = None

        if save_segments:
            try:
                waveform, original_sr = sf.read(audio_file)
            except (RuntimeError, OSError, ValueError) as exc:
                logger.warning("Skipping %s due to audio read error: %s", audio_file, exc)
                continue

            samples_per_segment = max(
                1, int(round(segment_duration * float(original_sr)))
            )

        if batched_embeddings.ndim == 1:
            batched_embeddings = np.expand_dims(batched_embeddings, axis=0)

        # Step 4: Save per-segment embeddings with time-based filenames
        for segment_index, embedding in enumerate(batched_embeddings):
            start_time = int(segment_duration * segment_index)
            end_time = int(segment_duration * (segment_index + 1))
            segment_filename = f"{audio_file.stem}_{start_time:04d}_{end_time:04d}.wav"
            # Save individual embedding file (matching segment filename)
            embedding_file = output_dir / f"{Path(segment_filename).stem}_{model_name}.npy"
            np.save(embedding_file, embedding)

            if save_segments:
                start_sample = segment_index * samples_per_segment
                end_sample = start_sample + samples_per_segment
                segment_waveform = waveform[start_sample:end_sample]

                if segment_waveform.shape[0] < samples_per_segment:
                    if np.ndim(waveform) == 1:
                        pad_width = (0, samples_per_segment - segment_waveform.shape[0])
                    else:
                        pad_width = (
                            (0, samples_per_segment - segment_waveform.shape[0]),
                            (0, 0),
                        )
                    segment_waveform = np.pad(
                        segment_waveform,
                        pad_width,
                        mode="constant",
                        constant_values=0,
                    )

                segment_file = segments_dir / segment_filename
                sf.write(segment_file, segment_waveform, int(original_sr))

        processed_files.append(audio_file.name)

    return processed_files
