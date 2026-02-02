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
from pathlib import Path
import numpy as np
import shutil
import torchaudio as ta

# Re-export Embedder for type hints
__all__ = ['initialise', 'generate_embeddings', 'Embedder']

def initialise(model_name: str = 'birdnet') -> Embedder:
    # Download model checkpoint if needed
    bacpipe.ensure_models_exist(
        model_base_path=Path(bacpipe.settings.model_base_path),
        model_names=[model_name]
    )

    # Initialize the embedder with model
    # Pass all settings as kwargs (model_utils_base_path is set internally from package)
    embedder = Embedder(
        model_name=model_name,
        **vars(bacpipe.settings),
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
) -> dict:
    """
    Generate segments and embeddings from multilength audio files.

    Audio is automatically segmented based on model requirements for example BirdNet: 3-second windows at 48kHz (144000 samples)

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

    Returns
    -------
    dict
        Dictionary mapping audio filenames to their embedding arrays
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    segments_dir = Path(segments_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)


    # Process each audio file
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.flac")) + list(audio_dir.glob("*.ogg"))
    print(f"\nFound {len(audio_files)} audio files")

    embeddings_dict = {}
    total = len(audio_files)

    for i, audio_file in enumerate(audio_files, 1):
        print(f"\rProcessing {i}/{total}: {audio_file.name[:30]:<30}", end="", flush=True)

        # Step 1: Load and resample audio
        audio = embedder.model.load_and_resample(audio_file)
        audio = audio.to(embedder.model.device)

        # Step 2: Window audio into segments
        frames = embedder.model.window_audio(audio)
        num_segments = frames.shape[0]
        segment_duration = embedder.model.segment_length / embedder.model.sr

        # print(f"  Segments: {num_segments} x {segment_duration:.1f}s")

        # Step 3: Save each segment as individual audio file
        segment_files = []
        for i, segment in enumerate(frames):
            start_time = int(segment_duration * i)
            end_time = int(segment_duration * (i + 1))
            segment_filename = f"{audio_file.stem}_{start_time:03d}_{end_time:03d}.wav"
            segment_path = segments_dir / segment_filename
            # Save as mono wav at model's sample rate
            ta.save(segment_path, segment.unsqueeze(0).cpu(), embedder.model.sr)
            segment_files.append(segment_path)
        # print(f"  Saved {num_segments} segments to {segments_dir}/")

        # Step 4: Compute embeddings on individual segments and save each separately
        segment_embeddings = []
        for segment_path in segment_files:
            # Load segment, preprocess, and get embedding
            preprocessed = embedder.prepare_audio(segment_path)
            embedding = embedder.get_embeddings_for_audio(preprocessed)

            # Ensure 1D shape for single segment embedding
            if embedding.ndim > 1:
                embedding = embedding.squeeze()

            # Save individual embedding file (matching segment filename)
            embedding_file = output_dir / f"{segment_path.stem}_{model_name}.npy"
            np.save(embedding_file, embedding)
            segment_embeddings.append((segment_path.name, embedding))

        # print(f"  Saved {len(segment_embeddings)} embeddings to {output_dir}/")
        # print(f"  Embedding dim: {segment_embeddings[0][1].shape[0]}")

        embeddings_dict[audio_file.name] = segment_embeddings
    return embeddings_dict
