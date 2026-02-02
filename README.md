<h1>
  <img src="baseAL_logo.png" alt="Logo" width="35" height="30">
  BaseAL - Dataset Generator
</h1>

>This repository is for dataset generation compatible with [BaseAL](https://github.com/BenMcEwen1/BaseAL). Audio segments and embeddings are generated using [bacpipe](https://github.com/bioacoustic-ai/bacpipe) Currently only audio pre-trained models are supported. 

## Setup
Install dependencies using `uv sync`

**You will need linux (e.g. WSL) to run perch_v2*

For an overview of the data format and existing pipeline see `generator.ipynb`.

### Run on server

Example usage:
```
./run.sh \
    --model birdnet \
    --audio-path /data/HSN/HSN_train_shard_0001 \
    --metadata-path /data/HSN/HSN_metadata_train.parquet \
    --output-path /output/HSN_BASEAL
```

## Note
You may want to setup datasets (huggingface) for BirdSet. I encountered compatibility issues setting this up with bacpipe so I manually installed the data. You're welcome to have another go at it.