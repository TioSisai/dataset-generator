## Planning

Data Format (Recommended):
```
dataset_name/
  data/
      filename_1.wav/jpg/...  (segment)
      ...
  embeddings/
      model_A/
          filename_1_{model_A}.npy
          ...
      model_B/
          ...
  labels/
      labels.csv
      metadata.csv
```

Data (audio) format: audio segment (e.g. HSN_001_20150708_061805_000_005.ogg) corresponding to the length of the model input (5 seconds).

Embeddings format: (e.g. HSN_001_20150708_061805_000_005_perch_v2.npy), 1536-dimensional embedding, shape (1536,1)

**Example (labels.csv)**
```
columns {filename; label; validation} <---> 
row 1 : HSN_001_20150708_061805_000_005.ogg; gcrfin; False
row 2 : HSN_001_20150708_061805_095_100.ogg; whcspa,amepip; True
```
*Each row is an audio segment*

**Example (metadata.csv)** --> HSN_metadata_test_5s.csv (can be the exact same content)
*Important that rows match*

We will use the existing test sets for evaluation of participants. The format will be the same.


Demo format (*this is the old format, to be updated*):
- Audio segments 
- Embeddings (pre-generated)
- Evaluation (labels csv) - It currently contains redundant directories