# Checkpoint evaluation

## Inputs

Run the commands from the repository root. Set `--DATA_DIR` to the parent of the dataset folders shown below. The launchers read the existing pre-extracted feature and label matrices. They load all three splits during initialization, including the training split in evaluation mode.

```text
datasets/
├── MIRFlickr/
│   ├── mir_train.mat
│   ├── mir_query.mat
│   └── mir_database.mat
└── NUS-WIDE/
    ├── nus_train.mat
    ├── nus_query.mat
    └── nus_database.mat
```

Each split supplies image features, text features, and multi-label indicators in aligned rows. The variable suffixes are `_tr`, `_te`, and `_db` for train, query, and database, respectively. For example, `I_te`, `T_te`, and `L_te` are the query arrays.

| Dataset | Train / query / database rows | Image / text / label dimensions |
| --- | --- | --- |
| MIRFlickr-25K | 5,000 / 2,000 / 18,015 | 4,096 / 1,386 / 24 |
| NUS-WIDE | 5,000 / 2,000 / 184,577 | 4,096 / 1,000 / 10 |

## Run evaluation

Use `--TRAIN False` to evaluate a checkpoint. Set `--MODEL_DIR` to its containing directory. The filename follows `<DATASET>_<HASH_BIT>bits.pth`, where `DATASET` is `MIRFlickr` or `NUSWIDE`. The file contains the `ImgNet` and `TxtNet` state dictionaries. Change `--HASH_BIT` to match a checkpoint with a different code length.

```text
python main_mir.py --TRAIN False --DATA_DIR ./datasets --MODEL_DIR ./checkpoints --HASH_BIT 16 --LOG_DIR ./logs
python main_nus.py --TRAIN False --DATA_DIR ./datasets --MODEL_DIR ./checkpoints --HASH_BIT 16 --LOG_DIR ./logs
```

The launchers write `<DATASET>_<HASH_BIT>.txt` under `--LOG_DIR`, which defaults to `./logs`. Relative directories are resolved from the working directory. The CUDA device defaults to `--GPU_ID 0`.

## MAP@5000

- **Image-to-text (I→T)** ranks database text codes for each image query. **Text-to-image (T→I)** ranks database image codes for each text query.
- The evaluator L2-normalizes the pre-extracted features, computes each modality's code with its hash network, and applies `torch.sign`. With ±1 codes, it ranks by ascending Hamming distance, computed as `(code length − query code · database code) / 2`.
- A result is relevant if its label vector shares at least one positive label with the query. To compute AP, the evaluator sums precision at the relevant ranks within the first 5,000 results and divides by the number of relevant hits within that cutoff, as in Eq. 12 of the paper. A query with no relevant hit contributes zero. MAP is the mean AP over all queries.
- Equal-distance items follow NumPy's default `argsort` order. The evaluator applies no additional tie-break rule.

## Validation environment

GPU smoke tests used Windows, Python 3.7.16, and PyTorch `1.13.1+cu117` on an NVIDIA GeForce RTX 4090. The validation covered a one-epoch run on the supplied MIRFlickr features and a small synthetic NUS-WIDE-format run. The MIRFlickr preset uses batch size 512, while Section 4.3 of the paper specifies 1,024. Full NUS-WIDE benchmarking and reproduction of the paper's scores remain unverified. [Environment setup and troubleshooting](environment.md)
