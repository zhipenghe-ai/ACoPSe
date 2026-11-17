# Datasets and feature preparation

## Upstream benchmarks

- **MIRFlickr-25K.** Leiden University's LIACS Media Lab provides the 25,000-image collection with Flickr tags and EXIF metadata, plus a separate annotation archive. [LIACS download page](https://press.liacs.nl/mirflickr/mirdownload.html)
- **NUS-WIDE.** The National University of Singapore's Lab for Media Search introduced the collection of 269,648 Flickr images with tags and annotations for 81 concepts. The original dataset page was unreachable during release validation, and availability of the raw files remains unverified. [NUS dataset page](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) · [Original dataset paper on ResearchGate](https://www.researchgate.net/publication/221368756_NUS-WIDE_A_real-world_web_image_database_from_National_University_of_Singapore) · [ACM citation](https://doi.org/10.1145/1646396.1646452)

These pages document the upstream collections. Access to the exact ACoPSe feature matrices through those pages is unverified. The ACoPSe paper's data availability statement says that data are available on request. [ACoPSe paper](https://doi.org/10.1016/j.inffus.2025.104078)

## Experimental subsets

Section 4.1 and Table 2 of the paper report the following protocol. For each benchmark, the training set is sampled from the retrieval set. The query set is drawn separately from the same filtered collection. [ACoPSe paper](https://doi.org/10.1016/j.inffus.2025.104078)

| Dataset | Filtered pairs | Train | Query | Retrieval | Label columns |
| --- | ---: | ---: | ---: | ---: | ---: |
| MIRFlickr-25K | 20,015 | 5,000 | 2,000 | 18,015 | 24 |
| NUS-WIDE | 186,577 | 5,000 | 2,000 | 184,577 | 10 |

- For MIRFlickr-25K, the paper describes excluding instances with fewer than 20 tags. LIACS separately documents 1,386 distinct tags that occur in 20 or more images. These are different counting rules. The original feature files do not record how the per-instance filter was implemented. [ACoPSe paper, §4.1](https://doi.org/10.1016/j.inffus.2025.104078) · [LIACS Flickr tag documentation](https://press.liacs.nl/mirflickr/)
- For NUS-WIDE, the paper restricts the benchmark to the ten most frequent concepts, draws 2,000 queries at random, and randomly samples 5,000 training pairs from the remaining gallery. [ACoPSe paper, §4.1](https://doi.org/10.1016/j.inffus.2025.104078)

The supplied matrices match the reported row counts and label dimensions. In both benchmarks, the original training text-and-label rows can be matched within the retrieval split, including repeated rows. No full image-feature, text-feature, and label row matches exactly across splits. The MAT files lack source image IDs, so row comparisons leave exact raw-sample membership and split identity unverified.

## Feature representation

The paper describes 4,096-dimensional image descriptors from AlexNet with its final fully connected layer omitted. Text uses bag-of-words vectors, with 1,386 columns for MIRFlickr-25K and 1,000 for NUS-WIDE. The inspected matrices store binary text features and multi-hot labels. The release starts from these numeric matrices. [ACoPSe paper, §§4.1 and 4.3](https://doi.org/10.1016/j.inffus.2025.104078) · [MAT loaders](../load_data.py)

The research snapshot lacks the feature extraction scripts, the raw-image split manifests, and the original tag vocabulary and label-column ordering. The exact AlexNet weights, image transforms, and tag preprocessing settings remain unverified. Reconstructing the reported matrices from raw assets therefore requires additional provenance even when the new matrices have the expected dimensions.

## Use existing MAT files

Set `--DATA_DIR` to the parent directory that contains `MIRFlickr/` and `NUS-WIDE/`. Each launcher needs the three files for its own dataset, including the training file during checkpoint evaluation. The default parent directory is `./datasets`, relative to the launch directory.

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

The loaders read MATLAB matrices through `scipy.io.loadmat`. Keep samples in rows and align image features, text features, and labels across the three arrays in each file. Keep feature and label column ordering consistent across splits.

| File suffix | Image key | Text key | Label key |
| --- | --- | --- | --- |
| `_train.mat` | `I_tr` | `T_tr` | `L_tr` |
| `_query.mat` | `I_te` | `T_te` | `L_te` |
| `_database.mat` | `I_db` | `T_db` | `L_db` |

| Dataset | Image width | Text width | Label width |
| --- | ---: | ---: | ---: |
| MIRFlickr-25K | 4,096 | 1,386 | 24 |
| NUS-WIDE | 4,096 | 1,000 | 10 |

From the repository root, pass the location of the feature matrices to the appropriate launcher. [Checkpoint evaluation](evaluation.md)

```text
python main_mir.py --DATA_DIR ./datasets
python main_nus.py --DATA_DIR ./datasets
```
