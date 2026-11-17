# Training data and model provenance

The final neural models were trained on other matches only. The competition tracking is an inference input, not supervised training data. We added no manual labels. The provider-origin targets below are used as supplied or filtered for consistency; their upstream annotation process is not reclassified as our own manual annotation.

## Sportec / IDSSE — D/LIVE

Source: [PySport mirror](https://huggingface.co/datasets/pysport/idsse-data), [dataset DOI](https://doi.org/10.6084/m9.figshare.28196177), [dataset paper](https://doi.org/10.1038/s41597-025-04505-y). Tracking was collected by Sportec Solutions; the data owner is the DFL. The dataset card lists **CC BY 4.0**. We acknowledge the data creators and distribution by PySport. Raw data is not redistributed here.

Pinned revision: `a715a38dfbaf5f58e431727c2b78d174101a703c`. `DATA_SOURCES.json` identifies the exact XML files and their SHA-256 hashes; use these pinned files rather than assuming an upstream listing has remained unchanged.

Targets: raw `BallStatus=0` maps to D, `BallStatus=1` to LIVE; unknown status is excluded. Positions are resampled to 10 Hz. Provider ball status, ball possession and event annotations do not enter the D features. Ball geometry is also excluded from D.

| Split used by the final D models | Match IDs | Labelled frames at 10 Hz |
|---|---|---:|
| Training | J03WOH, J03WOY, J03WPY, J03WQQ | 227,325 |
| Validation | J03WR9, J03WMX | 117,112 |
| Test | J03WN1 | 56,624 |

The training labels contain 101,700 D and 125,625 LIVE frames. Counts above are unique retained frames, not augmented copies or repeated training windows. Normalisation and duration hazards are fitted on training matches. Clean/noisy validation views use the same held-out matches. J03WMX was an evaluation match in an earlier study and became validation in the final split; it is not claimed as an untouched test. The remaining test match was also reused in development.

## SkillCorner — H/A

Source: [SkillCorner Open Data](https://github.com/SkillCorner/opendata), an initiative with [PySport](https://pysport.org/). Data and automatic possession/phase targets are provided by **SkillCorner**. See its [documentation](https://skillcorner.github.io/opendata/) and [license](https://github.com/SkillCorner/opendata/blob/master/LICENSE). We retain attribution to SkillCorner. No source match records are included in this repository.

Pinned revision: `4340d274572876239c154c90bc507a9b3250a656`. The experiment used twenty available matches at this revision; a current README's headline match count is not the training manifest.

| Split | Match IDs | Retained labelled frames at 10 Hz |
|---|---|---:|
| Training | 1899585, 2007721, 1996436, 2007448, 1953632, 2006229, 2015213, 1996435, 2010085, 1986691, 2006363, 1874553 | 305,354 |
| Validation | 1925299, 2016236, 2011166, 2013725 | 98,290 |
| Test | 1959846, 2017461, 1886347, 1927964 | 103,042 |

Inputs retain detected player/ball observations and mask extrapolated coordinates. Player IDs are used to determine team membership, not as model features. Targets are the possession group where it agrees with the live phase's team in possession; ambiguous and phase-boundary frames are excluded. Missing or conflicting labels remain unknown, not D. The training subset contains 150,657 H and 154,697 A targets. Unlabelled input frames can provide context, but do not contribute supervised targets.

The encoder used contiguous ball occlusion, player dropout, mirrored coordinates and team swaps. The final residual head reused the frozen encoder and a train-fitted embedding scaler. This head was trained on the original retained SkillCorner views; the later noisy-ball/geometry-cleanup experiment did not replace it. All splits are by match, not by neighbouring frames; teams may appear in more than one split.

## Frozen component ledger

| Component | Selection | Final artifact |
|---|---|---|
| Player BiGRU192, two bidirectional layers | Sportec validation; epoch 8 | `dead_bigru.pt` |
| Player TCN192, eight residual blocks | Sportec validation; epoch 16 | `dead_tcn.pt` |
| Equal-weight D ensemble + duration model | Sportec validation; duration weight 0.25 | `duration.json` |
| SkillCorner set encoder + GRU48 | External validation; epoch 3 of 8 | `possession_encoder.pt` |
| Residual H/A BiGRU64 | External validation; epoch 19 | `possession_bigru.pt` |
| H/A decoder | SkillCorner validation; switch penalty 6.0 | `config.json` |

The weight manifest records both the exported SHA-256 and its original checkpoint SHA-256. D exports remove unused training-only boundary heads; the state classifier is unchanged. Preprocessing arrays are copied unchanged. Duration hazards are frozen training-derived statistics. The final inference reproduction check confirms that this export produces the original CSV exactly.

This repository ships trained models and sufficient code to regenerate the submission, not the entire research/training workspace. Training targets, raw external datasets, optimiser checkpoints and prediction caches are not shipped.
