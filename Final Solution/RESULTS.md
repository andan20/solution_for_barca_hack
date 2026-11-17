# Final Solution - external validation results

The released solution combines a player-only BiGRU-TCN D/LIVE ensemble with a frozen SkillCorner encoder and residual BiGRU for H/A. All performance figures below are from external validation matches. These splits were used for checkpoint and decoder selection, so the results describe model development rather than an independent final test.

## Released components

| Dataset and validation protocol | Metric | Value |
|---|---|---:|
| Sportec: two matches; equal mean over matches and clean/noisy views | D-interval F1 | 0.83694 |
| SkillCorner: four matches; original retained views | H/A-only clipped spellF1 | 0.87229 |
| Same SkillCorner validation | H/A-only macroF1 | 0.95882 |

Sportec validation matches: **J03WR9, J03WMX**. SkillCorner validation matches: **1925299, 2016236, 2011166, 2013725**.

D-interval F1 matches stoppages. H/A-only spellF1 matches possession intervals within provider-labelled LIVE spans; unknown spans break intervals. The component scores are not interchangeable and are not a combined H/A/D score. Normalisation and duration statistics are fitted on training matches.

## H/A head comparison

All variants reuse the frozen encoder and the same four validation matches. Every listed result uses switch cost 6.0. The residual BiGRU is selected by H/A-only spellF1.

| Model | H/A-only spellF1 | H/A-only macroF1 |
|---|---:|---:|
| Frozen encoder + decoder | 0.84479 | 0.94532 |
| Encoder + linear residual head | 0.84279 | 0.94534 |
| Encoder + residual TCN | 0.86092 | 0.95164 |
| Encoder + residual BiGRU (released) | 0.87229 | 0.95882 |

## Released H/A model by validation match

| Match | H/A-only spellF1 |
|---|---:|
| 1925299 | 0.89130 |
| 2016236 | 0.84990 |
| 2011166 | 0.87847 |
| 2013725 | 0.87017 |

## H/A class metrics on the same validation split

| Class | Precision | Recall | F1 | Labelled frames |
|---|---:|---:|---:|---:|
| H | 0.96499 | 0.95412 | 0.95952 | 50260 |
| A | 0.95255 | 0.96377 | 0.95813 | 48030 |

## Scope and reproduction

External validation sets were reused during development; clubs can recur across SkillCorner splits. Tracking and annotation conventions differ by provider. These figures do not establish end-to-end generalisation to a new match. Source revisions, exact splits and training amounts are documented in [TRAINING_DATA.md](TRAINING_DATA.md) and [DATA_SOURCES.json](DATA_SOURCES.json).

The inference package has already reproduced the frozen submission byte-for-byte from the original tactical ATD TXT/XML with caches disabled. The checksum enforced by `config.json` is:

```text
92a836b63b25f7a3f7576bcbbcb8c072d469c9c3a344bfdf559c25129c6d421b
```

This document version reuses the same trained weights, preprocessing, decoders and prediction file. The package supports exact inference reproduction; it does not contain a one-command retraining pipeline.
