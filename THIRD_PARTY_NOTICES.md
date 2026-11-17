# Third-party notices

## Metrica Sports competition code

`vendor/possession_metrics.py` is an unmodified copy of the official scorer from the More than a Hack 2027 repository. It is covered by the original MIT license in `vendor/METRICA_LICENSE`. Copyright (c) 2026–2027 Metrica Sports. The original repository is https://github.com/metrica-sports/more-than-a-hack-2027.

The ATD metadata interpretation and timing conventions follow the organiser's provided code and documentation. The source code license does not grant redistribution rights to the competition videos, tracking, homographies or Smart Tagging. Those files are not included.

## SkillCorner Open Data

Training used SkillCorner's open tracking data and automatic possession/phase outputs, distributed in its joint initiative with PySport. Credit: **SkillCorner** — https://skillcorner.com; https://github.com/SkillCorner/opendata. Its MIT copyright notice is preserved in `vendor/SKILLCORNER_LICENSE`.

## Sportec / IDSSE

Training used the integrated dataset of synchronized spatiotemporal and event data in elite soccer, collected by Sportec Solutions, owned by the DFL and distributed through PySport. Source: https://huggingface.co/datasets/pysport/idsse-data. Dataset DOI: https://doi.org/10.6084/m9.figshare.28196177. Paper: https://doi.org/10.1038/s41597-025-04505-y. License stated by the distribution: **Creative Commons Attribution 4.0**, https://creativecommons.org/licenses/by/4.0/.

Our transformations comprise resampling, masking, feature extraction and neural-model training; they are not corrections supplied or endorsed by the data creators. The raw datasets are not redistributed.
