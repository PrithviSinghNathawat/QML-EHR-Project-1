# Data Source

## Dataset

UCI Heart Disease dataset (all 4 processed site files: Cleveland, Hungarian,
Switzerland, VA — 920 records total). See `docs/decisions.md`, D-001 and
D-018, for why all 4 sites are used and how each file was processed.

## Download

- **URL:** https://archive.ics.uci.edu/static/public/45/heart+disease.zip
- **Retrieved:** 2026-08-18
- **Files kept in this repo:** `data/raw/processed.cleveland.data`,
  `data/raw/processed.hungarian.data`, `data/raw/processed.switzerland.data`,
  `data/raw/processed.va.data`, `data/raw/heart-disease.names` (column
  documentation). Other files in the original archive (unprocessed
  variants, cost data) were not kept — not used anywhere in this project.

## Citation

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart
Disease [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C52P4X

**Origin paper** (for provenance, not the dataset citation itself):
Detrano, R., Jánosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu,
S., Guppy, K., Lee, S., & Froelicher, V. (1989). International application
of a new probability algorithm for the diagnosis of coronary artery
disease. *American Journal of Cardiology*.

## License

CC BY 4.0. Attribution is a license condition and must appear in the paper
(see `paper/`) as well as here.

## Known caveats (see `docs/decisions.md` for full detail)

- The "Switzerland" file combines two institutions — University Hospital
  Zurich and University Hospital Basel. It is not a single site, despite
  being treated as one client in the partitioning scheme (D-018).
- `chol = 0` is treated as a missing-value code in this project's
  preprocessing (D-006), but this is **not** documented by the dataset
  creators — it is an inference from physiological implausibility, common
  in the literature but not an official fact about the dataset.
