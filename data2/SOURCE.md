# Data Source — Second Dataset

## Dataset

Diabetes 130-US Hospitals for Years 1999-2008 (UCI). 101,766 encounters, 50 columns.

## Download

- **URL:** https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip
- **Retrieved:** 2026-08-28
- **Files kept in this repo:** `data2/raw/diabetic_data.csv` (the encounter-level
  table), `data2/raw/IDS_mapping.csv` (integer-code lookup for
  `admission_type_id`, `discharge_disposition_id`, `admission_source_id`).

## Citation

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals
for Years 1999-2008 [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5230J

**Origin paper** (for provenance, not the dataset citation itself):
Strack, B., DeShazo, J., Gennings, C., Olmo, J., Ventura, S., Cios, K., & Clore, J.
(2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of
70,000 Clinical Database Patient Records. *BioMed Research International*, 2014,
Article ID 781670.

## License

CC BY 4.0. Attribution is a license condition and must appear in the paper
(see `paper/`) as well as here.

## Known caveats

- **No per-record hospital identifier.** The source paper describes the data as
  drawn from "130 US hospitals and integrated delivery networks," but the released
  file has no hospital/facility ID column — confirmed by inspecting all 50 columns
  directly, not assumed. "130" describes the source data warehouse, not an
  available partition key. See `docs/dataset2_characterization.md` and P-009 for
  the full consequence for client construction.
- **Repeat patients.** 101,766 encounters come from only 71,518 unique
  `patient_nbr` values — about 30,248 encounters (29.7%) are repeat visits from a
  patient already in the dataset. A naive random train/test split can leak the
  same patient across splits. Not present in the heart-disease dataset (one row
  per patient there).
- **Missing values are coded as the literal string `"?"`**, not blank/NaN — must
  read with `na_values="?"` or every missingness computation is silently wrong.
- **15 of 23 medication columns are >99% a single value** (2 of them —
  `examide`, `citoglipton` — are 100% one value, i.e. zero variance). These pass
  a pure missingness filter but carry effectively no signal; see
  `docs/dataset2_characterization.md`.
