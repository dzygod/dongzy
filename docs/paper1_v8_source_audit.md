# Paper 1 source audit and analysis version

This is a historical v8 audit. The current Structures manuscript uses v9 for the primary analysis and v10_core for the extension; see [paper1_current_reproduction.md](paper1_current_reproduction.md). The v8 manuscript version used `uhpc_nc_bond_literature_data_v8.csv`,
derived from the author's v7 literature table. The integrated table is
available from the corresponding author on reasonable request; the repository's
`literature_sample.csv` is synthetic and cannot reproduce manuscript results.

Two predictor corrections were made without changing bond-strength outcomes:

1. For 16 records from Sun et al. (2024), DOI
   `10.1617/s11527-024-02507-7`, values previously entered in `roughness_mm`
   were the paper's γ, interface slip at failure. They are now missing
   roughness values. The source's true image-based Ra/Std measurements were
   not available as row-level numerical values in the extraction table.
2. For 25 records from Valikhani et al. (2020), DOI
   `10.1016/j.conbuildmat.2019.117753`, the source states that testing was
   performed ten months after UHPC casting. The v7 table coded 28 days, which
   was the reference age for compressive strength rather than the bond test.
   v8 codes approximately 300 days, using an explicit 30-day/month conversion.

Recreate the corrected CSV and rerun the grouped analysis with:

```bash
python scripts/paper1_correct_source_fields_v8.py \
  --input /path/to/uhpc_nc_bond_literature_data_v7.csv \
  --output /path/to/uhpc_nc_bond_literature_data_v8.csv
python scripts/paper1_group_validation_v4.py \
  --csv /path/to/uhpc_nc_bond_literature_data_v8.csv \
  --out outputs/paper1_v8
```

The manuscript reports 440 prescreened records from 14 studies; 259 concern
shear strength and 181 concern tensile strength. The corrected shear XGBoost
random five-fold pooled OOF R² is 0.815, and grouped LOGO R² is 0.350.
The specimen-type mean LOGO baseline is 0.492; the all-feature unregularized
linear baseline is −0.983. These numbers apply to the stated cohort and are
not transferable design-strength estimates.

The raw EI/CNKI search exports and full candidate list were not retained.
Search yields and screening counts in the manuscript are author-reported,
not independently reconstructible from this repository.
