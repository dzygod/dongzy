# Structures manuscript: current data and reproduction boundary

The current manuscript has two explicitly different cohorts. The primary analysis uses `uhpc_nc_bond_literature_data_v9.csv`: 527 source rows, 467 modeled rows after the stated selection rules, comprising 286 shear and 181 tensile records from 15 studies. The verified extension uses `uhpc_nc_bond_literature_data_v10_core.csv`: 549 source rows, 484 modeled rows, comprising 296 shear and 188 tensile records from 18 studies. These cohorts and their metrics must not be combined.

The real literature-integrated CSV files are **not included** in this public repository. `data/raw/literature_sample.csv` is synthetic; it verifies that a workflow runs but cannot reproduce the paper's figures, tables, or metrics. The real input files and the matching run reports may be requested from the corresponding author under the manuscript's data availability terms. No result should be described as independently reproducible from the public synthetic sample alone.

With the corresponding real data file available locally, run:

```bash
python scripts/paper1_group_validation_v4.py --csv /path/to/uhpc_nc_bond_literature_data_v9.csv --out outputs/run-paper1-v9
python scripts/paper1_group_validation_v4.py --csv /path/to/uhpc_nc_bond_literature_data_v10_core.csv --out outputs/run-paper1-v10-core
```

The script writes `report.json` for each cohort. The primary v9 report gives shear XGBoost repeated random five-fold R² = 0.834, leave-one-study-out (LOGO) R² = 0.431, and specimen-type mean LOGO R² = 0.549; tensile XGBoost values are 0.635 and 0.251. The v10_core extension gives shear values 0.841, 0.453, and 0.563; tensile XGBoost values are 0.629 and 0.374. Values are rounded to three decimals. The global IQR prescreen results are conditional on the prescreened records; the script also reports a train-only IQR inclusion sensitivity analysis.

The manuscript's source-level caveats remain relevant to any reanalysis. Some retained records lack a standardized interface-preparation category. In the Aaleti subset, 21 retained records have that category missing, and their curing age is coded as an estimate because the exact test age was not given in the source. Missing categorical values must not be silently recoded as a known treatment. Historical v8 source corrections are documented separately in [`paper1_v8_source_audit.md`](paper1_v8_source_audit.md).
