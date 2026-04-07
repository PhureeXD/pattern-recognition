# IEEE-CIS Fraud Detection Project

Common project structure for the team workspace.

## Structure

- `docs/`: project documentation and dataset notes
- `notebooks/`: Colab and experiment notebooks
- `notebooks/team1/`: Team 1 notebooks
- `notebooks/team2/`: Team 2 modeling notebooks
- `src/`: reusable Python code for preprocessing, features, and utilities
- `data/raw/`: original downloaded data
- `data/interim/`: temporary cleaned outputs and intermediate parquet files
- `data/processed/`: finalized model-ready datasets
- `models/`: saved models and artifacts
- `reports/`: figures, summaries, and presentation outputs

## Current Files

- `docs/CLAUDE.md`
- `docs/dataset.md`
- `notebooks/team1/team1_member1_colab_setup.ipynb`
- `notebooks/team1/team1_member2_colab_setup.ipynb`
- `notebooks/team2/team2_member3_lgbm_baseline_colab.ipynb`

## Suggested Next Step

Move repeated notebook helper logic into `src/` when the preprocessing flow becomes stable.
