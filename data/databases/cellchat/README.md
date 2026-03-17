# CellChatDB — Ligand-Receptor Pair Database

Source: [CellChatDB](https://github.com/jinworks/CellChat/tree/main/data) (Jin et al., 2021)

The CSV files were extracted from the original `.rda` R data files in the CellChat repository.

Accessed: October 19, 2025

## Files

- `CellChatDB_human_interaction.csv` — human ligand-receptor pairs
- `CellChatDB_mouse_interaction.csv` — mouse ligand-receptor pairs

## Description

CellChatDB is the default ligand-receptor pair database for scCChain. The database
can be filtered to include only user-selected interactions, e.g., interactions
corresponding to secreted signaling. For each entry, we extract the ligand-receptor
pair name, pathway, and ligand/receptor gene names (which may be multi-subunit).

Custom databases can also be used — see the tutorial notebook for instructions on
preparing and loading custom databases.

## Citation

Jin, S., Guerrero-Juarez, C. F., Zhang, L., et al. (2021). Inference and analysis of
cell-cell communication using CellChat. *Nature Communications*, 12, 1088.
