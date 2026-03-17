# CellNEST — Protein-Protein Interaction Database

Source: [CellNEST](https://github.com/schwartzlab-methods/CellNEST/tree/main/database) (Zohora et al., 2025)

Accessed: September 23, 2025

## Files

- `human_signaling_ppi.csv` — human directed protein-protein interactions
- `mouse_signaling_ppi.csv` — mouse directed protein-protein interactions

## Description

The protein-protein interaction data is used to compute downstream signaling scores
for receptor genes. The directed protein-protein interactions were originally obtained
from the NicheNet database (Browaeys et al., 2020) and interactions were queried
against STRING (v12.0) (Szklarczyk et al., 2023) to obtain experimental confidence
scores. Only interactions with positive experimental scores were retained (for details,
see Zohora et al., 2025, Supplementary Information, Supplementary Note 7).

## Citations

Zohora, F. T., Paliwal, D., Flores-Figueroa, E., et al. (2025). CellNEST reveals
cell-cell relay networks using attention mechanisms on spatial transcriptomics.
*Nature Methods*, 22(7), 1505-1519.

Browaeys, R., Saelens, W., & Saeys, Y. (2020). NicheNet: modeling intercellular
communication by linking ligands to target genes. *Nature Methods*, 17, 159-162.

Szklarczyk, D., Kirsch, R., Koutrouli, M., et al. (2023). The STRING database in
2023. *Nucleic Acids Research*, 51(D1), D638-D646.
