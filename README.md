# EXPORT
![GitHub Logo](/Miscel/Fig1.png)
## Overview
**EXP**lainable VAE for **OR**dinally perturbed **T**ranscriptomics data (**EXPORT**) is  an interpretable VAE model with a
biological pathway informed architecture, to analyze ordinally perturbed transcriptomics data. Specifically, the low-dimensional latent representations in EXPORT are
ordinally-guided by training an auxiliary deep ordinal regressor network and explicitly modeling the ordinality in the training loss function with an additional ordinal-based
cumulative link loss term.

## Citation

If you use this code, please cite our preprint [paper](https://www.biorxiv.org/content/10.1101/2024.03.28.587231v1):

```
Niyakan, S., Luo, X., Yoon, B. & Qian, X. (2024). Biologically Interpretable VAE with Supervision for Transcriptomics Data Under Ordinal Perturbations, bioRxiv. https://www.biorxiv.org/content/early/2024/03/29/2024.03.28.587231
```
## Getting Started
EXPORT needs 3 main items to analyze your perturbed transcriptomics data:

1.  **Perturbed Gene Expression Data**: The gene expression table of perturbed transcriptomics data under study.
1.  **Biological Pathway Data**: A GMT file specifying the gene-pathway annotation data extracted from resources such as publicly available KEGG, Wikipathways and Reactome databases.
1.  **Ordinal Perturbation levels**: The actual ordinal labels corresponding to perturbation ordinal dosages. 


## Tutorials
- [Pre-processing Single cell-level TCDD data](https://github.com/namini94/EXPORT/blob/main/Tutorials/TCDD/TCDD-preproces-tutorial.md)
- [Analysis of Single cell-level TCDD data with `EXPORT`](https://github.com/namini94/EXPORT/blob/main/Tutorials/TCDD/TCDD-tutorial.md)



