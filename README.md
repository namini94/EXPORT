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
## Quick Start
In order to analyze your multi-sample spatial transcriptional (ST) data with MUSTANG, 4 main steps should be performed:

1.  **Spots Spatial Graph**: The adjacency matrix of spots spatial graph should be extracted based on the layout.
1.  **Spots Transcriptional Graph**: The adjacency matrix of spots transcriptional graph in which spots that are transcriptionally similar to eachother are connected with an edge should be extracted.
1.  **Spots Similarity Graph**: The adjacency matrix of spots similarity graph should be constructed based on adjacency matrices of spots spatial and transcriptional graphs. 
1.  **Bayesian Deconvolution Analysis**: The Poisson discrete deconvolution model should be applied to extract the deconvolution parameters.

## Tutorials
- [Analysis of Mouse Brain ST data with `MUSTANG`](https://github.com/namini94/MUSTANG/blob/main/Tutorial/Mouse%20Brain%20/Mouse_Brain.md)
- [Semi-Synthetic Multi-sample Data Generation](https://github.com/namini94/MUSTANG/blob/main/Tutorial/Semi-synthetic%20Data%20Simulation/DataSimulation.md)


