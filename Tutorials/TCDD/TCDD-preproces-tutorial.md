# Single-cell RNA Analysis Tutorial with Scanpy

This tutorial demonstrates how to perform basic single-cell RNA analysis using the Scanpy library, focusing on preprocessing and filtering cell populations.

## Prerequisites

```python
import scanpy as sc
import pandas as pd
```

## Data Loading and Initial Setup

The analysis begins with loading a dataset from an H5AD file, which is the standard format for storing annotated data matrices in Scanpy.

```python
# Load the dataset
adata = sc.read_h5ad("path/to/nault2021_multiDose.h5ad")
```

## Cell Type Filtering

We filter the dataset to focus on specific cell types of interest, particularly liver-related cells.

```python
# Define cell types of interest
cell_types_of_int = [
    "Hepatocytes - central",
    "Hepatocytes - portal",
    "Cholangiocytes",
    "Stellate Cells",
    "Portal Fibroblasts",
    "Endothelial Cells"
]

# Filter AnnData object to keep only desired cell types
adata = adata[adata.obs['celltype'].isin(cell_types_of_int)]
```

## Data Preprocessing

The preprocessing steps include normalization, log transformation, and selecting highly variable genes.

```python
# Normalize the data
sc.pp.normalize_total(adata)

# Log transform the data
sc.pp.log1p(adata)

# Identify highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=5000)

# Filter to keep only highly variable genes
adata = adata[:, adata.var.highly_variable]
```

## Subset Analysis for Portal Hepatocytes

Extract data specifically for portal hepatocytes for further analysis.

```python
# Select portal hepatocytes
cell = 'Hepatocytes - portal'
portal_adata = adata[(adata.obs["celltype"] == cell)]
```

## Data Export

Export the processed data to CSV files for further analysis or visualization.

```python
# Export count matrix
count = portal_adata.to_df()
pd.DataFrame(count).to_csv("count_portal.csv", index=True)

# Export metadata
pd.DataFrame(portal_adata.obs).to_csv("metadata_portal.csv", index=True)

# Export highly variable genes information
pd.DataFrame(portal_adata.var['highly_variable']).to_csv("HVG_portal.csv", index=True)
```

## Key Concepts and Notes

1. **AnnData Object**: The central data structure in Scanpy
   - `.obs`: Contains cell annotations (metadata)
   - `.var`: Contains gene annotations
   - `.X`: Contains the expression matrix

2. **Preprocessing Steps**:
   - Normalization adjusts for differences in sequencing depth
   - Log transformation helps to handle the large dynamic range of expression values
   - Highly variable genes selection reduces dimensionality and focuses on biologically relevant signals

3. **Data Organization**:
   - The final exported files provide:
     - Raw counts matrix
     - Cell metadata
     - Information about highly variable genes

## Troubleshooting Tips

- Ensure the H5AD file path is correct
- Check if cell type names match exactly with the data
- Monitor memory usage when working with large datasets
- Verify that the exported CSV files contain the expected information

## Next Steps

After completing these preprocessing steps, you can proceed with:
- Dimensionality reduction (PCA, UMAP)
- Clustering analysis
- Differential expression analysis
- Trajectory inference

Remember to adjust parameters like `n_top_genes` based on your specific analysis needs and dataset characteristics.
