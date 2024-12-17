# Single-cell resoloution TCDD data preprocessing tutorial

This tutorial demonstrates how to preprocess the single-cell TCDD data for further anaysis using our EXPORT model.

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

Export the processed data to CSV files for further analysis.

```python
# Export count matrix
count = portal_adata.to_df()
pd.DataFrame(count).to_csv("count_portal.csv", index=True)

# Export metadata
pd.DataFrame(portal_adata.obs).to_csv("metadata_portal.csv", index=True)

# Export highly variable genes information
pd.DataFrame(portal_adata.var['highly_variable']).to_csv("HVG_portal.csv", index=True)
```


