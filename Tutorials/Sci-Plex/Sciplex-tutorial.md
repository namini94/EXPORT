# VEGA: Gene Expression Analysis Tutorial

This tutorial explains the codebase for VEGA (Variational Encoder for Gene Activity), a deep learning framework for analyzing gene expression data with pathway annotations. We'll break down the code into its main components and explain each part.

## Code Structure

The code is organized into several main components:
1. Required Imports
2. Decoder Implementation
3. Mask Creation Utilities
4. SVEGA Model
5. Dataset Handler
6. Training and Analysis Functions
7. Visualization Utilities

## 1. Required Imports

```python
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import scvi
from scvi.dataloaders import AnnDataLoader
from scvi.nn import FCLayers, one_hot

import pandas as pd
import numpy as np
import math
from typing import Iterable, List
from typing import Union
from collections import OrderedDict
from anndata import AnnData
import warnings
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
from spacecutter.models import OrdinalLogisticModel
from skorch import NeuralNet
from spacecutter.callbacks import AscensionCallback
from spacecutter.losses import CumulativeLinkLoss, cumulative_link_loss

import umap
from umap import UMAP
import os
import inspect
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from adjustText import adjust_text
from matplotlib import rcParams
```

These imports provide necessary functionality for:
- Deep learning (PyTorch)
- Data handling (Pandas, NumPy)
- Single-cell analysis (scvi)
- Machine learning (scikit-learn)
- Visualization (matplotlib, seaborn)
- Ordinal regression (spacecutter)

## 2. Decoder Classes

The decoder implementation includes several classes:
1. `DecoderVEGA`: Main decoder for log-transformed data
2. `DecoderVEGACount`: Decoder for count data
3. `SparseLayer`: Custom sparse layer implementation
4. `CustomizedLinearFunction`: Custom autograd function
5. `CustomizedLinear`: Custom linear layer with masking

Key features:
- Masking capabilities for pathway-based analysis
- Support for both log-transformed and count data
- Custom regularization options
- Positive weight constraints

## 3. Mask Creation Utilities

```python
def create_mask( feature_list,
                gmt_paths: Union[str,list] = None,
                add_nodes: int = 1,
                min_genes: int = 0,
                max_genes: int = 1000):
    """ 
    Initialize mask M for GMV from one or multiple .gmt files.
    Parameters
    ----------
        adata
            Scanpy single-cell object.
        gmt_paths
            One or several paths to .gmt files.
        add_nodes
            Additional latent nodes for capturing additional variance.
        min_genes
            Minimum number of genes per GMV.
        max_genes
            Maximum number of genes per GMV.
        copy
            Whether to return a copy of the updated Anndata object.
    Returns
    -------
        adata
            Scanpy single-cell object.
    """
    dict_gmv = OrderedDict()
    # Check if path is a string
    if type(gmt_paths) == str:
        gmt_paths = [gmt_paths]
    for f in gmt_paths:
        d_f = _read_gmt(f, sep='\t', min_g=min_genes, max_g=max_genes)
        # Add to final dictionary
        dict_gmv.update(d_f)

    # Create mask
    mask = _make_gmv_mask(feature_list=feature_list, dict_gmv=dict_gmv, add_nodes=add_nodes)

    
    gmv_names = list(dict_gmv.keys()) + ['UNANNOTATED_'+str(k) for k in range(add_nodes)]
    add_nodes = add_nodes  
    pd.DataFrame(mask).to_csv("/Users/naminiyakan/Documents/VEGA_Code/TCDD/latent/mask.csv",index=True)
    return mask
        

def _make_gmv_mask(feature_list, dict_gmv, add_nodes):
    """ 
    Creates a mask of shape [genes,GMVs] where (i,j) = 1 if gene i is in GMV j, 0 else.
    Note: dict_gmv should be an Ordered dict so that the ordering can be later interpreted.
    Parameters
    ----------
        feature_list
            List of genes in single-cell dataset.
        dict_gmv
            Dictionary of gene_module:genes.
        add_nodes
            Number of additional, fully connected nodes.
    Returns
    -------
        p_mask
            Gene module mask
    """
    assert type(dict_gmv) == OrderedDict
    p_mask = np.zeros((len(feature_list), len(dict_gmv)))
    for j, k in enumerate(dict_gmv.keys()):
        for i in range(p_mask.shape[0]):
            if feature_list[i] in dict_gmv[k]:
                p_mask[i,j] = 1.
    # Add unannotated nodes
    n = add_nodes
    vec = np.ones((p_mask.shape[0], n))
    p_mask = np.hstack((p_mask, vec))
    return p_mask

def _dict_to_gmt(dict_obj, path_gmt, sep='\t', second_col=True):
    """ 
    Write dictionary to gmt format.
    Parameters
    ----------
        dict_obj
            Dictionary with gene_module:[members]
        path_gmt
            Path to save gmt file
        sep
            Separator to use when writing file
        second_col
            Whether to duplicate the first column        
    """
    with open(path_gmt, 'w') as f:
        for k,v in dict_obj.items():
            if second_col:
                to_write = sep.join([k,'SECOND_COL'] + v)+'\n'
            else:
                to_write = sep.join([k] + v) + '\n'
            f.write(to_write)
    return
         

def _read_gmt(fname, sep='\t', min_g=0, max_g=5000):
    """
    Read GMT file into dictionary of gene_module:genes.
    min_g and max_g are optional gene set size filters.
    
    Parameters
    ----------
        fname
            Path to gmt file
        sep
            Separator used to read gmt file.
        min_g
            Minimum of gene members in gene module
        max_g
            Maximum of gene members in gene module
    Returns
    -------
        dict_gmv
            Dictionary of gene_module:genes
    """
    dict_gmv = OrderedDict()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                dict_gmv[val[0]] = val[2:]
    return dict_gmv
```

This section includes functions for:
- Reading GMT files
- Creating gene module masks
- Processing pathway information

## 4. SVEGA Model Implementation

The main model class `SVEGA` includes:

```python
class SVEGA(nn.Module):
    def __init__(self, input_dim, dropout, n_gmvs, z_dropout, ...):
        # Model initialization
```

Key components:
- Encoder network
- Latent space representation
- Decoder network
- Regressor for dose prediction
- Training and inference methods

## 5. Dataset Handler

```python
class Radbio_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_size, train=True, ratio=0.8):
        # Dataset initialization
```

Handles:
- Data loading
- Train/test splitting
- Batch creation

## 6. Training Loop

```python
def train(train_loader, test_loader):
    # Training parameters
    learning_rate = 0.0001
    w2 = 100000
    beta = 0.00005
```

The training procedure includes:
- Model initialization
- Optimization setup
- Loss computation
- Training loop with validation

## 7. Visualization and Analysis

The code includes functions for:
1. Volcano Plot Creation:
```python
def volcano(dfe_res, group1, group2, sig_lvl=3., metric_lvl=3., ...):
    # Volcano plot implementation
```

2. Differential Expression Analysis:
```python
def differential_activity(self, X, group1=None, group2=None, ...):
    # Differential analysis implementation
```

## Usage Example

```python
# Model parameters
input_dim = 5000
dropout = 0.1
n_gmvs = 203
z_dropout = 0.3

# Create data loaders
train_loader = DataLoader(dataset=Radbio_Dataset(dataset_size=57284, train=True, ratio=0.95), 
                         batch_size=128)
test_loader = DataLoader(dataset=Radbio_Dataset(dataset_size=57284, train=False, ratio=0.95), 
                        batch_size=128)

# Train model
model, res = train(train_loader, test_loader)
```

## Analysis and Visualization

After training, you can:
1. Save the model
2. Generate UMAP visualizations
3. Analyze pathway activities
4. Perform differential activity analysis

Example visualization:
```python
# UMAP visualization
reducer = umap.UMAP(random_state=42, min_dist=0.5, n_neighbors=15)
embedding = reducer.fit_transform(latent)
```

## File Paths

The code uses several data files:
- Expression data: 'count_portal.csv'
- Metadata: 'metadata_portal.csv'
- Pathway information: 'Finalized_Wikipathway.gmt'

Make sure to update these paths according to your data location.

