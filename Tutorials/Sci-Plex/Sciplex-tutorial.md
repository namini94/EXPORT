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

```python
"""
Custom modules and layers for VEGA.
Acknowledgements:
Customized Linear from Uchida Takumi with modifications.
https://github.com/uchida-takumi/CustomizedLinear/blob/master/CustomizedLinear.py
Masked decoder based on LinearDecoderSCVI.
https://github.com/YosefLab/scvi-tools/blob/8f5a9cc362325abbb7be1e07f9523cfcf7e55ec0/scvi/core/modules/_base/__init__.py  
"""

n_out = 5000


class DecoderVEGA(nn.Module):
    """
    Decoder for VEGA model (log-transformed data).
    Parameters
    ----------
    mask
        gene-gene module membership matrix
    n_cat_list
        list encoding number of categories for each covariate
    regularizer
        choice of regularizer for the decoder. Default to masking (VEGA)
    positive_decoder
        whether to constrain decoder weigths to positive values
    reg_kwargs
        keyword arguments for the regularizer
    """
    def __init__(self,
                mask: np.ndarray,
                n_cat_list: Iterable[int] = None, 
                regularizer: str = 'mask',
                positive_decoder: bool = True,
                reg_kwargs=None):
        super(DecoderVEGA, self).__init__()
        self.n_input = mask.shape[0]
        self.n_output = mask.shape[1]
        self.reg_method = regularizer
        if reg_kwargs and (reg_kwargs.get('d', None) is None):
            reg_kwargs['d'] = ~mask.T.astype(bool)
        if reg_kwargs is None:
            reg_kwargs = {}
        if regularizer=='mask':
            print('Using masked decoder', flush=True)
            self.decoder = SparseLayer(mask,
                                        n_cat_list=n_cat_list,
                                        use_batch_norm=False,
                                        use_layer_norm=False,
                                        bias=True,
                                        dropout_rate=0)
        else:
            raise ValueError("Regularizer not recognized. Choose one of ['mask', 'gelnet', 'l1']")

    def forward(self, x: torch.Tensor, *cat_list:int):
        """ Forward method for VEGA decoder """
        return self.decoder(x, *cat_list)
    
    def _get_weights(self):
        """ Returns weight matrix of linear decoder (for regularization purposes)"""
        if isinstance(self.decoder, SparseLayer):
            w = self.decoder.sparse_layer[0].weight
        elif isinstance(self.decoder, FCLayers):
            w = self.decoder.fc_layers[0][0].weight
        return w
        
    def quadratic_penalty(self):
        """ Returns loss associated with quadratic penalty of regularizer """
        if self.reg_method == 'mask':
            return torch.tensor(0)
        

    def proximal_update(self):
        """ Directly updates weights using proximal operator (for non-smooth regularizer) """
        if self.reg_method == 'mask':
            return
        
    
    def _positive_weights(self, use_softplus=False):
        """ Set negative weights to 0 if positive_decoder is True """
        w = self._get_weights()
        if use_softplus:
            w.data = nn.functional.softplus(w.data)
        else:
            w.data = w.data.clamp(0)
        return


class DecoderVEGACount(nn.Module):
    """
    Masked linear decoder for VEGA in SCVI mode (count data). Note: positive weights not included yet.
    Parameters
    ----------
    mask
        gene-gene module membership matrix
    n_cat_list
        list encoding number of categories for each covariate
    n_continuous_cov
        number of continuous covariates
    use_batch_norm
        whether to use batch normalization in the decoder
    use_layer_norm
        whether to use layer normalization in the decoder
    bias
        whether to use a bias parameter in the linear decoder 
    """
    def __init__(self, 
                mask,
                n_cat_list: Iterable[int] = None,
                n_continuous_cov: int = 0,
                use_batch_norm: bool = False,
                use_layer_norm: bool = False,
                bias: bool = False
                ):
        super(DecoderVEGACount, self).__init__()
        self.n_input = mask.shape[1]
        self.n_output = mask.shape[0]
        # Mean and dropout decoder - dropout is fully connected
        self.px_scale = SparseLayer(
            mask=mask,
            n_cat_list=n_cat_list,
            n_continuous_cov = n_continuous_cov,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            bias=bias,
            dropout_rate=0)
        self.px_dropout = SparseLayer(
            mask=mask,
            n_cat_list=n_cat_list,
            n_continuous_cov = n_continuous_cov,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            bias=bias,
            dropout_rate=0)

    def forward(self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int):
        """ Forward pass through VEGA's decoder """
        raw_px_scale = self.px_scale(z, *cat_list)
        px_scale = torch.softmax(raw_px_scale, dim=-1)
        px_dropout = self.px_dropout(z, *cat_list)
        px_rate = torch.exp(library) * px_scale
        px_r = None

        return px_scale, px_r, px_rate, px_dropout
    
class SparseLayer(nn.Module):
    """
    Sparse Layer class. Inspired by SCVI 'FCLayers' but constrained to 1 layer.
    
    Parameters:
    -----------
    mask
        gene-gene module membership matrix
    n_cat_list
        list encoding number of categories for each covariate
    n_continuous_cov
        number of continuous covariates
    use_activation
        whether to use an activation layer in the decoder
    use_batch_norm
        whether to use batch normalization in the decoder
    use_layer_norm
        whether to use layer normalization in the decoder
    bias
        whether to use a bias parameter in the linear decoder
    """
    def __init__(self,
                mask: np.ndarray,
                n_cat_list: Iterable[int] = None,
                n_continuous_cov: int = 0,
                use_activation: bool = False,
                use_batch_norm: bool = False,
                use_layer_norm: bool = False,
                bias: bool = True,
                dropout_rate: float = 0.1,
                activation_fn: nn.Module = None
                ):
        # Initialize custom sparse layer
        super().__init__()
        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []
        self.n_continuous_cov = n_continuous_cov
        self.cat_dim = sum(self.n_cat_list)
        mask_with_cov = np.vstack((mask, np.ones((self.n_continuous_cov+self.cat_dim, mask.shape[1]))))
        self.sparse_layer = nn.Sequential(
                                    CustomizedLinear(mask_with_cov),
                                    nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                                    if use_batch_norm else None,
                                    nn.LayerNorm(n_out, elementwise_affine=False)
                                    if use_layer_norm
                                    else None,
                                    activation_fn() if use_activation else None,
                                    nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None
                                    )
    def forward(self, x: torch.Tensor, *cat_list: int):
        """
        Forward computation on x for sparse layer.
        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample
        x: torch.Tensor
        Returns
        -------
        py:class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError(
                "nb. categorical args provided doesn't match init. params."
            )
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for layer in self.sparse_layer:
            if layer is not None:
                if isinstance(layer, nn.BatchNorm1d):
                    if x.dim() == 3:
                        x = torch.cat(
                            [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                        )
                    else:
                        x = layer(x)
                else:
                    if isinstance(layer, CustomizedLinear):
                        if x.dim() == 3:
                            one_hot_cat_list_layer = [
                                o.unsqueeze(0).expand(
                                    (x.size(0), o.size(0), o.size(1))
                                )
                                for o in one_hot_cat_list
                            ]
                        else:
                            one_hot_cat_list_layer = one_hot_cat_list
                        x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                    x = layer(x)
        return x 
    

class CustomizedLinearFunction(torch.autograd.Function):
    """
    Autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask
        #if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask


class CustomizedLinear(nn.Module):
    def __init__(self, mask, bias=True):
        """
        Extended torch.nn module which mask connection.
        Parameters
        ----------
        mask
            gene-gene module membership matrix
        bias
            whether to use a bias term
        """
        super(CustomizedLinear, self).__init__()
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()

        self.mask = nn.Parameter(self.mask, requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()

        # mask weight
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_params_pos(self):
        """ Same as reset_parameters, but only initialize to positive values. """
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(0,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

```
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

