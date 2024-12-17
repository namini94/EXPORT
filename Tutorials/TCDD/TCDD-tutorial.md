# EXPORT: Pathway-based analysis of ordinally perturbed single-cell TCDD data

This tutorial explains the codebase for EXPORT (Explainable VAE for analyzing ordinally perturbed transcriptomics data), a deep learning framework for analyzing ordinally perturbed gene expression data with pathway annotations. We'll break down the code into its main components and explain each part.

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
    def __init__(self, input_dim, dropout, n_gmvs, z_dropout, gmt_paths: Union[list,str] =None, add_nodes: int = 1,min_genes: int = 0,
                max_genes: int =5000,
                positive_decoder: bool = True, exp_paths: Union[list,str] =None, regularizer: str = 'mask' ):
        super(SVEGA, self).__init__()
        self.add_nodes_ = add_nodes
        self.min_genes_ = min_genes
        self.max_genes_ = max_genes
        self.pos_dec_ = positive_decoder
        self.X = pd.read_csv(exp_paths,header=0,index_col=0)
        self.X = self.X.T
        self.features =  self.X.index.tolist()
        
        self.regularizer_ = regularizer
         
        if gmt_paths:
            self.gmv_mask = create_mask(self.features ,gmt_paths, add_nodes, self.min_genes_, self.max_genes_)
        self.encoder = FCLayers(n_in=input_dim,         ## This is going to be FC-1 and FC-2 in Sup_VEGA.pdf
                n_out=800,
                n_cat_list=None,
                n_layers=2,
                n_hidden=800,
                dropout_rate=dropout)
        self.mean = nn.Sequential(nn.Linear(800, n_gmvs), 
                                    nn.Dropout(z_dropout))  ## This is the mean in Sup_VEGA.pdf
        self.logvar = nn.Sequential(nn.Linear(800, n_gmvs), 
                                    nn.Dropout(z_dropout))  ## This is the logvar in Sup_VEGA.pdf
        
        #=========== decoder ================
        self.decoder = DecoderVEGA(mask = self.gmv_mask.T,
                                    n_cat_list = None,
                                    regularizer = self.regularizer_,
                                    positive_decoder = self.pos_dec_
                                    )

        if self.pos_dec_:
            print('Constraining decoder to positive weights', flush=True)
            #self.decoder.sparse_layer[0].reset_params_pos()
            #self.decoder.sparse_layer[0].weight.data *= self.decoder.sparse_layer[0].mask
            self.decoder._positive_weights()
        #=========== classifier ================
        #self.cl1 = nn.Sequential(nn.Linear(n_gmvs, 3))              ###TODOO: What is 2?
        
        #=========== Regressor =================
        #self.reg0 = nn.Linear(n_gmvs,n_gmvs)
        
        #self.reg1 = nn.Linear(n_gmvs, 1)
        #self.relu = nn.ReLU()
        
        self.regressor = nn.Sequential(
            nn.Linear(n_gmvs,n_gmvs),
            nn.Linear(n_gmvs, 1),
            nn.ReLU()
        )
        
        self.ordinal_reg = OrdinalLogisticModel(self.regressor,num_classes=9)
        


    def encode(self, X):
        y = self.encoder(X)
        mu, logvar = self.mean(y), self.logvar(y)
        z = self.sample_latent(mu, logvar)
        return z, mu, logvar

    def sample_latent(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = eps.mul_(std).add_(mu)
        return eps

    @torch.no_grad()
    def to_latent(self, X):
        """ Same as encode, but only returns z (no mu and logvar) """
        y = self.encoder(X)
        mu, logvar = self.mean(y), self.logvar(y)
        z = self.sample_latent(mu, logvar)
        return z
    
    @torch.no_grad()
    def differential_activity(self,
                            X ,
                            group1: Union[str,list] = None,
                            group2: Union[str,list] = None,
                            mode: str = 'change',
                            delta: float = 2.,
                            fdr_target: float = 0.05,
                            **kwargs):
        """
        Bayesian differential activity procedures for GMVs.
        Similar to scVI [Lopez2018]_ Bayesian DGE but for latent variables.
        Differential results are saved in the adata object and returned as a DataFrame.
 
        Parameters
        ----------
        groupby
            anndata object field to group cells (eg. `"cell type"`)
        adata
            scanpy single-cell object. If None, use Anndata attribute of VEGA.
        group1
            reference group(s).
        group2
            outgroup(s).
        mode
            differential activity mode. If `"vanilla"`, uses [Lopez2018]_, if `"change"` uses [Boyeau2019]_.
        delta
            differential activity threshold for `"change"` mode.
        fdr_target
            minimum FDR to consider gene as DE.
        **kwargs
            optional arguments of the bayesian_differential method.
        
        Returns
        -------
        Differential activity results
            
        """
        dosage = pd.read_csv('/Users/naminiyakan/Documents/VEGA_Code/TCDD/Finalized_metadata_portal.csv',header=0,index_col=0)
        level=dosage.iloc[:,3]
        # Check for grouping
        if type(group1)==str:
            group1 = [group1]
        # Loop over groups
        diff_res = dict()
        df_res = []
        for g in group1:
            # get indices and compute values
            idx_g1 = dosage.index.values[dosage.iloc[:,3] == g]
            name_g1 = str(g)
            if not group2:
                idx_g2 = ~idx_g1
                name_g2 = 'rest'
            else: 
                idx_g2 = dosage.index.values[dosage.iloc[:,3] == group2]
                name_g2 = str(group2)
            res_g = self.bayesian_differential(X,
                                                idx_g1,
                                                idx_g2,
                                                mode=mode,
                                                delta=delta,
                                                **kwargs)
            diff_res[name_g1+' vs.'+name_g2] = res_g
            # report results as df
            dict_gmv = OrderedDict()
            # Check if path is a string
            gmt_paths='/Users/naminiyakan/Documents/VEGA_Code/TCDD/Finalized_Wikipathway.gmt'
            if type(gmt_paths) == str:
                gmt_paths = [gmt_paths]
                for f in gmt_paths:
                    d_f = _read_gmt(f, sep='\t', min_g=min_genes, max_g=max_genes)
                    # Add to final dictionary
                    dict_gmv.update(d_f)


            gmv_names = list(dict_gmv.keys()) + ['UNANNOTATED_'+str(k) for k in range(add_nodes)]

            df = pd.DataFrame(res_g, index=gmv_names)
            sort_key = "p_da" if mode == "change" else "bayes_factor"
            df = df.sort_values(by=sort_key, ascending=False)
            if mode == 'change':
                df['is_da_fdr_{}'.format(fdr_target)] = _fdr_de_prediction(df['p_da'], fdr=fdr_target)
            # Add names to result df
            df['comparison'] = '{} vs. {}'.format(name_g1, name_g2)
            df['group1'] = name_g1
            df['group2'] = name_g2
            df_res.append(df)
        # Concatenate df results
        result = pd.concat(df_res, axis=0)
        # Put results in Anndata object
        #adata.uns['_vega']['differential'] = diff_res
        return result
    
    
    @torch.no_grad()    
    def bayesian_differential(self,
                                X,
                                cell_idx1: list, 
                                cell_idx2: list, 
                                n_samples: int = 5000, 
                                use_permutations: bool = True, 
                                n_permutations: int = 5000,
                                mode: int = 'change',
                                delta: float = 2.,
                                alpha: float = 0.66,
                                random_seed: bool = False):
        """ 
        Run Bayesian differential expression in latent space.
        Returns Bayes factor of all factors.
        Parameters
        ----------
        adata
            anndata single-cell object.
        cell_idx1
            indices of group 1.
        cell_idx2
            indices of group 2.
        n_samples
            number of samples to draw from the latent space.
        use_permutations
            whether to use permutations when computing the double integral.
        n_permutations
            number of permutations for MC integral.
        mode
            differential activity test strategy. `"vanilla"` corresponds to [Lopez2018]_, `"change"` to [Boyeau2019]_.
        delta
            for mode `"change"`, the differential threshold to be used.
        random_seed
            seed for reproducibility.
        Returns
        -------
        res
            dictionary with results (Bayes Factor, Mean Absolute Difference)
        """
        #self.eval()
        # Set seed for reproducibility
        #print(mode, delta, alpha)
        if random_seed:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        if mode not in ['vanilla', 'change']:
            raise ValueError('Differential mode not understood. Pick one of "vanilla", "change"')
        epsilon = 1e-12
        # Subset data
        #if sparse.issparse(adata.X):
            #adata1, adata2 = adata.X.A[cell_idx1,:], adata.X.A[cell_idx2,:]
        #else:
        X1, X2 = X.loc[cell_idx1], X.loc[cell_idx2]
        # Sample cell from each condition
        idx1 = np.random.choice(X1.index.values, n_samples)
        idx2 = np.random.choice(X2.index.values, n_samples)
        # To latent
        z1 = self.to_latent(torch.Tensor(X1.loc[idx1].values)).detach().numpy()
        z2 = self.to_latent(torch.Tensor(X2.loc[idx2].values)).detach().numpy()
        # Compare samples by using number of permutations - if 0, just pairwise comparison
        # This estimates the double integral in the posterior of the hypothesis
        if use_permutations:
            z1, z2 = self._scale_sampling(z1, z2, n_perm=n_permutations)
        if mode=='vanilla':
            p_h1 = np.mean(z1 > z2, axis=0)
            p_h2 = 1.0 - p_h1
            md = np.mean(z1 - z2, axis=0)
            bf = np.log(p_h1 + epsilon) - np.log(p_h2 + epsilon) 
            # Wrap results
            res = {'p_h1':p_h1,
                    'p_h2':p_h2,
                    'bayes_factor': bf,
                    'differential_metric':md}
        else:
            diffs = z1 - z2
            md = diffs.mean(0)
            #if not delta:
            #    delta = _estimate_delta(md, min_thresh=1., coef=0.6)
            p_da = np.mean(np.abs(diffs) > delta, axis=0)
            is_da_alpha = (np.abs(md) > delta) & (p_da > alpha)
            res = {'p_da':p_da,
                    'p_not_da':1.-p_da,
                    'bayes_factor':np.log(p_da+epsilon) - np.log((1.-p_da)+epsilon),
                    'is_da_alpha_{}'.format(alpha):is_da_alpha,
                    'differential_metric':md,
                    'delta':delta
                    }
        return res
    
    @staticmethod
    def _scale_sampling(arr1, arr2, n_perm=1000):
        """
        Use permutation to better estimate double integral (create more pair comparisons)
        Inspired by scVI (Lopez et al., 2018)
        Parameters
        ----------
        arr1
            array with data of group 1
        arr2
            array with data of group 2
        n_perm
            number of permutations
        Returns
        -------
        scaled1
            samples for group 1
        scaled2
            samples for group 2
        """
        u, v = (np.random.choice(arr1.shape[0], size=n_perm), np.random.choice(arr2.shape[0], size=n_perm))
        scaled1 = arr1[u]
        scaled2 = arr2[v]
        return scaled1, scaled2

    def decode(self, z, cat_covs=None):
        """ 
        Decode data from latent space.
        
        Parameters
        ----------
        z
            data embedded in latent space
        batch_index
            batch information for samples
        cat_covs
            categorical covariates.
        Returns
        -------
        X_rec
            decoded data
        """
        
        X_rec = self.decoder(z)
        return X_rec


    def forward(self, tensors):
        #input_encode = self._get_inference_input(tensors)
        z, mu, logvar = self.encode(tensors)
        #input_decode = self._get_generative_input(tensors, z)
        #X_rec = self.decode(**input_decode)
        X_rec = self.decode(z)
        #===== Classifier =============
        #cl1 = self.cl1(z)
        #y_hat = F.log_softmax(cl1,dim=1)
        #===== Regressor ==============
        #r1 = self.reg0(z)
        #r2 = self.reg1(r1)
        #dose_hat = self.relu(r2)
        dose_hat = self.ordinal_reg(z)
        cutpoints = self.ordinal_reg.link.cutpoints.detach().cpu().numpy()
        return dose_hat, X_rec, mu, logvar, cutpoints

    def save(self,
            path: str,
            save_adata: bool = False,
            save_history: bool = False,
            overwrite: bool = True,
            save_regularizer_kwargs: bool = True):
        """ 
        Save model parameters to input directory. Saving Anndata object and training history is optional.
        Parameters
        ----------
        path
            path to save directory
        save_adata
            whether to save the Anndata object in the save directory
        save_history
            whether to save the training history in the save directory
        save_regularizer_kwargs
            whether to save regularizer hyperparameters (lambda, penalty matrix...) in the save directory
        """
        attr = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attr = [a for a in attr if not (a[0].startswith("__") and a[0].endswith("__"))]
        attr_dict = {a[0][:-1]:a[1] for a in attr if a[0][-1]=='_'}
        # Save
        if not os.path.exists(path) or overwrite:
            os.makedirs(path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    path
                )
            )
        
        with open(os.path.join(path, 'vega_attr.pkl'), 'wb') as f:
            pickle.dump(attr_dict, f)
        torch.save(self.state_dict(), os.path.join(path, 'vega_params.pt'))
        if save_adata:
            self.adata.write(os.path.join(path, 'anndata.h5ad'))
        if save_history:
            with open(os.path.join(path, 'vega_history.pkl'), 'wb') as h:
                pickle.dump(self.epoch_history, h)
        
        print("Model files saved at {}".format(path))
        return

    @classmethod
    def load(cls,
            path: str,
            adata: AnnData = None,
            device: torch.device = torch.device('cpu'),
            reg_kwargs: dict = None):
        """
        Load model from directory. If adata=None, try to reload Anndata object from saved directory.
        Parameters
        ----------
        path 
            path to save directory
        adata
            scanpy single cell object
        device
            CPU or CUDA
        """
        # Reload model attributes
        with open(os.path.join(path, 'vega_attr.pkl'), 'rb') as f:
            attr = pickle.load(f)
        # Reload regularizer if possible
        
        
        # Reload history if possible
        try:
            with open(os.path.join(path, 'vega_history.pkl'), 'rb') as h:
                model.epoch_history = pickle.load(h)
        except:
            print('No epoch history file found. Loading model with blank training history.')
        # Reload model weights
        model.load_state_dict(torch.load(os.path.join(path, 'vega_params.pt'), map_location=device))
        
        print("Model successfully loaded.")
        return model

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
    def __init__(self, dataset_size, train=True , ratio=0.8):
        X = pd.read_csv('/Users/naminiyakan/Documents/VEGA_Code/TCDD/count_portal.csv',header=0,index_col=0)
        #X = X.to(torch.float32)
        #print(X.dtypes)
        dosage = pd.read_csv('/Users/naminiyakan/Documents/VEGA_Code/TCDD/metadata_portal.csv',header=0)
        Y=dosage.iloc[:,2]
        W=dosage.iloc[:,2]
        # set training and test data size
        train_size=int(ratio*dataset_size)
        self.train=train

        self.data=(X.values.astype(np.float32),Y.values.astype(int),W.values.astype(int))

        if self.train:
            X=X[:train_size]
            Y=Y[:train_size]
            W=W[:train_size]
            print("Training on {} examples".format(train_size))
        else:
            X=X[train_size:]
            Y=Y[train_size:]
            W=W[train_size:]
            print("Testing on {} examples".format(dataset_size-train_size))
    def __getitem__(self, idx):
        "accessing one element in the dataset by index"
        sample=(self.data[0][idx,...],self.data[1][idx],self.data[2][idx])
        return sample
 
    def __len__(self):
        "size of the entire dataset"
        return len(self.data[1])

```

Handles:
- Data loading
- Train/test splitting
- Batch creation

## 6. Training Loop

```python

#w = 1000000
w2 = 100000
beta = 0.00005

#GroundTruth = pd.read_csv('/Users/naminiyakan/Documents/VEGA/shuffled_data/shuffled_labels_with_test.txt',header=None,sep="\t")
#GroundTruth = GroundTruth.iloc[:,4]

def train(train_loader,test_loader):
    
    # Hyperparameters
    #batch_size = 128
    learning_rate = 0.0001
    
    DNN = SVEGA(input_dim, dropout, n_gmvs, z_dropout, gmt_paths, add_nodes, min_genes,
                max_genes,
                positive_decoder, exp_paths, regularizer)
    #DNN.cuda()
    
    epochs = 50

    classificationLoss = torch.nn.NLLLoss()
    #reconstructionLoss = nn.L1Loss()
    #reconstructionLoss = F.mse_loss()
    #optimizer = optim.SGD(DNN.parameters(), lr = 0.001, momentum = 0.8) ## Neurips paper Github code
    optimizer = optim.Adam(DNN.parameters(), lr=learning_rate, weight_decay=5e-4)
    res = []
    for epoch in range(epochs):
        DNN.train()
        model_acc = 0.0
        model_loss = 0.0
        for batch_idx, (data,label,dose) in enumerate(train_loader):
            #data = Variable(data).cuda()
            #label = Variable(label).cuda()
            optimizer.zero_grad()
            dose_hat, x_hat, mu, logvar, cutpoints = DNN(data)
            kld = -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp(), )
            #loss = wp*classificationLoss(y_hat,label)+wr*reconstructionLoss(x_hat,data)
            #dose_loss = w2 * F.mse_loss(dose_hat,dose,reduction="sum")
            #print(dose)
            #print(dose_hat)
            #print(dose_hat.shape)
            #print(dose.unsqueeze(1).shape)
            #dose_loss =  torch.as_tensor(CumulativeLinkLoss(dose_hat,dose,reduction="sum"),dtype=dose.dtype)
            dose_loss =  cumulative_link_loss(dose_hat,dose.unsqueeze(1))
            #print(torch.is_tensor(dose_loss))
            #print(torch.as_tensor(dose_loss))
            #loss = w * classificationLoss(y_hat,label) +  torch.mean(F.mse_loss(x_hat,data, reduction="sum") + beta * kld )+ w2*dose_loss
            loss = torch.mean(F.mse_loss(x_hat,data, reduction="sum") + beta * kld )+ w2*dose_loss
            loss.backward()               # Perform the backward pass to calculate the gradients
            optimizer.step()              # Take optimizer step to update the weights
            if positive_decoder:
                    #self.decoder.sparse_layer[0].apply(clipper)
                    DNN.decoder._positive_weights()
            #_,predicted = torch.max(y_hat.data,1)
            #num_correct = (predicted==label).sum().item()
            #model_acc += num_correct/(len(train_loader)*data.size(0))
            model_loss += loss.item()/len(train_loader)
        print('Epoch: ', epoch, ' - train_loss: ',model_loss)


        DNN.eval()
        
        test_acc = 0.0
        test_loss = 0.0
        total_data = 0.0
        for data, label, dose in test_loader:
            #data = Variable(data).cuda()
            #label = Variable(label).cuda()
            dose_hat, x_hat, mu, logvar, cutpoints = DNN(data)
            pred = np.array(dose_hat.cpu().detach()).argmax(axis=1)
            #print(pred)
            #print(cutpoints)
            #_,predicted = torch.max(y_hat.data,1)
            #total_data += label.size(0)
            #test_acc += (predicted==label).sum().item()/(len(test_loader)*data.size(0))
        #print('Model Test accuracy: ', test_acc)
        res.append([epoch, model_loss , pred])
    return DNN, res
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
def volcano(dfe_res,
            group1: str,
            group2: str,
            sig_lvl: float = 3.,
            metric_lvl: float = 3.,
            annotate_gmv: Union[str,list] = None,
            s:int = 10,
            fontsize: int = 10,
            textsize: int = 8,
            figsize: Union[tuple,list] = None,
            title: str = False,
            save: Union[str,bool] = False):
    """
    Plot Differential GMV results.
    Please run the Bayesian differential acitvity method of VEGA before plotting ("model.differential_activity()")
    
    Parameters
    ----------
    adata
        scanpy single-cell object
    group1
        name of reference group
    group2
        name of out-group
    sig_lvl
        absolute Bayes Factor cutoff (>=0)
    metric_lvl
        mean Absolute Difference cutoff (>=0)
    annotate_gmv
        GMV to be displayed. If None, all GMVs passing significance thresholds are displayed
    s
        dot size
    fontsize
        text size for axis
    textsize
        text size for GMV name display
    title
        title for plot
    save
        path to save figure as pdf
    """
   
    
    
    mad = np.abs(dfe_res['differential_metric'])
    xlim_v = np.abs(dfe_res['bayes_factor']).max() + 0.5
    ylim_v = mad.max()+0.5

    idx_sig = np.arange(len(dfe_res['bayes_factor']))[(np.abs(dfe_res['bayes_factor'])>sig_lvl) & (mad>metric_lvl)]
    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(dfe_res['bayes_factor'], mad,
                 color='grey', s=s, alpha=0.8, linewidth=0)
    ax.scatter(dfe_res['bayes_factor'][idx_sig], mad[idx_sig],
                 color='red', s=s*2, linewidth=0)
    ax.vlines(x=-sig_lvl, ymin=-0.5, ymax=ylim_v, color='black', linestyles='--', linewidth=1., alpha=0.2)
    ax.vlines(x=sig_lvl, ymin=-0.5, ymax=ylim_v, color='black', linestyles='--', linewidth=1., alpha=0.2)
    ax.hlines(y=metric_lvl, xmin=-xlim_v, xmax=xlim_v, color='black', linestyles='--', linewidth=1., alpha=0.2)
    texts = []
    if not annotate_gmv:
        for i in idx_sig:
            name = dfe_res.index.values[i]
            x = dfe_res['bayes_factor'][i]
            y = mad[i]
            texts.append(plt.text(x=x, y=y, s=name, fontdict={'size':textsize}))
    else:
        for name in annotate_gmv:
            i = list(dfe_res.index.values).index(name)
            x = dfe_res['bayes_factor'][i]
            y = mad[i]
            texts.append(plt.text(x=x, y=y, s=name, fontdict={'size':textsize}))
        

    ax.set_xlabel(r'$\log_e$(Bayes factor)', fontsize=fontsize)
    ax.set_ylabel('|Differential Metric|', fontsize=fontsize)
    ax.set_ylim([0,ylim_v])
    ax.set_xlim([-xlim_v,xlim_v])
    if title:
        ax.set_title(title, fontsize=fontsize)
    #adjust_text(texts, only_move={'texts':'xy'}, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    plt.grid(False)
    if save:
        plt.savefig(save, format=save.split('.')[-1], dpi=rcParams['savefig.dpi'], bbox_inches='tight')
    plt.show()


def _fdr_de_prediction(posterior_probas: np.ndarray, fdr: float = 0.05):
    """
    Compute posterior expected FDR and tag features as DE.
    From scvi-tools.
    """
    if not posterior_probas.ndim == 1:
        raise ValueError("posterior_probas should be 1-dimensional")
    sorted_genes = np.argsort(-posterior_probas)
    sorted_pgs = posterior_probas[sorted_genes]
    cumulative_fdr = (1.0 - sorted_pgs).cumsum() / (1.0 + np.arange(len(sorted_pgs)))
    d = (cumulative_fdr <= fdr).sum()
    pred_de_genes = sorted_genes[:d]
    is_pred_de = np.zeros_like(cumulative_fdr).astype(bool)
    is_pred_de[pred_de_genes] = True
    return is_pred_de
```


## Usage Example

```python
# Model parameters
input_dim = 5000
dropout = 0.1
n_gmvs = 203
z_dropout = 0.3

gmt_paths='/Users/naminiyakan/Documents/VEGA_Code/TCDD/Finalized_Wikipathway.gmt'
add_nodes=1
min_genes=0
max_genes= 5000
positive_decoder= True
regularizer = 'mask'

# Create data loaders
train_loader = DataLoader(dataset=Radbio_Dataset(dataset_size=57284, train=True, ratio=0.95), 
                         batch_size=128)
test_loader = DataLoader(dataset=Radbio_Dataset(dataset_size=57284, train=False, ratio=0.95), 
                        batch_size=128)

# Create an instance of the SVEGA model
model = SVEGA(input_dim, dropout, n_gmvs,z_dropout, gmt_paths, add_nodes, min_genes,
                max_genes,
                positive_decoder, exp_paths, regularizer)
# Train model
model, res = train(train_loader, test_loader)
```

## Analysis and Visualization

After training, you can:
1. Save the model
2. Generate UMAP visualizations
3. Analyze pathway activities
4. Perform differential activity analysis

Example Latent Space Visualization:
```python
dataset = Radbio_Dataset(dataset_size=57284,train=True, ratio=1)
x,y,d = dataset.data
print(y)
x = torch.from_numpy(x)
latent = model.to_latent(x)

reducer = umap.UMAP(random_state=42, min_dist=0.5, n_neighbors=15)
embedding = reducer.fit_transform(latent)

umap_df = pd.DataFrame({'UMAP-1':embedding[:,0], 'UMAP-2':embedding[:,1],
                        'level':y})

plt.figure(figsize=[5,4])
plt.scatter(x=embedding[:,0],y=embedding[:,1],c=d, cmap='plasma', s=3, alpha = 0.7,linewidths=0,marker='o')
cbar = plt.colorbar()
#sns.scatterplot(x='UMAP-1', y='UMAP-2', c=y, data=umap_df,
#                linewidth=0, alpha=0.7, s=13)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=12, frameon=False, markerscale=2)
plt.xlabel('UMAP-1', fontsize=12)
plt.ylabel('UMAP-2', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

```

Example Pathway Activity Visualization:
```python
pathway_dict = _read_gmt('/Users/naminiyakan/Documents/VEGA_Code/TCDD/Finalized_Wikipathway.gmt')
pathway_list = list(pathway_dict.keys())+['UNANNOTATED_'+str(k) for k in range(add_nodes)]
print(pathway_list)

dosage = pd.read_csv('/Users/naminiyakan/Documents/VEGA_Code/TCDD/metadata_portal.csv',header=0,index_col=0)

pathway_encoded_df = pd.DataFrame(data=latent,index=dosage.index.values, columns=pathway_list)
print(pathway_encoded_df)
pd.DataFrame(pathway_encoded_df).to_csv("/Users/naminiyakan/Documents/VEGA_Code/TCDD/latent/pathway_encoded_df.csv",index=True)


plt.figure(figsize=[5,4])
plt.scatter(embedding[:,0], embedding[:,1], alpha = 0.7, linewidths=0,
            c = pathway_encoded_df['Aflatoxin B1 metabolism%WikiPathways_20240101%WP1262%Mus musculus'], marker='o', s=3, cmap = 'seismic')
cbar = plt.colorbar()
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(12)
plt.title('Aflatoxin B1 Metabolism', fontsize=12)
plt.xlabel('UMAP-1', fontsize=12)
plt.ylabel('UMAP-2', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

```
Example Differential Pathway Activity Analysis:

```python
X = pd.read_csv('/Users/naminiyakan/Documents/VEGA_Code/TCDD/count_portal.csv',header=0,index_col=0)

da_df = model.differential_activity(X=X, fdr_target=0.05, group1='Low', group2='High', mode='vanilla')
print(da_df.head(20))
pd.DataFrame(da_df).to_csv("/Users/naminiyakan/Documents/VEGA_Code/TCDD/DE_Pathway_SVEGA/LH_vanilla.csv",index=True)
volcano(da_df, group1='Low', group2='High', title='Low vs. High')

```


## File Paths

The code uses several data files:
- Expression data: 'count_portal.csv'
- Metadata: 'metadata_portal.csv'
- Pathway information: 'Finalized_Wikipathway.gmt'

Make sure to update these paths according to your data location.

