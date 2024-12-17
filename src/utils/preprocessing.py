"""
Preprocessing utilities for VEGA.

This module contains functions for data preprocessing, including
mask creation and GMT file handling.
"""

import numpy as np
import pandas as pd
from typing import Union, List
from collections import OrderedDict

def create_mask(feature_list: List[str],
                gmt_paths: Union[str, List[str]] = None,
                add_nodes: int = 1,
                min_genes: int = 0,
                max_genes: int = 1000) -> np.ndarray:
    """
    Initialize mask M for GMV from one or multiple .gmt files.
    
    Parameters
    ----------
    feature_list : List[str]
        List of gene names
    gmt_paths : Union[str, List[str]], optional
        Path(s) to .gmt files
    add_nodes : int, optional
        Additional latent nodes for capturing additional variance
    min_genes : int, optional
        Minimum number of genes per GMV
    max_genes : int, optional
        Maximum number of genes per GMV
        
    Returns
    -------
    np.ndarray
        Mask matrix
    """
    dict_gmv = OrderedDict()
    
    # Handle single path case
    if isinstance(gmt_paths, str):
        gmt_paths = [gmt_paths]
        
    # Read all GMT files
    for f in gmt_paths:
        d_f = _read_gmt(f, sep='\t', min_g=min_genes, max_g=max_genes)
        dict_gmv.update(d_f)

    # Create mask
    mask = _make_gmv_mask(feature_list=feature_list, 
                         dict_gmv=dict_gmv, 
                         add_nodes=add_nodes)

    return mask

def _make_gmv_mask(feature_list: List[str], 
                   dict_gmv: OrderedDict,
                   add_nodes: int) -> np.ndarray:
    """
    Creates a mask of shape [genes,GMVs] where (i,j) = 1 if gene i is in GMV j, 0 else.
    
    Parameters
    ----------
    feature_list : List[str]
        List of genes in dataset
    dict_gmv : OrderedDict
        Dictionary of gene_module:genes
    add_nodes : int
        Number of additional, fully connected nodes
        
    Returns
    -------
    np.ndarray
        Gene module mask
    """
    if not isinstance(dict_gmv, OrderedDict):
        raise TypeError("dict_gmv must be an OrderedDict")
        
    p_mask = np.zeros((len(feature_list), len(dict_gmv)))
    
    for j, k in enumerate(dict_gmv.keys()):
        for i in range(p_mask.shape[0]):
            if feature_list[i] in dict_gmv[k]:
                p_mask[i,j] = 1.
                
    # Add unannotated nodes
    vec = np.ones((p_mask.shape[0], add_nodes))
    p_mask = np.hstack((p_mask, vec))
    
    return p_mask

def _read_gmt(fname: str, 
              sep: str = '\t',
              min_g: int = 0,
              max_g: int = 5000) -> OrderedDict:
    """
    Read GMT file into dictionary of gene_module:genes.
    
    Parameters
    ----------
    fname : str
        Path to gmt file
    sep : str, optional
        Separator used in gmt file
    min_g : int, optional
        Minimum genes in gene module
    max_g : int, optional
        Maximum genes in gene module
        
    Returns
    -------
    OrderedDict
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

def _dict_to_gmt(dict_obj: dict,
                 path_gmt: str,
                 sep: str = '\t',
                 second_col: bool = True) -> None:
    """
    Write dictionary to gmt format.
    
    Parameters
    ----------
    dict_obj : dict
        Dictionary with gene_module:[members]
    path_gmt : str
        Path to save gmt file
    sep : str, optional
        Separator to use when writing file
    second_col : bool, optional
        Whether to duplicate the first column
    """
    with open(path_gmt, 'w') as f:
        for k, v in dict_obj.items():
            if second_col:
                to_write = sep.join([k, 'SECOND_COL'] + v) + '\n'
            else:
                to_write = sep.join([k] + v) + '\n'
            f.write(to_write)
