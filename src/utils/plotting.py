"""
Plotting utilities for VEGA.

This module contains functions for visualizing results from VEGA analyses.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
from matplotlib import rcParams
from adjustText import adjust_text

def volcano(dfe_res: pd.DataFrame,
           group1: str,
           group2: str,
           sig_lvl: float = 3.,
           metric_lvl: float = 3.,
           annotate_gmv: Union[str, List[str]] = None,
           s: int = 10,
           fontsize: int = 10,
           textsize: int = 8,
           figsize: Union[tuple, list] = None,
           title: str = False,
           save: Union[str, bool] = False):
    """
    Plot Differential GMV results.
    
    Parameters
    ----------
    dfe_res : pd.DataFrame
        Differential expression results
    group1 : str
        Name of reference group
    group2 : str
        Name of out-group
    sig_lvl : float, optional
        Absolute Bayes Factor cutoff
    metric_lvl : float, optional
        Mean Absolute Difference cutoff
    annotate_gmv : Union[str, List[str]], optional
        GMV to be displayed
    s : int, optional
        Dot size
    fontsize : int, optional
        Text size for axis
    textsize : int, optional
        Text size for GMV name display
    figsize : Union[tuple, list], optional
        Figure size
    title : str, optional
        Title for plot
    save : Union[str, bool], optional
        Path to save figure
    """
    mad = np.abs(dfe_res['differential_metric'])
    xlim_v = np.abs(dfe_res['bayes_factor']).max() + 0.5
    ylim_v = mad.max() + 0.5

    # Find significant points
    idx_sig = np.arange(len(dfe_res['bayes_factor']))[
        (np.abs(dfe_res['bayes_factor']) > sig_lvl) & 
        (mad > metric_lvl)
    ]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all points
    ax.scatter(dfe_res['bayes_factor'], mad,
              color='grey', s=s, alpha=0.8, linewidth=0)
    
    # Highlight significant points
    ax.scatter(dfe_res['bayes_factor'][idx_sig], mad[idx_sig],
              color='red', s=s*2, linewidth=0)
    
    # Add threshold lines
    ax.vlines(x=-sig_lvl, ymin=-0.5, ymax=ylim_v, 
             color='black', linestyles='--', linewidth=1., alpha=0.2)
    ax.vlines(x=sig_lvl, ymin=-0.5, ymax=ylim_v, 
             color='black', linestyles='--', linewidth=1., alpha=0.2)
    ax.hlines(y=metric_lvl, xmin=-xlim_v, xmax=xlim_v, 
             color='black', linestyles='--', linewidth=1., alpha=0.2)

    # Add labels
    texts = []
    if not annotate_gmv:
        for i in idx_sig:
            name = dfe_res.index.values[i]
            x = dfe_res['bayes_factor'][i]
            y = mad[i]
            texts.append(plt.text(x=x, y=y, s=name, fontdict={'size': textsize}))
    else:
        for name in annotate_gmv:
            i = list(dfe_res.index.values).index(name)
            x = dfe_res['bayes_factor'][i]
            y = mad[i]
            texts.append(plt.text(x=x, y=y, s=name, fontdict={'size': textsize}))

    # Customize plot
    ax.set_xlabel(r'$\log_e$(Bayes factor)', fontsize=fontsize)
    ax.set_ylabel('|Differential Metric|', fontsize=fontsize)
    ax.set_ylim([0, ylim_v])
    ax.set_xlim([-xlim_v, xlim_v])
    
    if title:
        ax.set_title(title, fontsize=fontsize)
        
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    plt.grid(False)

    # Save if requested
    if save:
        plt.savefig(save, format=save.split('.')[-1], 
                   dpi=rcParams['savefig.dpi'], bbox_inches='tight')
    
    plt.show()


def _fdr_de_prediction(posterior_probas: np.ndarray, 
                      fdr: float = 0.05) -> np.ndarray:
    """
    Compute posterior expected FDR and tag features as DE.
    
    Parameters
    ----------
    posterior_probas : np.ndarray
        Posterior probabilities
    fdr : float, optional
        Target false discovery rate
        
    Returns
    -------
    np.ndarray
        Boolean array indicating DE features
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
