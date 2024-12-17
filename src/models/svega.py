import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Union, Iterable
import os
import pickle
import inspect
from collections import OrderedDict

from ..utils.preprocessing import create_mask, _read_gmt
from .decoder import DecoderVEGA
from spacecutter.models import OrdinalLogisticModel
from scvi.nn import FCLayers

class SVEGA(nn.Module):
    """
    SVEGA (Supervised Variational Encoder for Gene Activity) model.
    
    Parameters
    ----------
    input_dim : int
        Number of input genes
    dropout : float
        Dropout rate for the encoder
    n_gmvs : int
        Number of gene module variables
    z_dropout : float
        Dropout rate for latent space
    gmt_paths : Union[list, str]
        Path(s) to GMT file(s)
    add_nodes : int
        Number of additional nodes to add
    min_genes : int
        Minimum number of genes per module
    max_genes : int
        Maximum number of genes per module
    positive_decoder : bool
        Whether to constrain decoder weights to be positive
    exp_paths : Union[list, str]
        Path to expression data
    regularizer : str
        Type of regularization to use
    """
    def __init__(self, 
                 input_dim: int, 
                 dropout: float, 
                 n_gmvs: int, 
                 z_dropout: float,
                 gmt_paths: Union[list, str] = None,
                 add_nodes: int = 1,
                 min_genes: int = 0,
                 max_genes: int = 5000,
                 positive_decoder: bool = True,
                 exp_paths: Union[list, str] = None,
                 regularizer: str = 'mask'):
        super(SVEGA, self).__init__()
        
        self.add_nodes_ = add_nodes
        self.min_genes_ = min_genes
        self.max_genes_ = max_genes
        self.pos_dec_ = positive_decoder
        self.regularizer_ = regularizer
        
        # Load and process data
        self.X = pd.read_csv(exp_paths, header=0, index_col=0).T
        self.features = self.X.index.tolist()
        
        # Create mask if GMT paths provided
        if gmt_paths:
            self.gmv_mask = create_mask(self.features, gmt_paths, add_nodes, 
                                      self.min_genes_, self.max_genes_)
        
        # Initialize encoder
        self.encoder = FCLayers(
            n_in=input_dim,
            n_out=800,
            n_cat_list=None,
            n_layers=2,
            n_hidden=800,
            dropout_rate=dropout
        )
        
        # Mean and variance encoders
        self.mean = nn.Sequential(
            nn.Linear(800, n_gmvs),
            nn.Dropout(z_dropout)
        )
        self.logvar = nn.Sequential(
            nn.Linear(800, n_gmvs),
            nn.Dropout(z_dropout)
        )
        
        # Initialize decoder
        self.decoder = DecoderVEGA(
            mask=self.gmv_mask.T,
            n_cat_list=None,
            regularizer=self.regularizer_,
            positive_decoder=self.pos_dec_
        )
        
        # Initialize regressor
        self.regressor = nn.Sequential(
            nn.Linear(n_gmvs, n_gmvs),
            nn.Linear(n_gmvs, 1),
            nn.ReLU()
        )
        
        self.ordinal_reg = OrdinalLogisticModel(self.regressor, num_classes=9)
        
        if self.pos_dec_:
            print('Constraining decoder to positive weights', flush=True)
            self.decoder._positive_weights()

    def encode(self, X):
        """Encode data to latent space."""
        y = self.encoder(X)
        mu, logvar = self.mean(y), self.logvar(y)
        z = self.sample_latent(mu, logvar)
        return z, mu, logvar

    def sample_latent(self, mu, logvar):
        """Sample from latent space using reparameterization trick."""
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = eps.mul_(std).add_(mu)
        return eps

    @torch.no_grad()
    def to_latent(self, X):
        """Convert input data to latent representation."""
        y = self.encoder(X)
        mu, logvar = self.mean(y), self.logvar(y)
        z = self.sample_latent(mu, logvar)
        return z

    def decode(self, z, cat_covs=None):
        """Decode data from latent space."""
        X_rec = self.decoder(z)
        return X_rec

    def forward(self, tensors):
        """Forward pass through the model."""
        z, mu, logvar = self.encode(tensors)
        X_rec = self.decode(z)
        dose_hat = self.ordinal_reg(z)
        cutpoints = self.ordinal_reg.link.cutpoints.detach().cpu().numpy()
        return dose_hat, X_rec, mu, logvar, cutpoints

    def save(self, path: str, save_adata: bool = False, 
             save_history: bool = False, overwrite: bool = True):
        """Save model parameters and attributes."""
        attr = inspect.getmembers(self, lambda a: not(inspect.isroutine(a)))
        attr = [a for a in attr if not(a[0].startswith('__') and a[0].endswith('__'))]
        attr_dict = {a[0][:-1]:a[1] for a in attr if a[0][-1]=='_'}
        
        if not os.path.exists(path) or overwrite:
            os.makedirs(path, exist_ok=overwrite)
        else:
            raise ValueError(f"{path} already exists. Please provide an unexisting directory for saving.")
            
        with open(os.path.join(path, 'vega_attr.pkl'), 'wb') as f:
            pickle.dump(attr_dict, f)
            
        torch.save(self.state_dict(), os.path.join(path, 'vega_params.pt'))
        
        print(f"Model files saved at {path}")
        
    @classmethod
    def load(cls, path: str, device: torch.device = torch.device('cpu')):
        """Load saved model."""
        with open(os.path.join(path, 'vega_attr.pkl'), 'rb') as f:
            attr = pickle.load(f)
            
        model = cls(**attr)
        model.load_state_dict(torch.load(os.path.join(path, 'vega_params.pt'), 
                                       map_location=device))
        
        print("Model successfully loaded.")
        return model
