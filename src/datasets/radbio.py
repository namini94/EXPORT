"""
RadbioDataset implementation for handling gene expression data.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Tuple, Optional

class RadbioDataset(Dataset):
    """
    Dataset class for handling radiobiology gene expression data.
    
    Parameters
    ----------
    dataset_size : int
        Total size of the dataset
    train : bool, optional (default=True)
        Whether to load training or testing data
    ratio : float, optional (default=0.8)
        Train/test split ratio
    exp_path : str, optional
        Path to expression data CSV
    metadata_path : str, optional
        Path to metadata CSV
        
    Attributes
    ----------
    data : tuple
        Tuple containing (expression_data, labels, doses)
    """
    
    def __init__(self, 
                 dataset_size: int, 
                 train: bool = True, 
                 ratio: float = 0.8,
                 exp_path: Optional[str] = None,
                 metadata_path: Optional[str] = None):
        """Initialize the dataset."""
        # Set default paths if not provided
        self.exp_path = exp_path or '/path/to/count_portal.csv'
        self.metadata_path = metadata_path or '/path/to/metadata_portal.csv'
        
        # Load data
        X = pd.read_csv(self.exp_path, header=0, index_col=0)
        dosage = pd.read_csv(self.metadata_path, header=0)
        Y = dosage.iloc[:, 2]
        W = dosage.iloc[:, 2]

        # Convert to appropriate types
        self.X = X.values.astype(np.float32)
        self.Y = Y.values.astype(int)
        self.W = W.values.astype(int)
        
        # Set training and test data size
        self.train_size = int(ratio * dataset_size)
        self.train = train
        
        if self.train:
            self.X = self.X[:self.train_size]
            self.Y = self.Y[:self.train_size]
            self.W = self.W[:self.train_size]
            print(f"Training on {self.train_size} examples")
        else:
            self.X = self.X[self.train_size:]
            self.Y = self.Y[self.train_size:]
            self.W = self.W[self.train_size:]
            print(f"Testing on {dataset_size - self.train_size} examples")
        
        self.data = (self.X, self.Y, self.W)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, int]:
        """
        Get a sample from the dataset.
        
        Parameters
        ----------
        idx : int
            Index of the sample to retrieve
            
        Returns
        -------
        tuple
            Tuple containing (expression_data, label, dose)
        """
        return (self.data[0][idx, ...], self.data[1][idx], self.data[2][idx])
    
    def __len__(self) -> int:
        """
        Get the total number of samples.
        
        Returns
        -------
        int
            Number of samples in the dataset
        """
        return len(self.data[1])
    
    @property
    def input_size(self) -> int:
        """
        Get the input dimension (number of genes).
        
        Returns
        -------
        int
            Number of genes in the expression data
        """
        return self.X.shape[1]
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for balanced training.
        
        Returns
        -------
        torch.Tensor
            Tensor of class weights
        """
        class_counts = np.bincount(self.Y)
        total = len(self.Y)
        class_weights = total / (len(class_counts) * class_counts)
        return torch.FloatTensor(class_weights)
    
    def get_metadata(self) -> pd.DataFrame:
        """
        Get the full metadata DataFrame.
        
        Returns
        -------
        pandas.DataFrame
            Metadata for the dataset
        """
        return pd.read_csv(self.metadata_path)
    
    def get_gene_names(self) -> list:
        """
        Get the list of gene names.
        
        Returns
        -------
        list
            List of gene names in the expression data
        """
        return pd.read_csv(self.exp_path, header=0, index_col=0).columns.tolist()
