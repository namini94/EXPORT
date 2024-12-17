import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable
import math
from scvi.nn import FCLayers, one_hot

class DecoderVEGA(nn.Module):
    """
    Decoder for VEGA model (log-transformed data).
    
    Parameters
    ----------
    mask : np.ndarray
        Gene-gene module membership matrix
    n_cat_list : Iterable[int], optional
        List encoding number of categories for each covariate
    regularizer : str
        Choice of regularizer for the decoder
    positive_decoder : bool
        Whether to constrain decoder weights to positive values
    reg_kwargs : dict, optional
        Keyword arguments for the regularizer
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
            
        if regularizer == 'mask':
            print('Using masked decoder', flush=True)
            self.decoder = SparseLayer(mask,
                                     n_cat_list=n_cat_list,
                                     use_batch_norm=False,
                                     use_layer_norm=False,
                                     bias=True,
                                     dropout_rate=0)
        else:
            raise ValueError("Regularizer not recognized. Choose one of ['mask']")

    def forward(self, x: torch.Tensor, *cat_list: int):
        """Forward method for VEGA decoder."""
        return self.decoder(x, *cat_list)
    
    def _get_weights(self):
        """Returns weight matrix of linear decoder."""
        if isinstance(self.decoder, SparseLayer):
            w = self.decoder.sparse_layer[0].weight
        elif isinstance(self.decoder, FCLayers):
            w = self.decoder.fc_layers[0][0].weight
        return w
        
    def quadratic_penalty(self):
        """Returns loss associated with quadratic penalty of regularizer."""
        if self.reg_method == 'mask':
            return torch.tensor(0)
        
    def proximal_update(self):
        """Updates weights using proximal operator."""
        if self.reg_method == 'mask':
            return
        
    def _positive_weights(self, use_softplus=False):
        """Set negative weights to 0 if positive_decoder is True."""
        w = self._get_weights()
        if use_softplus:
            w.data = F.softplus(w.data)
        else:
            w.data = w.data.clamp(0)
        return

class DecoderVEGACount(nn.Module):
    """
    Masked linear decoder for VEGA in SCVI mode (count data).
    
    Parameters
    ----------
    mask : np.ndarray
        Gene-gene module membership matrix
    n_cat_list : Iterable[int], optional
        List encoding number of categories for each covariate
    n_continuous_cov : int
        Number of continuous covariates
    use_batch_norm : bool
        Whether to use batch normalization
    use_layer_norm : bool
        Whether to use layer normalization
    bias : bool
        Whether to use bias parameters
    """
    def __init__(self, 
                mask: np.ndarray,
                n_cat_list: Iterable[int] = None,
                n_continuous_cov: int = 0,
                use_batch_norm: bool = False,
                use_layer_norm: bool = False,
                bias: bool = False):
        super(DecoderVEGACount, self).__init__()
        self.n_input = mask.shape[1]
        self.n_output = mask.shape[0]
        
        # Mean and dropout decoder - dropout is fully connected
        self.px_scale = SparseLayer(
            mask=mask,
            n_cat_list=n_cat_list,
            n_continuous_cov=n_continuous_cov,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            bias=bias,
            dropout_rate=0
        )
        
        self.px_dropout = SparseLayer(
            mask=mask,
            n_cat_list=n_cat_list,
            n_continuous_cov=n_continuous_cov,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            bias=bias,
            dropout_rate=0
        )

    def forward(self, dispersion: str, z: torch.Tensor, 
                library: torch.Tensor, *cat_list: int):
        """Forward pass through VEGA's decoder."""
        raw_px_scale = self.px_scale(z, *cat_list)
        px_scale = torch.softmax(raw_px_scale, dim=-1)
        px_dropout = self.px_dropout(z, *cat_list)
        px_rate = torch.exp(library) * px_scale
        px_r = None
        return px_scale, px_r, px_rate, px_dropout

class SparseLayer(nn.Module):
    """
    Sparse Layer class. Inspired by SCVI 'FCLayers' but constrained to 1 layer.
    
    Parameters
    ----------
    mask : np.ndarray
        Gene-gene module membership matrix
    n_cat_list : Iterable[int], optional
        List encoding number of categories for each covariate
    n_continuous_cov : int
        Number of continuous covariates
    use_activation : bool
        Whether to use an activation layer
    use_batch_norm : bool
        Whether to use batch normalization
    use_layer_norm : bool
        Whether to use layer normalization
    bias : bool
        Whether to use a bias parameter
    dropout_rate : float
        Dropout rate
    activation_fn : nn.Module, optional
        Activation function to use
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
                activation_fn: nn.Module = None):
        super().__init__()
        
        if n_cat_list is not None:
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []
            
        self.n_continuous_cov = n_continuous_cov
        self.cat_dim = sum(self.n_cat_list)
        
        mask_with_cov = np.vstack((
            mask, 
            np.ones((self.n_continuous_cov + self.cat_dim, mask.shape[1]))
        ))
        
        self.sparse_layer = nn.Sequential(
            CustomizedLinear(mask_with_cov),
            nn.BatchNorm1d(mask.shape[1], momentum=0.01, eps=0.001)
            if use_batch_norm else None,
            nn.LayerNorm(mask.shape[1], elementwise_affine=False)
            if use_layer_norm else None,
            activation_fn() if use_activation else None,
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None
        )

    def forward(self, x: torch.Tensor, *cat_list: int):
        """Forward computation on x for sparse layer."""
        one_hot_cat_list = []

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError("nb. categorical args provided doesn't match init. params.")
            
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat
                one_hot_cat_list += [one_hot_cat]
                
        for layer in self.sparse_layer:
            if layer is not None:
                if isinstance(layer, nn.BatchNorm1d):
                    if x.dim() == 3:
                        x = torch.cat(
                            [(layer(slice_x)).unsqueeze(0) for slice_x in x], 
                            dim=0
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
    Autograd function which masks its weights by 'mask'.
    """
    @staticmethod
    def forward(ctx, input, weight, bias=None, mask=None):
        """Forward pass of customized linear function."""
        if mask is not None:
            weight = weight * mask
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of customized linear function."""
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                grad_weight = grad_weight * mask
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask

class CustomizedLinear(nn.Module):
    """
    Extended torch.nn module which masks connection.
    
    Parameters
    ----------
    mask : np.ndarray
        Gene-gene module membership matrix
    bias : bool
        Whether to use a bias term
    """
    def __init__(self, mask, bias=True):
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
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        """Initialize parameters."""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_params_pos(self):
        """Initialize parameters with positive values only."""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(0, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input):
        """Forward pass using CustomizedLinearFunction."""
        return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        """Set extra information about the module."""
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )
