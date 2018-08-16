from collections import OrderedDict, Iterable
from itertools import repeat

try:
    # python 3
    from queue import Queue
except ImportError:
    # python 2
    from Queue import Queue

import torch
import torch.nn as nn
import torch.autograd as autograd

from functions import wbn


class WBN2d(nn.Module):
    """Weighted Batch Normalization"""

    def __init__(self, k, num_features, eps=1e-5, momentum=0.1, affine=False):
        """Creates an InPlace Activated Batch Normalization module

        Parameters
        ----------
	k : int
            Number of latent domains.
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        """
        super(WBN2d, self).__init__()
	self.k=k
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features,1,1))
            self.bias = nn.Parameter(torch.Tensor(num_features,1,1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
	self.wbns=nn.ModuleList()
	for i in range(k):
		self.wbns.append(WBN(self.num_features))
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, x, ws):
	 y=x*0.
	 for i,wbn_component in enumerate(self.wbns):
	    w=ws[:,i].clone()
	    s=w.sum()*x.shape[2]*x.shape[3]
	    wn=w/s
	    wr=w.view(-1, 1,1,1)
            y+=wr*wbn_component(x.clone(), wn)
	  
	 if self.affine:
         	return self.weight*y+self.bias
	 else:
		return y

    def __repr__(self):
        rep = '{name}({num_features}, eps={eps}, momentum={momentum},' \
              ' affine={affine}, activation={activation}'
        if self.activation == "leaky_relu":
            rep += ' slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)


class WBN1d(nn.Module):
    """Weighted Batch Normalization"""

    def __init__(self, k, num_features, eps=1e-5, momentum=0.1, affine=True, activation="none"):
        """Creates an InPlace Activated Batch Normalization module

        Parameters
        ----------
	k : int
            Number of latent domains.
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        """
        super(WBN1d, self).__init__()
	self.k=k
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = activation
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features,1,1))
            self.bias = nn.Parameter(torch.Tensor(num_features,1,1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
	self.wbns=nn.ModuleList()
	for i in range(k):
		self.wbns.append(WBN(self.num_features))
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, x, ws):
	 y=x*0.
	 for i,wbn_component in enumerate(self.wbns):
	    w=ws[:,i].clone()
	    s=w.sum()
	    wn=w/s
	    wr=w.view(-1, 1)
            y += wr*wbn_component(x.clone(), wn)

	 if self.affine:
         	return self.weight*y+self.bias
	 else:
		return y

    def __repr__(self):
        rep = '{name}({num_features}, eps={eps}, momentum={momentum},' \
              ' affine={affine}, activation={activation}'
        if self.activation == "leaky_relu":
            rep += ' slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)



class WBN(nn.Module):
    """Weighted Batch Normalization"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, activation="none"):
        """Creates an InPlace Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        """
        super(WBN, self).__init__()
	self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = activation
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, x, w):
        return wbn(x.clone(), w, self.weight, self.bias, autograd.Variable(self.running_mean), autograd.Variable(self.running_var), self.training, self.momentum, self.eps,self.activation)

    def __repr__(self):
        rep = '{name}({num_features}, eps={eps}, momentum={momentum},' \
              ' affine={affine}, activation={activation}'
        if self.activation == "leaky_relu":
            rep += ' slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)


