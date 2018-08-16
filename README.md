# pytorch_wbn

This is the PyTorch implementation of the Weighted Batch Normalization layers used in [Boosting Domain Adaptation by Discovering Latent Domains](http://research.mapillary.com/img/publications/CVPR18b.pdf). The layer is composed by a native CUDA implementation plus a PyTorch interface.

**Note**: This code have been produced by taking inspiration from the [In-Place Activated BatchNorm](https://github.com/mapillary/inplace_abn) implementation.

## Installation
The only requirement for the layer is the [cffi](https://pypi.org/project/cffi/) package. It can be easily installed by just typing:
 
   `pip install cffi`

Before using the layer, it is necessary to compile the CUDA part. To compile it, just type:
   ```
   sh build.sh
   python build.py
   ```


## Layers
Three layers are available:
* `WBN2d` : the 2D weighted BN counterpart of [BatchNorm2d](https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d). It can be initialized specifying an additional parameter `k` denoting the number of latent domains (default 2). As input during the forward pass, for each sample in the batch it takes an additional tensor `w` denoting the probability that the sample belongs to each of the latent domains.
* `WBN1d` : the 1D weighted BN counterpart of [BatchNorm1d](https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm1d). The initialization and forward pass follows `WBN2d`.
* `WBN` : the single domain weighted batch norm. This layer is the base component for the previous two. It **does not** take `k` as additional initialization input. Moreover, the tensor `w` given as additional input has a number of element equal to the batch-size and **must sum to 1** in the batch dimension. 

## Usage 
The layer is initialized as standard BatchNorm except that requires an additional parameter regarding the number of latent domains. 
As an example, considering an input `x` of dimension NxCxHxW where N is the batch-size, C the number of channels and H, W the spatial dimensions.
The `WBN2d` counterpart of a *BatchNorm2d*  would be initialized through:

    wbn = wbn_layers.WBN2d(C, k=D) 

with D representing the desired number of latent domains.
Differently from a standard BatchNorm, this layer will receive an additional input, a vector `w` of shape NxD:

    out = wbn(x, w) 

`w` represents the probability that each sample belongs to each of the latent domains thus it **must sum to 1** in the domain dimensions (i.e. dim=1). Notice that for the `WBN` case `w` has dimension Nx1 and **must sum to 1** in the batch dimension (i.e. dim=0).



## References

If you find this code useful, please cite:

    @inProceedings{mancini2018boosting,
	author = {Mancini, Massimilano and Porzi, Lorenzo and Rota Bul\`o, Samuel and Caputo, Barbara and Ricci, Elisa},
  	title  = {Boosting Domain Adaptation by Discovering Latent Domains},
  	booktitle = {Computer Vision and Pattern Recognition (CVPR)},
  	year      = {2018},
  	month     = {June}
    }

