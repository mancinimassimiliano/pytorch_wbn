# pytorch_wbn

This is the PyTorch implementation of the Weighted Batch Normalization layers used in [Boosting Domain Adaptation by Discovering Latent Domains](http://research.mapillary.com/img/publications/CVPR18b.pdf). THe layer is composed by a native CUDA implementation plus a PyTorch interface.

## Installation
The only requirement for the layer is the [cffi](https://pypi.org/project/cffi/) package. It can be easily installed by just typing:
 
   pip install cffi

Before using the layer, it is necessary to compile the CUDA part. To compile it, just type:

   sh build.sh
   python build.py


## Layers Usage 
The layer is initialized as standard BatchNorm except that requires an additional parameter regarding the number of latent domains. 
As an example, considering an input x of dimension NxCxHxW where N is the batch-size, C the number of channels and H,W the spatial dimensions.
The WBN2d counterpart of a BatchNorm2d would be initialized through:

    wbn = wbn_layers.WBN2d(C, k=D) 

with D representing the number of latent domains.
Differently from a standard BatchNorm, this layer will receive an additional input, a vector `w` of shape NxD:

    out = wbn(x, w) 

`w` represents the probability that each sample belongs to each of the latent domains. Thus, it *must sum to one* in the in the latent domain dimensions.



## References

If you find this code useful, please cite:

    @inProceedings{mancini2018boosting,
	author = {Mancini, Massimilano and Porzi, Lorenzo and Rota Bul\`o, Samuel and Caputo, Barbara and Ricci, Elisa},
  	title  = {Boosting Domain Adaptation by Discovering Latent Domains},
  	booktitle = {Computer Vision and Pattern Recognition (CVPR)},
  	year      = {2018},
  	month     = {June}
    }

