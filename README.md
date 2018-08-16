# pytorch_wbn
PyTorch implementation of Weighted Batch-Normalization layers



This is the PyTorch implementation of the Weighted Batch Normalization layers used in [Boosting Domain Adaptation by Discovering Latent Domains](http://research.mapillary.com/img/publications/CVPR18b.pdf).

## Compilation

To compile the layer, just type:

'''
sh build.sh
python build.py
'''
Notice that the layer requires the cffi package installed.

## Layer Usage 


## References

If you find this code useful, please cite:

    @inProceedings{mancini2018boosting,
	author = {Mancini, Massimilano and Porzi, Lorenzo and Rota Bul\`o, Samuel and Caputo, Barbara and Ricci, Elisa},
  	title  = {Boosting Domain Adaptation by Discovering Latent Domains},
  	booktitle = {Computer Vision and Pattern Recognition (CVPR)},
  	year      = {2018},
  	month     = {June}
    }

