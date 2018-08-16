int wbn_mean_var_cuda(const THCudaTensor *x, const THCudaTensor *w,  THCudaTensor *mean, THCudaTensor *var);
int wbn_forward_cuda(const THCudaTensor *x, const THCudaTensor *w, const THCudaTensor *mean, const THCudaTensor *var,const THCudaTensor *weight, const THCudaTensor *bias, THCudaTensor *y, THCudaTensor *z,float eps);
int wbn_edz_eydz_cuda(const THCudaTensor *z, const THCudaTensor *dz, const THCudaTensor *weight,
                     const THCudaTensor *bias, THCudaTensor *edz, THCudaTensor *eydz, float eps);
int wbn_backward_cuda(const THCudaTensor *dz, const THCudaTensor *z, const THCudaTensor *w,const THCudaTensor *mean, const THCudaTensor *var,const THCudaTensor *weight, const THCudaTensor *bias, const THCudaTensor *edz,   const THCudaTensor *eydz, THCudaTensor *dx, THCudaTensor *dw, THCudaTensor *dweight, THCudaTensor *dbias,float eps);
