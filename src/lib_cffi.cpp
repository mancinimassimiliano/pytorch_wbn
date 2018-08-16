// All functions assume that input and output tensors are already initialized
// and have the correct dimensions
#include <THC/THC.h>

// Forward definition of implementation functions
extern "C" {
int _wbn_mean_var_cuda(int N, int C, int S, const float *x, const float *w, float *mean, float *var, cudaStream_t);
int _wbn_forward_cuda(int N, int C, int S, const float *x, const float *w, const float *mean, const float *var, const float *weight,
                     const float *bias, float *y, float *z, float eps, cudaStream_t);
int _wbn_edz_eydz_cuda(int N, int C, int S, const float *z, const float *dz, const float *weight, const float *bias,
                      float *edz, float *eydz, float eps, cudaStream_t stream);
int _wbn_backward_cuda(int N, int C, int S, const float *dz, const float *z, const float *w, const float *mean, const float *var, const float *weight,
                      const float *bias, const float *edz, const float *eydz, float *dx, float *dw, float *dweight, float *dbias,
                      float eps, cudaStream_t stream);
}

extern THCState *state;

void get_sizes(const THCudaTensor *t, int *N, int *C, int *S){
  // Get sizes
  *S = 1;
  *N = THCudaTensor_size(state, t, 0);
  *C = THCudaTensor_size(state, t, 1);
  if (THCudaTensor_nDimension(state, t) > 2) {
    for (int i = 2; i < THCudaTensor_nDimension(state, t); ++i) {
      *S *= THCudaTensor_size(state, t, i);
    }
  }
}

extern "C" int wbn_mean_var_cuda(const THCudaTensor *x, const THCudaTensor *w, THCudaTensor *mean, THCudaTensor *var) {
  cudaStream_t stream = THCState_getCurrentStream(state);

  int S, N, C;
  get_sizes(x, &N, &C, &S);

  // Get pointers
  const float *x_data = THCudaTensor_data(state, x);
  const float *w_data = THCudaTensor_data(state, w);
  float *mean_data = THCudaTensor_data(state, mean);
  float *var_data = THCudaTensor_data(state, var);

  return _wbn_mean_var_cuda(N, C, S, x_data, w_data, mean_data, var_data, stream);
}

extern "C" int wbn_forward_cuda(const THCudaTensor *x, const THCudaTensor *w, const THCudaTensor *mean, const THCudaTensor *var,
                               const THCudaTensor *weight, const THCudaTensor *bias, THCudaTensor *y, THCudaTensor *z,
                               float eps) {
  cudaStream_t stream = THCState_getCurrentStream(state);

  int S, N, C;
  get_sizes(x, &N, &C, &S);

  // Get pointers
  const float *x_data = THCudaTensor_data(state, x);
  const float *w_data = THCudaTensor_data(state, w);
  const float *mean_data = THCudaTensor_data(state, mean);
  const float *var_data = THCudaTensor_data(state, var);
  const float *weight_data = THCudaTensor_nDimension(state, weight) != 0 ? THCudaTensor_data(state, weight) : 0;
  const float *bias_data = THCudaTensor_nDimension(state, bias) != 0 ? THCudaTensor_data(state, bias) : 0;
  float *y_data = THCudaTensor_data(state, y);
  float *z_data = THCudaTensor_data(state, z);

  return _wbn_forward_cuda(N, C, S, x_data, w_data, mean_data, var_data, weight_data, bias_data, y_data, z_data, eps, stream);
}

extern "C" int wbn_edz_eydz_cuda(const THCudaTensor *z, const THCudaTensor *dz, const THCudaTensor *weight,
                                const THCudaTensor *bias, THCudaTensor *edz, THCudaTensor *eydz, float eps) {
  cudaStream_t stream = THCState_getCurrentStream(state);

  int S, N, C;
  get_sizes(z, &N, &C, &S);

  // Get pointers
  const float *z_data = THCudaTensor_data(state, z);
  const float *dz_data = THCudaTensor_data(state, dz);
  const float *weight_data = THCudaTensor_nDimension(state, weight) != 0 ? THCudaTensor_data(state, weight) : 0;
  const float *bias_data = THCudaTensor_nDimension(state, bias) != 0 ? THCudaTensor_data(state, bias) : 0;
  float *edz_data = THCudaTensor_data(state, edz);
  float *eydz_data = THCudaTensor_data(state, eydz);

  return _wbn_edz_eydz_cuda(N, C, S, z_data, dz_data, weight_data, bias_data, edz_data, eydz_data, eps, stream);
}

extern "C" int wbn_backward_cuda(const THCudaTensor *dz, const THCudaTensor *z, const THCudaTensor *w, const THCudaTensor *mean, const THCudaTensor *var,
                               const THCudaTensor *weight, const THCudaTensor *bias, const THCudaTensor *edz,
                               const THCudaTensor *eydz, THCudaTensor *dx, THCudaTensor *dw, THCudaTensor *dweight,
                               THCudaTensor *dbias, float eps) {
  cudaStream_t stream = THCState_getCurrentStream(state);

  int S, N, C;
  get_sizes(dz, &N, &C, &S);

  // Get pointers
  const float *dz_data = THCudaTensor_data(state, dz);
  const float *z_data = THCudaTensor_data(state, z);
  const float *w_data = THCudaTensor_data(state, w);
  const float *mean_data = THCudaTensor_data(state, mean);
  const float *var_data = THCudaTensor_data(state, var);
  const float *weight_data = THCudaTensor_nDimension(state, weight) != 0 ? THCudaTensor_data(state, weight) : 0;
  const float *bias_data = THCudaTensor_nDimension(state, bias) != 0 ? THCudaTensor_data(state, bias) : 0;
  const float *edz_data = THCudaTensor_data(state, edz);
  const float *eydz_data = THCudaTensor_data(state, eydz);
  float *dx_data = THCudaTensor_nDimension(state, dx) != 0 ? THCudaTensor_data(state, dx) : 0;
  float *dw_data = THCudaTensor_nDimension(state, dw) != 0 ? THCudaTensor_data(state, dw) : 0;
  float *dweight_data = THCudaTensor_nDimension(state, dweight) != 0 ? THCudaTensor_data(state, dweight) : 0;
  float *dbias_data = THCudaTensor_nDimension(state, dbias) != 0 ? THCudaTensor_data(state, dbias) : 0;

  return _wbn_backward_cuda(N, C, S, dz_data, z_data, w_data, mean_data, var_data, weight_data, bias_data, edz_data, eydz_data, dx_data, dw_data,
                           dweight_data, dbias_data, eps, stream);
}
