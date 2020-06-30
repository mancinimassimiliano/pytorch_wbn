import torch.autograd as autograd
import torch.cuda.comm as comm
from torch.autograd.function import once_differentiable

import _ext

# Activation names
ACT_LEAKY_RELU = "leaky_relu"
ACT_ELU = "elu"
ACT_NONE = "none"


def _check(fn, *args, **kwargs):
    success = fn(*args, **kwargs)
    if not success:
        raise RuntimeError("CUDA Error encountered in {}".format(fn))


def _broadcast_shape(x):
    out_size = []
    for i, s in enumerate(x.size()):
        if i != 1:
            out_size.append(1)
        else:
            out_size.append(s)
    return out_size


def _reduce(x):
    if len(x.size()) == 2:
        return x.sum(dim=0)
    else:
        n, c = x.size()[0:2]
        return x.contiguous().view((n, c, -1)).sum(2).sum(0)


def _count_samples(x):
    count = 1
    for i, s in enumerate(x.size()):
        if i != 1:
            count *= s
    return count

def _check_contiguous(*args):
    if not all([mod is None or mod.is_contiguous() for mod in args]):
        raise ValueError("Non-contiguous input")

class WBN(autograd.Function):
    @staticmethod
    def forward(ctx, x, w, weight, bias, running_mean, running_var,
                training=True, momentum=0.1, eps=1e-05, activation=ACT_NONE):
        # Save context
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation

        n = _count_samples(x)

        if ctx.training:
            mean = x.new().resize_as_(running_mean)
            var = x.new().resize_as_(running_var)
            _check_contiguous(x, w, mean, var)
            _check(_ext.wbn_mean_var_cuda, x, w, mean, var)

            # Update running stats
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * mean)
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * var * n / (n - 1))

        else:
            mean, var = running_mean, running_var

        _check_contiguous(x, w, mean, var, weight, bias)
        _check(_ext.wbn_forward_cuda,
               x, w, mean, var,
               weight if weight is not None else x.new(),
               bias if bias is not None else x.new(),
               x, x, ctx.eps)

        # Output
        ctx.save_for_backward(x, w, weight, bias, mean, var)
        ctx.mark_dirty(x)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, w, weight, bias, mean, var = ctx.saved_tensors
        dz = dz.contiguous()

        if ctx.needs_input_grad[0]:
            dx = dz.new().resize_as_(dz).zero_()
        else:
            dx = None
        #if ctx.needs_input_grad[1]:
        dw = dz.new().resize_as_(w).zero_()
        #else:
        #    dw = None

        if ctx.needs_input_grad[2]:
            dweight = dz.new().resize_as_(mean).zero_()
        else:
            dweight = None

        if ctx.needs_input_grad[3]:
            dbias = dz.new().resize_as_(mean).zero_()
        else:
            dbias = None

        if ctx.training:
            edz = dz.new().resize_as_(mean)
            eydz = dz.new().resize_as_(mean)
            _check_contiguous(z, dz, w, weight, bias, edz, eydz)
            _check(_ext.wbn_edz_eydz_cuda,
                   z, dz,
                   weight if weight is not None else dz.new(),
                   bias if bias is not None else dz.new(),
                   edz, eydz, ctx.eps)
        else:
            # TODO: implement CUDA backward for inference mode
            edz = dz.new().resize_as_(mean).zero_()
            eydz = dz.new().resize_as_(mean).zero_()
        _check_contiguous(dz, z, w, mean, var, weight, bias, edz, eydz, dx, dw, dweight, dbias)
        _check(_ext.wbn_backward_cuda,
               dz, z, w, mean, var,
               weight if weight is not None else dz.new(),
               bias if bias is not None else dz.new(),
               edz, eydz,
               dx if dx is not None else dz.new(),
	       dw if dw is not None else dz.new(),
               dweight if dweight is not None else dz.new(),
               dbias if dbias is not None else dz.new(),
               ctx.eps)
        #del ctx.var
	#del ctx.mean

        return dx, dw, dweight, dbias, None, None, None, None, None, None, None


wbn = WBN.apply

__all__ = ["wbn"]
