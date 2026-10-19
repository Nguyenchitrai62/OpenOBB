"""FrozenBatchNorm2d copied from RT-DETR / DEIM (Apache-2.0), originally from DETR (Apache-2.0)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


FROZEN_BN_KERNEL = True   # False: the pre-F112 elementwise expression (reference / A-B)


class FrozenBatchNorm2d(nn.Module):
    """copy and modified from https://github.com/facebookresearch/detr/blob/master/models/backbone.py
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """
    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        n = num_features
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps
        self.num_features = n

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        if FROZEN_BN_KERNEL:
            # one fused inference-BN kernel (cuDNN on a GPU), output in the input's dtype: the expression below makes
            # 3+ elementwise passes and turns bf16 activations into fp32 (bf16 x fp32 buffer), which the next conv
            # casts back; that doubled the trunk's memory traffic in training (F112). Same values to float rounding.
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}".format(**self.__dict__)
        )

def freeze_batch_norm2d(module: nn.Module) -> nn.Module:
    if isinstance(module, nn.BatchNorm2d):
        module = FrozenBatchNorm2d(module.num_features)
    else:
        for name, child in module.named_children():
            _child = freeze_batch_norm2d(child)
            if _child is not child:
                setattr(module, name, _child)
    return module


def unlock_all(model, momentum=0.03):
    """Freeze nothing anywhere in `model` (as the usual one-stage trainer trains every layer): each FrozenBatchNorm2d
    becomes a BatchNorm2d with the same weight / bias / running statistics / eps (identical eval output; train mode then
    normalises with batch statistics and the running ones move with `momentum`), and every parameter gets requires_grad
    (e.g. the trunk stem frozen by freeze_at). Call before loading weights or resuming: the loaders fill the new layers by
    name. -> (number of converted norm layers, number of tensors made trainable)."""
    count = [0]

    def convert(m):
        for name, child in m.named_children():
            if isinstance(child, FrozenBatchNorm2d):
                bn = nn.BatchNorm2d(child.num_features, eps=child.eps, momentum=momentum).to(child.weight.device)
                with torch.no_grad():
                    bn.weight.copy_(child.weight)
                    bn.bias.copy_(child.bias)
                    bn.running_mean.copy_(child.running_mean)
                    bn.running_var.copy_(child.running_var)
                setattr(m, name, bn)
                count[0] += 1
            else:
                convert(child)
    convert(model)
    thawed = 0
    for p in model.parameters():
        if not p.requires_grad:
            p.requires_grad_(True)
            thawed += 1
    return count[0], thawed
