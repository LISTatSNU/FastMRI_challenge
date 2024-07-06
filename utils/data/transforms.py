import numpy as np
import torch

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
    def __call__(self, mask, kspace, target, attrs, fname, slice):
        if not torch.is_tensor(target):
            target = to_tensor(target)
        if not self.isforward:
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1

        if not torch.is_tensor(kspace) and not torch.is_tensor(mask):
            masked_kspace = to_tensor(kspace * mask)
        else:
            kspace = np.array(kspace)
            mask = np.array(mask)
            masked_kspace = to_tensor(kspace * mask)
        masked_kspace = torch.stack((masked_kspace.real, masked_kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, masked_kspace.shape[-2], 1).astype(np.float32)).byte()
        return mask, masked_kspace, target, maximum, fname, slice
