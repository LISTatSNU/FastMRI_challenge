"""
Model dependent data transforms that apply MRAugment to 
training data before fed to the model.
Modified from https://github.com/facebookresearch/fastMRI/blob/master/fastmri/data/transforms.py
"""
from typing import Dict, Optional, Sequence, Tuple, Union
import utils.model.fastmri
import numpy as np
import torch

from utils.model.fastmri.data.subsample import MaskFunc
from utils.model.fastmri.data.transforms import to_tensor, apply_mask

class VarNetDataTransform:
    """
    Data Transformer for training VarNet models with added MRAugment data augmentation.
    """

    def __init__(self, augmentor = None, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
        """
        Args:
            augmentor: DataAugmentor object that encompasses the MRAugment pipeline and
                schedules the augmentation probability
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed
        if augmentor is not None:
            self.use_augment = True
            self.augmentor = augmentor
        else:
            self.use_augment = False

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int, float, torch.Tensor]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.
        Returns:
            tuple containing:
                masked_kspace: k-space after applying sampling mask.
                mask: The applied sampling mask
                target: The target image (if applicable).
                fname: File name.
                slice_num: The slice index.
                max_value: Maximum image value.
                crop_size: The size to crop the final image.
        """
        # Make sure data types match
        kspace = kspace.astype(np.complex64)
        target = target.astype(np.float32)
        
        if target is not None:
            target = to_tensor(target)
            max_value = attrs["max"]
        else:
            target = torch.tensor(0)
            max_value = 0.0

        kspace = to_tensor(kspace)
        
        # Apply augmentations if needed
        if self.use_augment: 
            if self.augmentor.schedule_p() > 0.0:                
                kspace, target = self.augmentor(kspace, target.shape)
                
        # Add singleton channel dimension if singlecoil
        if len(kspace.shape) == 3:
            kspace.unsqueeze_(0)
        assert len(kspace.shape) == 4
                
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = torch.tensor([target.shape[0], target.shape[1]])

        if self.mask_func:
            masked_kspace, mask = apply_mask(
                kspace, self.mask_func, seed, (acq_start, acq_end)
            )
        else:
            masked_kspace = kspace
            shape = np.array(kspace.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask = mask.reshape(*mask_shape)
            mask[:, :, :acq_start] = 0
            mask[:, :, acq_end:] = 0
        return (
            masked_kspace,
            mask.byte(),
            target,
            fname,
            slice_num,
            max_value,
            crop_size,
        )
    
    def seed_pipeline(self, seed):
        """
        Sets random seed for the MRAugment pipeline. It is important to provide
        different seed to different workers and across different GPUs to keep
        the augmentations diverse.
        
        For an example how to set it see worker_init in pl_modules/fastmri_data_module.py
        """
        if self.use_augment:
            if self.augmentor.aug_on:
                self.augmentor.augmentation_pipeline.rng.seed(seed)