"""
Model dependent data transforms that apply MRAugment to 
training data before fed to the model.
Modified from https://github.com/facebookresearch/fastMRI/blob/master/fastmri/data/transforms.py
"""
from typing import Dict, Optional, Sequence, Tuple, Union
import fastmri
import numpy as np
import torch

from fastmri.data.subsample import MaskFunc
from fastmri.data.transforms import to_tensor, tensor_to_complex_np, apply_mask
    
    
class VarNetDataTransform:
    """
    Data Transformer for training VarNet models with added MRAugment data augmentation.
    """

    def __init__(self, isforward, max_key, augmentor = None, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
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
        self.isforward = isforward
        self.max_key = max_key
        self.mask_func = mask_func
        self.use_seed = use_seed
        if augmentor is not None:
            self.use_augment = True
            self.augmentor = augmentor
        else:
            self.use_augment = False

    def __call__(
        self, mask, input, target, attrs, fname, slice
    ):
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
        
        input = input.astype(np.complex64)
        target = target.astype(np.float32)
        
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
            
        if self.use_augment: 
            kspace = to_tensor(input)
            if self.augmentor.schedule_p() > 0.0:                
                 kspace, target = self.augmentor(kspace, target.shape)
            input = tensor_to_complex_np(kspace)
        
        kspace = to_tensor(input * mask)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        
        return mask, kspace, target, maximum, fname, slice
    
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