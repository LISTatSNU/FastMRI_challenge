"""
MRAugment applies channel-by-channel random data augmentation to MRI slices.
For example usage on the fastMRI and Stanford MRI datasets check out the scripts
in mraugment_examples.
"""
import numpy as np
from math import exp
import torch
import torchvision.transforms.functional as TF
from mraugment.helpers import complex_crop_if_needed, crop_if_needed, complex_channel_first, complex_channel_last
from utils.model.fastmri.data import transforms as T
from utils.model.fastmri import fft2c, ifft2c, rss_complex, complex_abs

class AugmentationPipeline:
    """
    Describes the transformations applied to MRI data and handles
    augmentation probabilities including generating random parameters for 
    each augmentation.
    """
    def __init__(self, hparams):
        self.hparams = hparams
        self.weight_dict ={
                      'translation': hparams.aug_weight_translation,
                      'rotation': hparams.aug_weight_rotation,
                      'scaling': hparams.aug_weight_scaling,
                      'shearing': hparams.aug_weight_shearing,
                      'rot90': hparams.aug_weight_rot90,
                      'fliph': hparams.aug_weight_fliph,
                      'flipv': hparams.aug_weight_flipv
        }
        self.upsample_augment = hparams.aug_upsample
        self.upsample_factor = hparams.aug_upsample_factor
        self.upsample_order = hparams.aug_upsample_order
        self.transform_order = hparams.aug_interpolation_order
        self.augmentation_strength = 0.0
        self.rng = np.random.RandomState()

    def augment_image(self, im, max_output_size=None):
        # Trailing dims must be image height and width (for torchvision) 
        im = complex_channel_first(im)
        
        # ---------------------------  
        # pixel preserving transforms
        # ---------------------------  
        # Horizontal flip
        if self.random_apply('fliph'):
            im = TF.hflip(im)

        # Vertical flip 
        if self.random_apply('flipv'):
            im = TF.vflip(im)

        # Rotation by multiples of 90 deg 
        if self.random_apply('rot90'):
            k = self.rng.randint(0,2)
            im = torch.rot90(im, k, dims=[-2, -1])

        # Translation by integer number of pixels
        if self.random_apply('translation'):
            h, w = im.shape[-2:]
            t_x = self.rng.uniform(-self.hparams.aug_max_translation_x, self.hparams.aug_max_translation_x)
            t_x = int(t_x * h)
            t_y = self.rng.uniform(-self.hparams.aug_max_translation_y, self.hparams.aug_max_translation_y)
            t_y = int(t_y * w)
            
            pad, top, left = self._get_translate_padding_and_crop(im, (t_x, t_y))
            im = TF.pad(im, padding=pad, padding_mode='reflect')
            im = TF.crop(im, top, left, h, w)

        # ------------------------       
        # interpolating transforms
        # ------------------------  
        interp = False 

        # Rotation
        if self.random_apply('rotation'):
            interp = True
            rot = self.rng.uniform(-self.hparams.aug_max_rotation, self.hparams.aug_max_rotation)
        else:
            rot = 0.

        # Shearing
        if self.random_apply('shearing'):
            interp = True
            shear_x = self.rng.uniform(-self.hparams.aug_max_shearing_x, self.hparams.aug_max_shearing_x)
            shear_y = self.rng.uniform(-self.hparams.aug_max_shearing_y, self.hparams.aug_max_shearing_y)
        else:
            shear_x, shear_y = 0., 0.

        # Scaling
        if self.random_apply('scaling'):
            interp = True
            scale = self.rng.uniform(1-self.hparams.aug_max_scaling, 1 + self.hparams.aug_max_scaling)
        else:
            scale = 1.

        # Upsample if needed
        upsample = interp and self.upsample_augment
        if upsample:
            upsampled_shape = [im.shape[-2] * self.upsample_factor, im.shape[-1] * self.upsample_factor]
            original_shape = im.shape[-2:]
            interpolation  = TF.InterpolationMode.BICUBIC if self.upsample_order == 3 else TF.InterpolationMode.BILINEAR
            im = TF.resize(im, size=upsampled_shape, interpolation=interpolation)

        # Apply interpolating transformations 
        # Affine transform - if any of the affine transforms is randomly picked
        if interp:
            h, w = im.shape[-2:]
            pad = self._get_affine_padding_size(im, rot, scale, (shear_x, shear_y))
            im = TF.pad(im, padding=pad, padding_mode='reflect')
            im = TF.affine(im,
                           angle=rot,
                           scale=scale,
                           shear=(shear_x, shear_y),
                           translate=[0, 0],
                           interpolation=TF.InterpolationMode.BILINEAR
                          )
            im = TF.center_crop(im, (h, w))
        
        # ---------------------------------------------------------------------
        # Apply additional interpolating augmentations here before downsampling
        # ---------------------------------------------------------------------
        
        # Downsampling
        if upsample:
            im = TF.resize(im, size=original_shape, interpolation=interpolation)
        
        # Final cropping if augmented image is too large
        if max_output_size is not None:
            im = crop_if_needed(im, max_output_size)
            
        # Reset original channel ordering
        im = complex_channel_last(im)
        
        return im
    
    def augment_from_kspace(self, kspace, target_size, max_train_size=None):       
        im = ifft2c(kspace) 
        im = self.augment_image(im, max_output_size=max_train_size)
        target = self.im_to_target(im, target_size)
        kspace = fft2c(im)
        return kspace, target
    
    def im_to_target(self, im, target_size):     
        # Make sure target fits in the augmented image
        cropped_size = [min(im.shape[-3], target_size[0]), 
                        min(im.shape[-2], target_size[1])]
        
        if len(im.shape) == 3: 
            # Single-coil
            target = complex_abs(T.complex_center_crop(im, cropped_size))
        else:
            # Multi-coil
            assert len(im.shape) == 4
            target = T.center_crop(rss_complex(im), cropped_size)
        return target  
            
    def random_apply(self, transform_name):
        if self.rng.uniform() < self.weight_dict[transform_name] * self.augmentation_strength:
            return True
        else: 
            return False
        
    def set_augmentation_strength(self, p):
        self.augmentation_strength = p

    @staticmethod
    def _get_affine_padding_size(im, angle, scale, shear):
        """
        Calculates the necessary padding size before applying the 
        general affine transformation. The output image size is determined based on the 
        input image size and the affine transformation matrix.
        """
        h, w = im.shape[-2:]
        corners = [
            [-h/2, -w/2, 1.],
            [-h/2, w/2, 1.], 
            [h/2, w/2, 1.], 
            [h/2, -w/2, 1.]
        ]
        mx = torch.tensor(TF._get_inverse_affine_matrix([0.0, 0.0], -angle, [0, 0], scale, [-s for s in shear])).reshape(2,3)
        corners = torch.cat([torch.tensor(c).reshape(3,1) for c in corners], dim=1)
        tr_corners = torch.matmul(mx, corners)
        all_corners = torch.cat([tr_corners, corners[:2, :]], dim=1)
        bounding_box = all_corners.amax(dim=1) - all_corners.amin(dim=1)
        px = torch.clip(torch.floor((bounding_box[0] - h) / 2), min=0.0, max=h-1) 
        py = torch.clip(torch.floor((bounding_box[1] - w) / 2),  min=0.0, max=w-1)
        return int(py.item()), int(px.item())

    @staticmethod
    def _get_translate_padding_and_crop(im, translation):
        t_x, t_y = translation
        h, w = im.shape[-2:]
        pad = [0, 0, 0, 0]
        if t_x >= 0:
            pad[3] = min(t_x, h - 1) # pad bottom
            top = pad[3]
        else:
            pad[1] = min(-t_x, h - 1) # pad top
            top = 0
        if t_y >= 0:
            pad[0] = min(t_y, w - 1) # pad left
            left = 0
        else:
            pad[2] = min(-t_y, w - 1) # pad right
            left = pad[2]
        return pad, top, left

            
class DataAugmentor:
    """
    High-level class encompassing the augmentation pipeline and augmentation
    probability scheduling. A DataAugmentor instance can be initialized in the 
    main training code and passed to the DataTransform to be applied 
    to the training data.
    """
        
    def __init__(self, hparams, current_epoch_fn):
        """
        hparams: refer to the arguments below in add_augmentation_specific_args
        current_epoch_fn: this function has to return the current epoch as an integer 
        and is used to schedule the augmentation probability.
        """
        self.current_epoch_fn = current_epoch_fn
        self.hparams = hparams
        self.aug_on = hparams.aug_on
        if self.aug_on:
            self.augmentation_pipeline = AugmentationPipeline(hparams)
        self.max_train_resolution = hparams.max_train_resolution
        
    def __call__(self, kspace, target_size):
        """
        Generates augmented kspace and corresponding augmented target pair.
        kspace: torch tensor of shape [C, H, W, 2] (multi-coil) or [H, W, 2]
            where last dim is for real/imaginary channels
        target_size: [H, W] shape of the generated augmented target
        """
        # Set augmentation probability
        if self.aug_on:
            p = self.schedule_p()
            self.augmentation_pipeline.set_augmentation_strength(p)
        else:
            p = 0.0
        
        # Augment if needed
        if self.aug_on and p > 0.0:
            kspace, target = self.augmentation_pipeline.augment_from_kspace(kspace,
                                                                          target_size=target_size,
                                                                          max_train_size=self.max_train_resolution)
        else:
            # Crop in image space if image is too large
            target = fft2c(kspace)

        return kspace, target
        
    def schedule_p(self):
        D = self.hparams.aug_delay
        T = self.hparams.max_epochs
        t = self.current_epoch_fn()
        p_max = self.hparams.aug_strength

        if t < D:
            return 0.0
        else:
            if self.hparams.aug_schedule == 'constant':
                p = p_max
            elif self.hparams.aug_schedule == 'ramp':
                p = (t-D)/(T-D) * p_max
            elif self.hparams.aug_schedule == 'exp':
                c = self.hparams.aug_exp_decay/(T-D) # Decay coefficient
                p = p_max/(1-exp(-(T-D)*c))*(1-exp(-(t-D)*c))
            return p

        
    def add_augmentation_specific_args(parser):
        parser.add_argument(
            '--aug_on', 
            default=False,
            help='This switch turns data augmentation on.',
            action='store_true'
        )
        # --------------------------------------------
        # Related to augmentation strenght scheduling
        # --------------------------------------------
        parser.add_argument(
            '--aug_schedule', 
            type=str, 
            default='exp',
            help='Type of data augmentation strength scheduling. Options: constant, ramp, exp'
        )
        parser.add_argument(
            '--aug_delay', 
            type=int, 
            default=0,
            help='Number of epochs at the beginning of training without data augmentation. The schedule in --aug_schedule will be adjusted so that at the last epoch the augmentation strength is --aug_strength.'
        )
        parser.add_argument(
            '--aug_strength', 
            type=float, 
            default=0.0, 
            help='Augmentation strength, combined with --aug_schedule determines the augmentation strength in each epoch'
        )
        parser.add_argument(
            '--aug_exp_decay', 
            type=float, 
            default=5.0, 
            help='Exponential decay coefficient if --aug_schedule is set to exp. 1.0 is close to linear, 10.0 is close to step function'
        )

        # --------------------------------------------
        # Related to interpolation 
        # --------------------------------------------
        parser.add_argument(
            '--aug_interpolation_order', 
            type=int, 
            default=1,
            help='Order of interpolation filter used in data augmentation, 1: bilinear, 3:bicubic. Bicubic is not supported yet.'
        )
        parser.add_argument(
            '--aug_upsample', 
            default=False,
            action='store_true',
            help='Set to upsample before augmentation to avoid aliasing artifacts. Adds heavy extra computation.',
        )
        parser.add_argument(
            '--aug_upsample_factor', 
            type=int, 
            default=2,
            help='Factor of upsampling before augmentation, if --aug_upsample is set'
        )
        parser.add_argument(
            '--aug_upsample_order', 
            type=int, 
            default=1,
            help='Order of upsampling filter before augmentation, 1: bilinear, 3:bicubic'
        )

        # --------------------------------------------
        # Related to transformation probability weights
        # --------------------------------------------
        parser.add_argument(
            '--aug_weight_translation', 
            type=float, 
            default=1.0, 
            help='Weight of translation probability. Augmentation probability will be multiplied by this constant'
        )
        parser.add_argument(
            '--aug_weight_rotation', 
            type=float, 
            default=1.0, 
            help='Weight of arbitrary rotation probability. Augmentation probability will be multiplied by this constant'
        )  
        parser.add_argument(
            '--aug_weight_shearing', 
            type=float,
            default=1.0, 
            help='Weight of shearing probability. Augmentation probability will be multiplied by this constant'
        )
        parser.add_argument(
            '--aug_weight_scaling', 
            type=float, 
            default=1.0, 
            help='Weight of scaling probability. Augmentation probability will be multiplied by this constant'
        )
        parser.add_argument(
            '--aug_weight_rot90', 
            type=float, 
            default=1.0, 
            help='Weight of probability of rotation by multiples of 90 degrees. Augmentation probability will be multiplied by this constant'
        )  
        parser.add_argument(
            '--aug_weight_fliph', 
            type=float,
            default=1.0, 
            help='Weight of horizontal flip probability. Augmentation probability will be multiplied by this constant'
        )
        parser.add_argument(
            '--aug_weight_flipv',
            type=float,
            default=1.0, 
            help='Weight of vertical flip probability. Augmentation probability will be multiplied by this constant'
        ) 

        # --------------------------------------------
        # Related to transformation limits
        # --------------------------------------------
        parser.add_argument(
            '--aug_max_translation_x', 
            type=float,
            default=0.125, 
            help='Maximum translation applied along the x axis as fraction of image width'
        )
        parser.add_argument(
            '--aug_max_translation_y',
            type=float, 
            default=0.125, 
            help='Maximum translation applied along the y axis as fraction of image height'
        )
        parser.add_argument(
            '--aug_max_rotation', 
            type=float, 
            default=180., 
            help='Maximum rotation applied in either clockwise or counter-clockwise direction in degrees.'
        )
        parser.add_argument(
            '--aug_max_shearing_x', 
            type=float, 
            default=15.0, 
            help='Maximum shearing applied in either positive or negative direction in degrees along x axis.'
        )
        parser.add_argument(
            '--aug_max_shearing_y', 
            type=float, 
            default=15.0, 
            help='Maximum shearing applied in either positive or negative direction in degrees along y axis.'
        )
        parser.add_argument(
            '--aug_max_scaling', 
            type=float, 
            default=0.25, 
            help='Maximum scaling applied as fraction of image dimensions. If set to s, a scaling factor between 1.0-s and 1.0+s will be applied.'
        )
        
        #---------------------------------------------------
        # Additional arguments not specific to augmentations 
        #---------------------------------------------------
        parser.add_argument(
            "--max_train_resolution",
            nargs="+",
            default=None,
            type=int,
            help="If given, training slices will be center cropped to this size if larger along any dimension.",
        )
        return parser