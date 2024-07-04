import argparse
import os
from pathlib import Path
import numpy as np

from mraugment.data_augment import DataAugmentor
from mraugment.data_transforms import VarNetDataTransform
from utils.model.fastmri.data.subsample import create_mask_for_mask_type
from utils.data.load_data import create_data_loaders
from utils.data.visualize import save_figure
from utils.data.transforms import DataTransform


def main(args):
    # data loader 짜기
    # data epoch 200 기준 augmented data generation pipeline 만들어 두기.
    train_loader = create_data_loaders(data_path = args.data_path_train,
                                       args = args,
                                       shuffle=True,
                                       isforward=False,
                                       data_preprocessing=args.data_preprocessing)
    # data_preprocessing = False 인 경우, mask 를 씌우지 않는다.
    transform = DataTransform(isforward=False,
                              max_key=args.max_key)

    for epoch in range(100,102):
        current_epoch_fn = lambda : epoch
        mask = create_mask_for_mask_type(
            args.mask_type, args.center_fractions, args.accelerations
        )

        augmentor = DataAugmentor(args, current_epoch_fn)
        train_transform = VarNetDataTransform(augmentor=augmentor, mask_func=mask, use_seed=False)

        for iter, data in enumerate(train_loader):
            if not args.data_preprocessing:
                mask, origin_kspace, target, attrs, fname, dataslice = data
                kspace = origin_kspace
            else:
                mask, masked_kspace, target, maximum, fname, slice = data
                kspace = masked_kspace

            # mask = np.array(mask)
            # kspace = np.array(kspace) + 1e-10
            # target = np.array(target)
            # fname = fname[0]

            if augmentor.schedule_p() > 0.0:
                aug_kspace, aug_target = augmentor(kspace, target)

            (_,
             masked_aug_kspace,
             aug_target,
             _,
             _,
             _ ) = transform(mask, aug_kspace, aug_target, attrs, fname, dataslice)
            masked_aug_kspace = masked_aug_kspace.numpy()
            aug_target = aug_target.numpy()

            #### Visualize target image
            dir = os.path.join(os.getcwd(), "local", fname)
            os.makedirs(dir, exist_ok=True)
            img = target[0]; title = f"target_image-iter_{iter}"; path = os.path.join(dir, title)
            save_figure(img, title, path)

            #### Visualize kspace image
            kspace = kspace[0]
            if args.data_preprocessing:
                real_part = kspace[..., 0]
                imaginary_part = kspace[..., 1]
                kspace = real_part + 1j * imaginary_part
            magnitude = np.log(np.abs(kspace))
            img = np.mean(magnitude, axis=0)
            title = f"kspace_image-preprocessing_{args.data_preprocessing}-iter_{iter}"
            path = os.path.join(dir, title)
            save_figure(img, title, path)
            import pdb; pdb.set_trace()



if __name__ == '__main__':
    home_dir = Path.home()
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=200, help='max epochs')
    parser = DataAugmentor.add_augmentation_specific_args(parser)
    parser.add_argument('-dp', '--data_preprocessing', action=argparse.BooleanOptionalAction, default=False, help='data preprocessing')
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-t', '--data-path-train', type=Path, default=f'{home_dir}/Data/train/', help='Directory of train data')
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="equispaced",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.04],
        type=float,
        help="Number of center lines to use in mask | 마스크를 씌우지 않을 부분의 양. 즉 LF 영역의 양",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[8],
        type=int,
        help="Acceleration rates to use for masks",
    )
    args = parser.parse_args()
    print(args.data_preprocessing)
    main(args)