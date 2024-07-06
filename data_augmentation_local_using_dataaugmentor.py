import argparse
import os, torch
from pathlib import Path
import numpy as np

from mraugment.data_augment import DataAugmentor, AugmentationPipeline
from mraugment.data_transforms import VarNetDataTransform
from utils.model.fastmri.data.subsample import create_mask_for_mask_type
from utils.data.load_data import create_data_loaders
from utils.data.visualize import save_figure
from utils.data.transforms import DataTransform


def main(args):
    # data loader 짜기
    # data epoch 200 기준 augmented data generation pipeline 만들어 두기.
    epoch = 0
    current_epoch_fn = lambda: epoch

    if args.aug_on:
        data_augmentor = DataAugmentor(args, current_epoch_fn)
    else:
        data_augmentor = None
    train_loader = create_data_loaders(data_path = args.data_path_train,
                                       args = args,
                                       shuffle=True,
                                       isforward=False,
                                       augmentor=data_augmentor)
    # data_preprocessing = False 인 경우, mask 를 씌우지 않는다.

    for epoch in range(100,102):
        for iter, data in enumerate(train_loader):
            mask, kspace, target, _, fname, _ = data

            # dir = os.path.join(os.getcwd(), "local", fname)
            # os.makedirs(dir, exist_ok=True)
            # img = target; title = f"target_image-no_aug-iter_{iter}"; path = os.path.join(dir, title)
            # save_figure(img, title, path)
            #
            # dir = os.path.join(os.getcwd(), "local", fname)
            # os.makedirs(dir, exist_ok=True)
            # img = np.mean(np.log(np.abs(kspace + 1e-10)), axis=0); title = f"kspace_image-no_aug-iter_{iter}"; path = os.path.join(dir, title)
            # save_figure(img, title, path)




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