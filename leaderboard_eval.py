import argparse
import glob
import json
import os
from pathlib import Path

import h5py
import torch

from utils.common.metrics import SSIM, foreground_mask, ssim_full, ssim_bbox


def forward(args, ssim, device):
    leaderboard_data = sorted(glob.glob(os.path.join(args.leaderboard_data_path, '*.h5')))
    your_data = sorted(glob.glob(os.path.join(args.your_data_path, '*.h5')))
    if len(leaderboard_data) != len(your_data):
        raise NotImplementedError(
            f'Your Data Size ({len(your_data)}) Should Match Leaderboard ({len(leaderboard_data)})'
        )

    ssim_full_total, full_idx = 0.0, 0
    ssim_bbox_total, bbox_idx = 0.0, 0

    with torch.no_grad():
        for l_fname in leaderboard_data:
            y_fname = os.path.join(args.your_data_path, os.path.basename(l_fname))
            with h5py.File(l_fname, "r") as hf:
                target_vol = hf['image_label'][:]
                maximum = hf.attrs['max']
                # bbox annotations are embedded in the eval H5 (organizer-side).
                annotations = json.loads(hf.attrs.get('annotations', '{}'))
            with h5py.File(y_fname, "r") as hf:
                recon_vol = hf[args.output_key][:]

            for i_slice in range(target_vol.shape[0]):
                target_t = torch.from_numpy(target_vol[i_slice]).to(device=device)
                recon_t = torch.from_numpy(recon_vol[i_slice]).to(device=device)
                mask_t = torch.from_numpy(foreground_mask(target_vol[i_slice])).to(device=device).type(torch.float)

                value = ssim_full(ssim, recon_t, target_t, mask_t, maximum)
                if value is not None:
                    ssim_full_total += value
                    full_idx += 1

                for box in annotations.get(str(i_slice), []):
                    value = ssim_bbox(ssim, recon_t, target_t, box, maximum)
                    if value is not None:
                        ssim_bbox_total += value
                        bbox_idx += 1

    ssim_full_score = ssim_full_total / full_idx if full_idx > 0 else 0.0
    ssim_bbox_score = ssim_bbox_total / bbox_idx if bbox_idx > 0 else 0.0
    return ssim_full_score, ssim_bbox_score


if __name__ == '__main__':
    """
    Annotation Leaderboard Evaluation.
    Reports two scores per acceleration:
      - SSIM_full: SSIM averaged only inside the foreground mask
      - SSIM_bbox: SSIM inside the annotated lesion bounding boxes only
    Bounding boxes are read from the 'annotations' attribute of each
    leaderboard H5 file (already in the 384x384 image space).
    """
    parser = argparse.ArgumentParser(
        description='FastMRI challenge Annotation Leaderboard Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-g', '--GPU_NUM', type=int, default=0)
    parser.add_argument('-lp', '--path_leaderboard_data', type=Path, default='/Data/leaderboard/')

    """
    Modify Path Below To Test Your Results
    """
    parser.add_argument('-yp', '--path_your_data', type=Path, default='../result/test_Unet/reconstructions_leaderboard/')
    parser.add_argument('-key', '--output_key', type=str, default='reconstruction')

    args = parser.parse_args()

    assert (args.path_leaderboard_data / "acc4").is_dir() and (args.path_leaderboard_data / "acc8").is_dir()

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    ssim = SSIM().to(device=device)

    # acc4
    args.leaderboard_data_path = args.path_leaderboard_data / "acc4" / 'image'
    args.your_data_path = args.path_your_data / "acc4"
    SSIM_full_acc4, SSIM_bbox_acc4 = forward(args, ssim, device)

    # acc8
    args.leaderboard_data_path = args.path_leaderboard_data / "acc8" / 'image'
    args.your_data_path = args.path_your_data / "acc8"
    SSIM_full_acc8, SSIM_bbox_acc8 = forward(args, ssim, device)

    SSIM_full = (SSIM_full_acc4 + SSIM_full_acc8) / 2
    SSIM_bbox = (SSIM_bbox_acc4 + SSIM_bbox_acc8) / 2

    print("Leaderboard SSIM_full : {:.4f}".format(SSIM_full))
    print("Leaderboard SSIM_bbox : {:.4f}".format(SSIM_bbox))
    print("=" * 10 + " Details " + "=" * 10)
    print("SSIM_full (acc4): {:.4f}   SSIM_full (acc8): {:.4f}".format(SSIM_full_acc4, SSIM_full_acc8))
    print("SSIM_bbox (acc4): {:.4f}   SSIM_bbox (acc8): {:.4f}".format(SSIM_bbox_acc4, SSIM_bbox_acc8))
