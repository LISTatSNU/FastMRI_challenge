"""Fixed reconstruction + evaluation harness (do not edit for submission).

Times only the per-slice model forward pass (recon_slice) with batch=1,
after a warmup, using CUDA synchronisation. Disk I/O, warmup slices and H5
writes are excluded. SSIM_full is averaged inside the foreground mask;
SSIM_bbox inside each fastMRI+ box. The organiser re-runs this file and
utils/common/metrics.py unchanged, so timing and scoring cannot be altered
by the model code in utils/learning and utils/model.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import h5py
import torch

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from utils.common.metrics import SSIM, foreground_mask, ssim_bbox, ssim_full
from utils.common.utils import save_reconstructions
from utils.learning.test_part import INPUT_KIND, load_model, prep_volume, recon_slice

WARMUP_SLICES = 5
VOLUME_TIME_WARN_S = 60.0


def parse():
    parser = argparse.ArgumentParser(
        description='Reconstruct + evaluate in a single pass (SSIM_full / SSIM_bbox / time)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0)
    parser.add_argument('-n', '--net_name', type=Path, default='test_varnet')
    parser.add_argument('-p', '--path_data', type=Path, default='/Data/leaderboard/')
    parser.add_argument('--cascade', type=int, default=1, help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net')
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net')
    parser.add_argument('--input_key', type=str, default='kspace')
    return parser.parse_args()


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_acc(model, ssim, device, acc_dir, save_dir):
    image_dir = acc_dir / 'image'
    kspace_dir = acc_dir / 'kspace'

    full_total, full_idx = 0.0, 0
    bbox_total, bbox_idx = 0.0, 0
    slice_times = []
    slow_count = 0
    reconstructions = {}

    for img_path in sorted(image_dir.iterdir()):
        ks_path = kspace_dir / img_path.name
        with h5py.File(img_path, 'r') as hf:
            target_vol = hf['image_label'][:]
            maximum = hf.attrs['max']
            annotations = json.loads(hf.attrs.get('annotations', '{}'))

        ctx = prep_volume(img_path, ks_path if ks_path.exists() else None, device)
        n = ctx['num_slices']

        with torch.no_grad():
            for s in range(min(WARMUP_SLICES, n)):
                recon_slice(model, ctx, s)

            recon_vol = []
            vol_time = 0.0
            for s in range(n):
                _sync()
                t0 = time.perf_counter()
                out = recon_slice(model, ctx, s)
                _sync()
                dt = time.perf_counter() - t0
                slice_times.append(dt)
                vol_time += dt
                recon_vol.append(out)

        if vol_time > VOLUME_TIME_WARN_S:
            slow_count += 1

        for s in range(n):
            recon_t = recon_vol[s]
            target_t = torch.from_numpy(target_vol[s]).to(device=device)
            mask_t = torch.from_numpy(foreground_mask(target_vol[s])).to(device=device).type(torch.float)

            value = ssim_full(ssim, recon_t, target_t, mask_t, maximum)
            if value is not None:
                full_total += value
                full_idx += 1
            for box in annotations.get(str(s), []):
                value = ssim_bbox(ssim, recon_t, target_t, box, maximum)
                if value is not None:
                    bbox_total += value
                    bbox_idx += 1

        reconstructions[img_path.name] = torch.stack(recon_vol).cpu().numpy()

    save_reconstructions(reconstructions, save_dir)

    full_score = full_total / full_idx if full_idx > 0 else 0.0
    bbox_score = bbox_total / bbox_idx if bbox_idx > 0 else 0.0
    mean_ms = statistics.mean(slice_times) * 1e3 if slice_times else 0.0
    recon_time = mean_ms * len(slice_times) / 1e3
    return full_score, bbox_score, recon_time, mean_ms, slow_count, len(slice_times)


if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    assert (args.path_data / 'acc4').is_dir() and (args.path_data / 'acc8').is_dir()

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    model = load_model(args, device)
    ssim = SSIM().to(device=device)
    recon_base = '../result' / args.net_name / 'reconstructions_leaderboard'

    full4, bbox4, t4, m4, slow4, n4 = run_acc(model, ssim, device, args.path_data / 'acc4', recon_base / 'acc4')
    full8, bbox8, t8, m8, slow8, n8 = run_acc(model, ssim, device, args.path_data / 'acc8', recon_base / 'acc8')

    total_time = t4 + t8
    total_slices = n4 + n8
    total_ms = total_time * 1e3 / total_slices if total_slices > 0 else 0.0

    print("Leaderboard SSIM_full : {:.4f}".format((full4 + full8) / 2))
    print("Leaderboard SSIM_bbox : {:.4f}".format((bbox4 + bbox8) / 2))
    print("Leaderboard Recon Time : {:.2f}s ({:.1f} ms/slice)".format(total_time, total_ms))
    print("=" * 10 + " Details " + "=" * 10)
    print("SSIM_full (acc4): {:.4f}   SSIM_full (acc8): {:.4f}".format(full4, full8))
    print("SSIM_bbox (acc4): {:.4f}   SSIM_bbox (acc8): {:.4f}".format(bbox4, bbox8))
    print("Recon Time (acc4): {:.2f}s ({:.1f} ms/slice)   (acc8): {:.2f}s ({:.1f} ms/slice)".format(
        t4, m4, t8, m8))
    print("Recon Time (total): {:.2f}s".format(t4 + t8))

    slow_total = slow4 + slow8
    if slow_total > 0:
        print("[WARNING] {} volume(s) exceeded {:.0f}s of forward time "
              "(acc4: {}, acc8: {}) - reconstruction is too slow, consider optimizing.".format(
                  slow_total, VOLUME_TIME_WARN_S, slow4, slow8))
