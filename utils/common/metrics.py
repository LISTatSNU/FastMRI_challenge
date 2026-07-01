"""Scoring core shared by self-evaluation (val) and the leaderboard.

Two metrics are reported for the challenge:
  - SSIM_full: SSIM averaged only inside the foreground mask
  - SSIM_bbox: SSIM inside each fastMRI+ lesion bounding box

Bounding boxes live in the `annotations` attribute of each image H5
(384 x 384 image space). `data_range` is the volume `max` attribute.
Participants can score their own validation reconstructions with the same
functions used on the leaderboard.
"""

import cv2
import numpy as np
import torch.nn.functional as F

from utils.common.loss_function import SSIMLoss


class SSIM(SSIMLoss):
    """SSIMLoss that returns the per-pixel SSIM map for a single 2D image."""

    def forward(self, X, Y, data_range):
        if X.dim() != 2 or Y.dim() != 2:
            raise ValueError("SSIM expects 2D (H, W) inputs")
        X = X.unsqueeze(0).unsqueeze(0)
        Y = Y.unsqueeze(0).unsqueeze(0)
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        S = (A1 * A2) / (B1 * B2)
        return S[0, 0]


def foreground_mask(target):
    """Binary foreground mask for a 2D image, matching the leaderboard."""
    mask = np.zeros(target.shape)
    mask[target > 2e-5] = 1
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=15)
    mask = cv2.erode(mask, kernel, iterations=14)
    return mask


def ssim_full(ssim, recon_t, target_t, mask_t, data_range):
    """SSIM averaged only inside the foreground mask.

    Returns None when the mask is empty so the caller can skip the slice.
    """
    ssim_map = ssim(recon_t * mask_t, target_t * mask_t, data_range)
    pad = ssim.win_size // 2
    mask_valid = mask_t[pad:mask_t.shape[0] - pad, pad:mask_t.shape[1] - pad]
    denom = mask_valid.sum()
    if denom <= 0:
        return None
    return ((ssim_map * mask_valid).sum() / denom).item()


def ssim_bbox(ssim, recon_t, target_t, box, data_range):
    """SSIM inside a single annotation box. Returns None if the box is too small."""
    win = ssim.win_size
    x0, y0 = max(0, box["x"]), max(0, box["y"])
    x1 = min(target_t.shape[1], box["x"] + box["width"])
    y1 = min(target_t.shape[0], box["y"] + box["height"])
    if (x1 - x0) < win or (y1 - y0) < win:
        return None
    recon_crop = recon_t[y0:y1, x0:x1]
    target_crop = target_t[y0:y1, x0:x1]
    return ssim(recon_crop, target_crop, data_range).mean().item()
