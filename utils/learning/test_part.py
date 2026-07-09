import h5py
import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.data.transforms import to_tensor
from utils.model.varnet import VarNet

# ---------------------------------------------------------------------------
# Team-editable reconstruction contract.
# recon_eval.py (the fixed timing harness) only calls the three functions
# below. This branch feeds `kspace` + `mask` (k-space domain) to a VarNet; a
# U-Net branch reimplements the same three functions for the image domain.
# ---------------------------------------------------------------------------
INPUT_KIND = "kspace"      # harness delivers the kspace H5 to prep_volume


def load_model(args, device):
    model = VarNet(num_cascades=args.cascade,
                   chans=args.chans,
                   sens_chans=args.sens_chans).to(device=device)
    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def prep_volume(image_path, kspace_path, device):
    """Load one volume's k-space and mask onto the host. Untimed: no model compute here."""
    with h5py.File(kspace_path, 'r') as hf:
        kspace = hf['kspace'][:]
        mask = np.array(hf['mask'])
    return {"kspace": kspace, "mask": mask, "device": device, "num_slices": kspace.shape[0]}


def recon_slice(model, ctx, s):
    """Reconstruct a single slice (batch=1). Timed by the harness."""
    device = ctx["device"]
    mask = ctx["mask"]
    kspace = to_tensor(ctx["kspace"][s] * mask)
    kspace = torch.stack((kspace.real, kspace.imag), dim=-1).unsqueeze(0).to(device=device)
    mask_t = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
    mask_t = mask_t.unsqueeze(0).to(device=device)
    return model(kspace, mask_t)[0]


def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)

    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device ', torch.cuda.current_device())

    model = load_model(args, device)

    forward_loader = create_data_loaders(data_path=args.data_path, args=args, isforward=True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)
