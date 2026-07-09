import h5py
import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.unet import Unet

# ---------------------------------------------------------------------------
# Team-editable reconstruction contract.
# recon_eval.py (the fixed timing harness) only calls the three functions
# below. This branch feeds `image_input` (image domain) to a U-Net; a VarNet
# branch reimplements the same three functions for the k-space domain.
# ---------------------------------------------------------------------------
INPUT_KIND = "image"      # harness delivers the image H5 to prep_volume
INPUT_KEY = "image_input"


def load_model(args, device):
    model = Unet(in_chans=args.in_chans, out_chans=args.out_chans).to(device=device)
    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def prep_volume(image_path, kspace_path, device):
    """Load one volume's model inputs onto the device. Untimed: no model compute here."""
    with h5py.File(image_path, 'r') as hf:
        x = torch.from_numpy(hf[INPUT_KEY][:]).float().to(device=device)
    return {"input": x, "num_slices": x.shape[0]}


def recon_slice(model, ctx, s):
    """Reconstruct a single slice (batch=1). Timed by the harness."""
    x = ctx["input"][s].unsqueeze(0)
    return model(x)[0]


def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    inputs = defaultdict(dict)

    with torch.no_grad():
        for (input, _, _, fnames, slices) in data_loader:
            input = input.cuda(non_blocking=True)
            output = model(input)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                inputs[fnames[i]][int(slices[i])] = input[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in inputs:
        inputs[fname] = np.stack(
            [out for _, out in sorted(inputs[fname].items())]
        )
    return reconstructions, inputs


def forward(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device ', torch.cuda.current_device())

    model = load_model(args, device)

    forward_loader = create_data_loaders(data_path=args.data_path, args=args, isforward=True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)
