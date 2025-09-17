import torch
import torch.nn.functional as F
import math
from glob import glob
from PIL import Image
import numpy as np

# --- PSNR helper ---
def compute_psnr(x, y, max_val=1.0):
    """
    x, y: torch tensors [B,T,C,H,W] or [C,H,W] in range [0,1]
    """
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    mse = F.mse_loss(x, y, reduction='mean').item()
    if mse == 0:
        return float('inf')
    psnr = 10.0 * math.log10((max_val ** 2) / mse)
    return psnr

# --- Load a trained checkpoint ---
ckpt_path = "./checkpoints_finetune_lambda0.01/best_epoch000.pth.tar"  # adjust if different
device = "cuda" if torch.cuda.is_available() else "cpu"

from src.models.video_model import DMC as VideoModel
model = VideoModel().to(device)

ck = torch.load(ckpt_path, map_location="cpu")
sd = ck.get("state_dict", ck)
model.load_state_dict({k.replace("module.",""): v for k,v in sd.items()}, strict=False)
model.eval()

# --- Pick a few validation frames ---
val_files = sorted(glob("data/frames/val/**/*.png", recursive=True))
seq_len = 4
files = val_files[:seq_len]

imgs = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)/255.0 for p in files]
seq = np.stack(imgs, axis=0)  # [T,H,W,3]
seq = torch.from_numpy(seq).permute(0,3,1,2).unsqueeze(0).to(device)  # [B,T,C,H,W]

# --- Run encode-decode ---
from training import encode_decode_sequence
out = encode_decode_sequence(model, seq, qp=0)
x_hat = out["x_hat"].clamp(0,1)

# --- Compute PSNR ---
psnr_val = compute_psnr(seq, x_hat)
print(f"PSNR = {psnr_val:.2f} dB")

# Optionally show side by side
import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.imshow(seq[0,0].permute(1,2,0).cpu().numpy())
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(x_hat[0,0].permute(1,2,0).detach().cpu().numpy())
plt.title("Reconstruction")
plt.axis("off")
plt.show()
